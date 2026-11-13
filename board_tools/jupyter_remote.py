"""
Remote control of the board's classic Jupyter Notebook server (Tornado, cookie+XSRF auth, password "xilinx") through
its REST API and the kernel-channels websocket:
  - login()               -- authenticate, return (session, xsrf_token)
  - upload_file()          -- PUT raw bytes to /api/contents/<path>
  - restart_kernel()       -- POST /api/kernels/<id>/restart
  - execute_in_kernel()    -- run code in a kernel via the channels websocket, Jupyter messaging protocol v5.3

Deploy sequence (deploy_and_verify.deploy_new_bitstream() runs it):
  1. list_sessions() to find "start nameserver.ipynb"'s kernel id (it changes on every restart).
  2. upload_file() the new .bit and .hwh.
  3. update_notebook_cell() to point the bitfile-load cell at the new file; put_notebook() to save it.
  4. restart_kernel().
  5. execute_in_kernel() the server cell without waiting for idle (it runs the server loop and never returns); confirm
     it went "busy" with no immediate error.
  6. verify with deploy_and_verify.verify_deployment() (fresh Pyro4 connection, banner check), with retries.
If the kernel does not come back cleanly within a few minutes, a full board reboot may be needed (PYNQ can serve
stale cached overlay metadata after a clean-looking restart).
"""
import base64
import json
import re
import time
import uuid
from datetime import datetime, timezone

import requests
import websocket

# Board A by default; set QICK_NS_HOST=10.1.4.105 for board B (same as calib.py).
BOARD_HOST = __import__("os").environ.get("QICK_NS_HOST", "10.1.9.26")
JUPYTER_PORT = 9090
JUPYTER_PASSWORD = __import__("os").environ.get("PYNQ_JUPYTER_PASSWORD", "xilinx")  # PYNQ default
BASE = f"http://{BOARD_HOST}:{JUPYTER_PORT}"
WS_BASE = f"ws://{BOARD_HOST}:{JUPYTER_PORT}"


def login(password=JUPYTER_PASSWORD):
    """Authenticate against the classic Jupyter Notebook server. Returns
    (requests.Session, xsrf_token) -- pass both to every other function
    here."""
    s = requests.Session()
    r = s.get(f"{BASE}/login")
    r.raise_for_status()
    m = re.search(r'name="_xsrf" value="([^"]+)"', r.text)
    xsrf = m.group(1) if m else s.cookies.get("_xsrf")
    if not xsrf:
        raise RuntimeError("Could not obtain _xsrf token from /login page")

    r2 = s.post(
        f"{BASE}/login",
        data={"_xsrf": xsrf, "password": password},
        headers={"Referer": f"{BASE}/login"},
    )
    r2.raise_for_status()
    if "/login" in r2.url:
        raise RuntimeError("Login appears to have failed (redirected back to /login)")
    return s, xsrf


def _headers(xsrf):
    return {"X-XSRFToken": xsrf}


def list_sessions(session, xsrf):
    r = session.get(f"{BASE}/api/sessions", headers=_headers(xsrf))
    r.raise_for_status()
    return r.json()


def list_contents(session, xsrf, path=""):
    r = session.get(f"{BASE}/api/contents/{path}", headers=_headers(xsrf))
    r.raise_for_status()
    return r.json()


def upload_file(session, xsrf, local_path, board_path, chunk_mb=8):
    """Upload a local file (binary-safe) to the board's Jupyter filesystem
    at board_path (relative to the notebook server's root, i.e. same
    level as 'start nameserver.ipynb'). Uses base64 content + format for
    binary safety (works for .bit/.hwh alike). Large files (.bit is
    ~34MB) are sent in one PUT -- classic notebook's contents API doesn't
    require chunking the way JupyterLab's newer large-file API does, but
    if this proves unreliable for 34MB in one shot, chunk_mb is reserved
    for a future chunked-checkpoint variant."""
    with open(local_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    payload = {"type": "file", "format": "base64", "content": b64}
    r = session.put(
        f"{BASE}/api/contents/{board_path}",
        headers=_headers(xsrf),
        json=payload,
        timeout=300,
    )
    r.raise_for_status()
    result = r.json()
    print(f"Uploaded {local_path} -> {board_path} ({len(raw)} bytes, "
          f"server reports size={result.get('size')})")
    return result


def get_notebook(session, xsrf, nb_path):
    r = session.get(f"{BASE}/api/contents/{nb_path}", headers=_headers(xsrf))
    r.raise_for_status()
    return r.json()  # ['content'] is the full nbformat JSON


def put_notebook(session, xsrf, nb_path, notebook_content):
    payload = {"type": "notebook", "format": "json", "content": notebook_content}
    r = session.put(
        f"{BASE}/api/contents/{nb_path}", headers=_headers(xsrf), json=payload
    )
    r.raise_for_status()
    return r.json()


def restart_kernel(session, xsrf, kernel_id):
    r = session.post(
        f"{BASE}/api/kernels/{kernel_id}/restart", headers=_headers(xsrf), json={}
    )
    r.raise_for_status()
    return r.json()


def start_kernel(session, xsrf, kernel_name="python3"):
    r = session.post(
        f"{BASE}/api/kernels", headers=_headers(xsrf), json={"name": kernel_name}
    )
    r.raise_for_status()
    return r.json()


def shutdown_kernel(session, xsrf, kernel_id):
    r = session.delete(f"{BASE}/api/kernels/{kernel_id}", headers=_headers(xsrf))
    r.raise_for_status()


def _jupyter_msg(msg_type, content, session_id):
    msg_id = str(uuid.uuid4())
    return {
        "header": {
            "msg_id": msg_id,
            "username": "qick-nn",
            "session": session_id,
            "msg_type": msg_type,
            "version": "5.3",
            "date": datetime.now(timezone.utc).isoformat(),
        },
        "parent_header": {},
        "metadata": {},
        "content": content,
        "channel": "shell",
    }, msg_id


def execute_in_kernel(session, xsrf, kernel_id, code, wait_for_idle=True,
                       timeout_s=60, print_output=True):
    """Run `code` in kernel_id via the kernel channels websocket (raw
    Jupyter messaging protocol v5.3 -- classic notebook has no simpler
    execute-and-wait REST endpoint).

    If wait_for_idle=True (default): blocks until the kernel reports
    status idle for this request (execution finished), collecting
    stdout/stderr/error content along the way, and returns
    {"ok": bool, "stdout": str, "error": dict|None}.

    If wait_for_idle=False: fires the execute_request, waits just long
    enough to see it acknowledged as "busy" (confirms the kernel accepted
    it) or an immediate error, then returns without waiting for
    completion -- use this for a cell that's expected to run forever
    (e.g. daemon.requestLoop()).
    """
    cookies = "; ".join(f"{k}={v}" for k, v in session.cookies.get_dict().items())
    ws_session_id = str(uuid.uuid4())
    ws_url = f"{WS_BASE}/api/kernels/{kernel_id}/channels?session_id={ws_session_id}"

    ws = websocket.create_connection(
        ws_url,
        header=[f"Cookie: {cookies}", f"X-XSRFToken: {xsrf}"],
        timeout=timeout_s,
    )
    try:
        msg, msg_id = _jupyter_msg(
            "execute_request",
            {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            ws_session_id,
        )
        ws.send(json.dumps(msg))

        stdout_parts = []
        error_content = None
        seen_busy = False
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            m = json.loads(raw)
            parent_id = m.get("parent_header", {}).get("msg_id")
            if parent_id != msg_id:
                continue  # message for a different request, ignore

            mtype = m.get("msg_type") or m.get("header", {}).get("msg_type")
            content = m.get("content", {})

            if mtype == "status" and content.get("execution_state") == "busy":
                seen_busy = True
                if not wait_for_idle:
                    break
            elif mtype == "stream":
                stdout_parts.append(content.get("text", ""))
                if print_output:
                    print(content.get("text", ""), end="")
            elif mtype == "error":
                error_content = content
                if print_output:
                    print(f"\n*** KERNEL ERROR: {content.get('ename')}: "
                          f"{content.get('evalue')} ***")
            elif mtype == "status" and content.get("execution_state") == "idle":
                if wait_for_idle:
                    break

        return {
            "ok": error_content is None,
            "seen_busy": seen_busy,
            "stdout": "".join(stdout_parts),
            "error": error_content,
        }
    finally:
        ws.close()


if __name__ == "__main__":
    print("Self-test: login + list contents/sessions (read-only, safe)")
    s, xsrf = login()
    print("Login OK")
    sessions = list_sessions(s, xsrf)
    print(f"Active sessions: {len(sessions)}")
    for sess in sessions:
        print(f"  {sess['path']}  kernel={sess['kernel']['id']} "
              f"state={sess['kernel']['execution_state']}")
