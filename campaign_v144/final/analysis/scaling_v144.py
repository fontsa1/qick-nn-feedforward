"""Runs scaling_study/resource_scaling.py on the v1.44 routed reports; writes scaling_study/resource_scaling_v144.json."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "paper_tables" / "table3"))
import v141_reports as R  # noqa: E402

D = ROOT / "paper_tables" / "table3" / "data"
R.HIER, R.FLAT, R.TIMING = (D / f"v1.44_routed_{n}.rpt" for n in ("util_hier", "utilization", "timing"))
os.environ["SCALING_OUT"] = "resource_scaling_v144.json"
runpy.run_path(str(ROOT / "scaling_study" / "resource_scaling.py"), run_name="__main__")
