"""Board-side QICK server for the v1.44 build (runs on the RFSoC 4x2 under PYNQ). Same code as cell 0 of
start_nameserver.ipynb: QICK metadata patches for the added blocks, the NNQickSoc class, and the Pyro server.
Set QICK_NS_HOST to the board's address and QICK_BITFILE to the bitstream (its .hwh must sit next to it)."""
#from qick.pyro import start_nameserver
#start_nameserver(ns_port=8888)

from pathlib import Path
import subprocess
import time
from qick.pyro import start_server
from qick import QickSoc
from pynq import MMIO


# --- axis_broadcaster patch ---
# QickMetadata.trace_dma (qick/ip.py) walks the block design downstream of axis_avg_buffer_0's m1_axis expecting
# only axi_dma or axis_switch. An axis_broadcaster on that port also feeds NN_axi_0; the DMA path continues on its
# M00_AXIS. This lets the tracer pass through an axis_broadcaster via M00_AXIS, so acquire_decimated()'s
# auto-discovery works unmodified.
import qick.ip as _qickip

def _bd2_trace_dma(self, direction, start_block, start_port):
    if direction == 'forward':
        switch_common = "M00_AXIS"
    elif direction == 'backward':
        switch_common = "S00_AXIS"
    else:
        raise RuntimeError("trace_dma direction must be 'forward' or 'backward', got %s" % (direction))
    block, port = start_block, start_port
    while True:
        ((nblock, nport),) = self.trace_bus(block, port)
        blocktype = self.mod2type(nblock)
        if blocktype == 'axi_dma':
            return nblock, None, None
        elif blocktype == 'axis_switch':
            switch_path = nblock
            switch_ch = int(nport.split('_')[0][1:])
            ((nblock2, nport2),) = self.trace_bus(nblock, switch_common)
            blocktype2 = self.mod2type(nblock2)
            if blocktype2 != 'axi_dma':
                raise RuntimeError("tracing port %s from block %s: expected to find axi_dma after axis_switch, found %s instead" % (start_port, start_block, blocktype2))
            return nblock2, switch_path, switch_ch
        elif blocktype == 'axis_broadcaster':
            block, port = nblock, "M00_AXIS"
            continue
        else:
            raise RuntimeError("tracing port %s from block %s: expected to find axi_dma or axis_switch, found %s instead" % (start_port, start_block, blocktype))

_qickip.QickMetadata.trace_dma = _bd2_trace_dma
print("BD 2.0 patch applied: QickMetadata.trace_dma now passes through axis_broadcaster nodes")
# --- end axis_broadcaster patch ---

# --- NN_live tracer patch: hide live_gate_0 from QICK's net tracing; it shares trig_0_o with axis_avg_buffer_0.
_orig_trace_net = _qickip.QickMetadata._trace_net

def _live_trace_net(self, parser, blockname, portname):
    return [x for x in _orig_trace_net(self, parser, blockname, portname)
            if not (len(x) == 2 and x[0] == 'live_gate_0')]

_qickip.QickMetadata._trace_net = _live_trace_net
print("NN_live patch applied: QickMetadata._trace_net ignores live_gate_0")
# --- end NN_live tracer patch ---
# --- dac_gate tracer patch: trace_forward passes through dac_gate_0 (gen 0 -> register slice -> dac_gate_0 ->
# RFdc) like a register slice.
_orig_mod2type = _qickip.QickMetadata.mod2type

def _gate_mod2type(self, block):
    t = _orig_mod2type(self, block)
    return 'axis_register_slice_nb' if (block == 'dac_gate_0' or t == 'dac_gate') else t

_qickip.QickMetadata.mod2type = _gate_mod2type
print("dac_gate patch applied: QickMetadata.mod2type treats dac_gate_0 as a register slice")
# --- end dac_gate tracer patch ---
# --- boxcar_tap tracer patch: QICK sees the stock readout topology.
_BOX_OUT = {('boxcar_tap_0', 'm_box_axis'): ['axis_dyn_readout_v1_1', 'm1_axis'],
            ('boxcar_tap_0', 'm_mr_axis'): ['axis_dyn_readout_v1_0', 'm0_axis']}
_BOX_IN = {('axis_dyn_readout_v1_1', 'm1_axis'): [['axis_broadcaster_1', 'S_AXIS']],
           ('axis_dyn_readout_v1_0', 'm0_axis'): [['axis_switch_mr', 'S00_AXIS']]}
_orig_trace_bus = _qickip.QickMetadata.trace_bus

def _box_trace_bus(self, blockname, portname):
    if 'boxcar_tap_0' not in self.modinfo:
        return _orig_trace_bus(self, blockname, portname)
    if (blockname, portname) in _BOX_IN:
        return [list(x) for x in _BOX_IN[(blockname, portname)]]
    return [list(_BOX_OUT.get(tuple(x), x)) if len(x) == 2 else x
            for x in _orig_trace_bus(self, blockname, portname)]

_qickip.QickMetadata.trace_bus = _box_trace_bus
print("boxcar_tap patch applied: QickMetadata.trace_bus aliases boxcar_tap_0 to the stock readout ports")
# --- end boxcar_tap tracer patch ---
# --- rf_tap / trig_sync tracer patch: rf_tap_0 and trig_sync_0 are invisible to QICK.
_L9_BUS_OUT = {('rf_tap_0', 'm_axis'): ['usp_rf_data_converter_0', 'm00_axis']}
_L9_BUS_FWD = {('rf_tap_0', 's_axis'): ['axis_register_slice_2', 'S_AXIS']}
_l9_prev_trace_bus = _qickip.QickMetadata.trace_bus
_l9_prev_trace_sig = _qickip.QickMetadata.trace_sig

def _l9_trace_bus(self, blockname, portname):
    r = _l9_prev_trace_bus(self, blockname, portname)
    if 'rf_tap_0' not in self.modinfo:
        return r
    out = []
    for x in r:
        if len(x) == 2 and tuple(x) in _L9_BUS_OUT and blockname != 'usp_rf_data_converter_0':
            out.append(list(_L9_BUS_OUT[tuple(x)]))
        elif len(x) == 2 and tuple(x) in _L9_BUS_FWD and blockname == 'usp_rf_data_converter_0':
            out.append(list(_L9_BUS_FWD[tuple(x)]))
        else:
            out.append(x)
    return out

def _l9_trace_sig(self, blockname, portname):
    r = _l9_prev_trace_sig(self, blockname, portname)
    if 'trig_sync_0' not in self.modinfo or blockname == 'trig_sync_0':
        return r
    out = []
    for x in r:
        if len(x) == 2 and x[0] == 'trig_sync_0' and x[1] == 'trig_out':
            src = [y for y in _l9_prev_trace_sig(self, 'trig_sync_0', 'trig_in')
                   if len(y) == 2 and y[0].startswith('qick_processor')]
            out.extend(src)
        else:
            out.append(x)
    return out

_qickip.QickMetadata.trace_bus = _l9_trace_bus
_qickip.QickMetadata.trace_sig = _l9_trace_sig
print("v1.40L9 patch applied: rf_tap_0 and trig_sync_0 aliased for QickMetadata tracing")
# --- end rf_tap / trig_sync tracer patch ---

class NNQickSoc(QickSoc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bram_info = self.mem_dict['axi_bram_ctrl_0']
        self._nn_bram_mmio = MMIO(bram_info['phys_addr'], bram_info['addr_range'])
        # NN_live record BRAM (absent on builds without NN_live)
        live = self.mem_dict.get('axi_bram_ctrl_live')
        self._live_mmio = MMIO(live['phys_addr'], live['addr_range']) if live else None
        # seg_stamp timestamp BRAM (absent on builds without seg_stamp)
        seg = self.mem_dict.get('axi_bram_ctrl_seg')
        self._seg_mmio = MMIO(seg['phys_addr'], seg['addr_range']) if seg else None
        # dac_gate config/statistics BRAM (absent on builds without dac_gate)
        gate = self.mem_dict.get('axi_bram_ctrl_gate')
        self._gate_mmio = MMIO(gate['phys_addr'], gate['addr_range']) if gate else None
        # classify trigger: trig_11 drives PMOD0_7 and live_gate_0, which qick's pin discovery skips; list it again
        # so NN_TRIGGER_PIN stays trig_11.
        self._fix_classify_pins()

    def _fix_classify_pins(self):
        tp = self['tprocs'][0]
        pins = tp['output_pins']
        have = {(p[0], p[1]) for p in pins}
        added = []
        for n in range(tp.get('out_trig_qty', 0)):
            if ('trig', n) in have:
                continue
            try:
                ports = self.metadata.trace_sig(tp['fullpath'], 'trig_%d_o' % n)
            except KeyError:
                continue
            fpga = [p for p in ports if len(p) == 1]
            other = [p for p in ports if len(p) != 1]
            if len(fpga) == 1 and other and all(p[0] in ('live_gate_0', 'dac_gate_0', 'trig_sync_0') for p in other):
                pins.append(('trig', n, 0, fpga[0][0]))
                added.append(n)
        if added:
            pins.sort(key=lambda p: (p[0] != 'trig', p[1], p[2]))
            print("classify trigger: re-listed tProc trigger output(s) %s as output pins" % added)

    def gate_present(self):
        return self._gate_mmio is not None

    def gate_read_block(self, start_word, n_words):
        m = self._gate_mmio
        return [m.read(4 * (start_word + k)) for k in range(n_words)]

    def gate_write(self, word, value):
        self._gate_mmio.write(4 * word, int(value) & 0xFFFFFFFF)

    def gate_stats(self):
        """Words 0..14: config, windows, passed, blanked, stale, decisions, margin,
        marker->pulse, decision lo, decision hi, flags, magic, late decisions,
        last late (cycles after window start), classify triggers (dac_gate.v)."""
        return self.gate_read_block(0, 21)   # GAT4: words 0..20 (19 binding rejects, 20 expected shot field)

    def box_present(self):
        return 'boxcar_tap_0' in self.metadata.modinfo

    def box_get(self):
        w = int(self.seg_read_block(3, 1)[0])
        return dict(sel=w & 1, shift=(w >> 4) & 0xF, m=((w >> 8) & 0x1F) + 1, word=w)

    def box_config(self, sel=None, shift=None, m=None):
        """boxcar_tap_0 cfg (seg_stamp word 3): sel 0 = NN_live on the readout FIR,
        1 = on the boxcar path; shift 0..15; m = moving-sum length 1..16."""
        c = self.box_get()
        sel = c['sel'] if sel is None else int(sel)
        shift = c['shift'] if shift is None else int(shift)
        m = c['m'] if m is None else int(m)
        if not (0 <= shift <= 15 and 1 <= m <= 16 and sel in (0, 1)):
            raise ValueError("sel 0/1, shift 0..15, m 1..16")
        self.seg_write(3, (sel & 1) | (shift << 4) | ((m - 1) << 8))
        return self.box_get()

    def list_ips(self):
        return list(self.ip_dict.keys())

    def list_mems(self):
        return {name: {"phys_addr": info.get("phys_addr"), "addr_range": info.get("addr_range")}
                for name, info in self.mem_dict.items()}

    def nn_configure(self, window_size, window_offset, scaling_factor):
        self.NN_axi_0.register_map.window_size = window_size
        self.NN_axi_0.register_map.window_offset = window_offset
        self.NN_axi_0.register_map.scaling_factor = scaling_factor

    def nn_reset(self):
        self.NN_axi_0.register_map.out_reset = 255 #1
        self.NN_axi_0.register_map.out_reset = 0

    def nn_status(self):
        reg = self.NN_axi_0.register_map
        return {"out_offset": int(reg.out_offset), "out_offset_ctrl": int(reg.out_offset_ctrl), "trig_count": int(reg.trig_count)}

    def nn_status_raw(self):
        # raw AXI-lite MMIO read at the register byte offsets, bypassing PYNQ's register_map
        m = self.NN_axi_0.mmio
        return {"out_offset_raw": m.read(0x30), "out_offset_ctrl_raw": m.read(0x34),
                "trig_count_raw": m.read(0x40), "trig_count_ctrl_raw": m.read(0x44)}

    def nn_read_bram(self, offset):
        return self._nn_bram_mmio.read(offset)

    def seg_present(self):
        return self._seg_mmio is not None

    def seg_read_block(self, start_word, n_words):
        m = self._seg_mmio
        return [m.read(4 * (start_word + k)) for k in range(n_words)]

    def seg_write(self, word, value):
        self._seg_mmio.write(4 * word, int(value) & 0xFFFFFFFF)

    def seg_latest(self):
        """[records, thr, magic] + the newest record (seg_stamp.v): 16 words (SEG2/SEG3: +8 edge0, +9 edge3),
        8 on SEG1."""
        m = self._seg_mmio
        hdr = [m.read(0), m.read(4), m.read(8)]
        n = 16 if hdr[2] in (0x53454732, 0x53454733) else 8
        if hdr[0] == 0:
            return hdr + [0] * n
        base = 16 + ((hdr[0] - 1) % 64) * n
        return hdr + [m.read(4 * (base + k)) for k in range(n)]

    def live_present(self):
        return self._live_mmio is not None

    def live_read_block(self, start_word, n_words):
        m = self._live_mmio
        return [m.read(4 * (start_word + k)) for k in range(n_words)]

    def live_latest(self):
        """[shots, frame_err, magic] + the newest 8-word record (NN_live.h)."""
        m = self._live_mmio
        hdr = [m.read(0), m.read(4), m.read(8)]
        if hdr[0] == 0:
            return hdr + [0] * 8
        base = 1024 + ((hdr[0] - 1) % 512) * 8
        return hdr + [m.read(4 * (base + k)) for k in range(8)]

    def nn_bram_word_count(self):
        return self._nn_bram_mmio.length // 4
    
    def nn_reset_debug(self):
        self.NN_axi_0.register_map.out_reset = 255
        readback_255 = int(self.NN_axi_0.register_map.out_reset)
        self.NN_axi_0.register_map.out_reset = 0
        readback_0 = int(self.NN_axi_0.register_map.out_reset)
        return {"wrote_255_read_back_as": readback_255, "wrote_0_read_back_as": readback_0}

    def nn_read_out_reset_raw(self):
        # raw MMIO read of out_reset, bypassing register_map
        return self.NN_axi_0.mmio.read(0x28)

    def nn_write_out_reset_raw(self, value):
        self.NN_axi_0.mmio.write(0x28, value)
        return self.NN_axi_0.mmio.read(0x28)

    def nn_set_onset_threshold(self, value):
        # Raw MMIO, bypassing register_map -- the same TypeError
        # ("__int__ returned non-int (type str)") seen on out_reset in
        # v1.8.0 also hits onset_threshold's register_map getter, so use
        # the already-proven-reliable raw-MMIO pattern instead.
        self.NN_axi_0.mmio.write(0x50, value)
        return self.NN_axi_0.mmio.read(0x50)

    def nn_get_onset_threshold(self):
        return self.NN_axi_0.mmio.read(0x50)

    def nn_onset_diagnostics(self):
        return {"drain_count_last": self.NN_axi_0.mmio.read(0x58),
                "onset_miss_count": self.NN_axi_0.mmio.read(0x68)}


        
    
    
proxy_name = 'myqick'
ns_port = 8888
ns_host = __import__('os').environ.get('QICK_NS_HOST', '192.168.2.99')   # this board's address
bitfile = __import__('os').environ.get('QICK_BITFILE', '/home/xilinx/jupyter_notebooks/OG_build_v1.44.bit')   # copy OG_build_v1.44.bit and .hwh here

# start the nameserver as a background process (non-blocking)
ns_proc = subprocess.Popen(
    [f'PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 pyro4-ns -n {ns_host} -p {ns_port}'],
    shell=True
)

# wait for the nameserver to actually come up
time.sleep(5)

# start the qick proxy server (this blocks forever, same as before)
start_server(
    bitfile=bitfile,
    proxy_name=proxy_name,
    ns_host=ns_host,
    ns_port=ns_port,
    soc_class=NNQickSoc,
)
