# Streaming neural-network qubit readout with feed-forward on QICK

Code, firmware sources and results behind the qick feedback classifier project. Everything here
belongs to the final build, **v1.44**, on the Xilinx RFSoC 4x2 (XCZU48DR) running QICK (tProc v2). Two classifiers are
built from the same recorded readout data [16]:

- **NN_live**, a streaming classifier on the readout's decimated stream. It decides 16.3 ns after the last sample, and its
  decision drives a pulse gate (`dac_gate`) on signal generator 0 and the tProc. Measured loop latency is ~176 ns.
- **NN_axi**, a buffered hls4ml classifier in the conventional QICK deployment, whose results the host reads afterwards.

Run every script from this folder's root; paths in the scripts are relative to it. Some folder names are historical
(`campaign_v140L1/`, `campaign_v143/`, `audit_fixes/`). The paths below say what each one holds.


## Layout

| Path | What it holds |
|---|---|
| `firmware_v1.44/` | The block design as built (`d_1.bd`: QICK's design with our delta), hardware handoff (`.hwh`) and the Vivado build log. The bitstream is a release asset (`DATA_MANIFEST.csv`, with sha256) |
| `neural_net4_live576/rtl/`, `neural_net4_live576/constraints/` | Our RTL: `boxcar_tap.v` (low-latency decimation), `dac_gate.v` (pulse gate), `live_res_cdc.v` (decision clock crossing), `trig_sync.v`, `seg_stamp.v` / `rf_tap.v` (timestamps and taps for measurement), `live_rec_wreg.v`; CDC constraints |
| `neural_net4_live/rtl/live_window_gate.v` | Window gate in front of NN_live |
| `neural_net4_live576box/` | Test-vector generator for NN_live on the boxcar path (`make_tb_data.py`) |
| `neural_net4_live576s/` | NN_live HLS source (`firmware/NN_live.cpp`), C testbench and vectors, Vitis HLS scripts, hls4ml config, ROM generator; packaged core `NN_live 1.5` (zip) with its C-sim and synthesis logs |
| `neural_net4_tlast/` | NN_axi HLS source (hls4ml network + wrapper), testbench, scripts, synthesis reports; packaged core `NN_axi 1.41` (zip) |
| `vivado_scripts/v1.43/from_qick/` | Building our design from QICK's: `nnqick_from_qick.tcl`, the block-design delta, and the implementation script. These made v1.44 |
| `vivado_scripts/v1.43/` | The same design built earlier in our own project (v1.43): block-design edit (stage 1) and implementation with the full report set (stage 2). Kept for reference |
| `audit_fixes/RB_prep/sim/`, `audit_fixes/RB_prep/hls/rtl_sim/` | RTL simulation: custom blocks against reference models (`final_regression.sh`), and window gate + NN_live kernel on a continuous stream (`run_tb.sh`), with logs |
| `scaling_study/` | Multi-classifier resource count (`resource_scaling.py`) and out-of-context timing runs (`run_ooc*.tcl`, `rtl/`); findings |
| `host/` | Board-side server: `nnqick_server.py` (same as `start_nameserver.ipynb`). This holds the QICK metadata patches for the added blocks, the `NNQickSoc` class and the Pyro server |
| `board_tools/` | Host tools: deploy and verify a bitstream, Jupyter access, server start, calibration storage |
| `notebooks/` | Notebook versions of the main workflows (see Notebooks below) |
| `tools/` | Top-level scripts: replay-channel calibration (`run_calibration.py`), replay (`run_loopback.py`), test subsets (`make_test_subsets.py`), the leakage audit (`verify_no_leakage.py`), the synthetic-data generator (`qubit_synth_lib.py`, `generate_*.py`), and the on-board scoring harness |
| `calibration/` | Replay-channel calibration (`k.npy`, `c.npy`: the active pair, written by `run_calibration.py`) and the two label-free domain maps (`k2_loopback.npy` / `c2_loopback.npy` on the FIR path, `box_domain_map.json` on the boxcar path) |
| `data/` | Labels and manifests of the test subsets and replays (`test20k_real_*`, `test100k_real_y.npy`, `loopback20k_real_*`, `domain_calib_y.npy`); the arrays themselves are regenerated or fetched into this folder (see Data) |
| `results/` | Small result files that the analyses read back (three-way comparison, calibration ablation, raw reference) |
| `analysis_scripts/diagnostics/` | Every board measurement: fidelity scoring, bit-exact checks, loop latency, tProc path, boxcar check, functional suite, safety tests, drift |
| `analysis_scripts/analysis/`, `analysis_scripts/characterization/` | Offline analyses: fixed-point emulator, linear baselines, calibration ablation, three-way comparison; readout-path characterization |
| `analysis_scripts/bd_delta/` | Generates our block-design delta from QICK's design and checks that QICK's design plus the delta matches ours |
| `campaign_v144/` | The v1.44 data campaign: both boards at once (`run_all.sh`, per board `run_6_3.sh`), second board session (`run_board2*.sh`), 3 h drift (`run_drift_full.sh`), reload and cold-start latency (`run_loads.sh`), and the offline analyses (`run_analyses.sh`) |
| `audit_fixes/R18/`, `audit_fixes/R9/`, `audit_fixes/R3_R2/`, `audit_fixes/R10/`, `campaign_v140L1/` | In-loop campaign, synthetic-data study, linear baselines, fixed-point scoring, architecture and window sweeps (scripts called by the campaign and the analysis) |
| `analysis_scripts/acquisition/` | Training-set construction (domain maps) and model training/export (`export_signed_domain_model.py`) |
| `models/` | The three deployed networks: NN_live on the boxcar path (`..._w576_box`), NN_live on the FIR path (`..._w576`), NN_axi (`model_v140_200k_...`) |
| `campaign_v144/final/analysis/` | Analysis of the v1.44 data, and the results ledger: `make_v144_ledger.py` writes `paper_notes/V144_RESULTS.md`, `check_paper_v144.py` checks a draft's numbers against it |
| `campaign_v143/` | Files of the v1.43 campaign that the v1.44 scripts still read, among them the results ledger's row template (`final/analysis/v143_results.json`) |
| `paper_figures/fig1..fig8/`, `paper_tables/` | One generator per figure and table, with its values file and outputs |

## Rebuilding the firmware

Tools: Vivado and Vitis HLS 2023.1; QICK firmware sources at commit `779dffa`
([openquantumhardware/qick](https://github.com/openquantumhardware/qick), firmware version 0.2.420, MIT licence). We
use QICK's IP unmodified, together with its `qick_tprocv2_4x2_standard` constraints (`ios.xdc`, `timing.xdc`).

Our design is QICK's standard tProc v2 design for the RFSoC 4x2 with our blocks added. You build QICK's design with
QICK's own scripts, and then one script of ours turns it into ours.

1. **QICK's project.** Check out QICK at commit `779dffa`, then run its project script:
   ```
   cd firmware/projects/qick_tprocv2_4x2_standard
   vivado -mode batch -source proj.tcl
   ```
   This makes `top/top.xpr` with block design `d_1`. You don't need to build QICK's bitstream. Use a full checkout:
   QICK's IP cores pull shared sources from `firmware/hdl/` and `firmware/common_ip_files/`, and IP generation fails
   if those folders are missing. On Windows, keep the
   checkout at a short path (for example `C:\qick`). Vivado fails on paths longer than 260 characters, and QICK's IP
   paths are deep.
2. **Our design on top.** From this repository's root:
   ```
   vivado -mode batch -source vivado_scripts/v1.43/from_qick/nnqick_from_qick.tcl -tclargs <QICK>/firmware/projects/qick_tprocv2_4x2_standard/top/top.xpr <new folder>
   ```
   This saves QICK's project as a new project in `<new folder>` and leaves QICK's own project unchanged. It then
   installs the two HLS cores from their zips, adds our RTL and the CDC constraint, and applies
   `nnqick_bd_delta.tcl` to the block design. Use forward slashes in both paths.
3. **Implementation.** From `<new folder>`, run
   `vivado -mode batch -source <repo>/vivado_scripts/v1.43/from_qick/build_nnqick.tcl`. It's the flow we ran
   (`vivado_scripts/v1.43/build_v1.43.tcl`), with the same checks, constraints, directives and reports, adapted to
   QICK's design name. It writes `OG_build_v1.43.bit` and `.hwh`.

These three steps made the paper's firmware, v1.44: the script's outputs, renamed `OG_build_v1.44`. The scripts sit in
`vivado_scripts/v1.43/` because the design is v1.43's; v1.44 is the build they produced from QICK's project.

**What the delta changes in QICK's design** (`vivado_scripts/v1.43/from_qick/nnqick_bd_delta.tcl`):
- Removes the time tagger and its DMA path.
- Changes six QICK settings:
  - The tProc core clock (`clk_core`) runs at 200 MHz instead of 245.76 MHz.
  - The tProc gets a fourth input port, which receives NN_live's result.
  - Readout 0's broadcaster gets a third output, which feeds the boxcar.
  - The DDR4 buffer is 512 bits wide.
  - The AXI interconnect has five more outputs.
  - The DMA SmartConnect has one fewer input.
- Adds our blocks and wiring, and sets the address map.

We checked the result in Vivado. QICK's design at `779dffa` plus the delta matches our v1.43 block design in every
cell, every setting, all connectivity and the whole address map. The only differences are the instance-name prefix
(`d_1_` instead of `OG_build_`), one unconnected net stub, and two internal address exclusions inside QICK's AXI
interconnect. None of them changes the hardware.

**The exact design.** `firmware_v1.44/d_1.bd` is the block design as built for v1.44. It doesn't depend on QICK's
project scripts, only on QICK's IP repository at `779dffa`, so it still works if a later QICK release reorganises its
standard design. To use it, create a project for `xczu48dr-ffvg1517-2-e` with board part
`realdigital.org:rfsoc4x2:part0:1.0`. Add QICK's `firmware/ip` and the two cores as IP repositories, add our RTL, add
the block design, then QICK's `ios.xdc` and `timing.xdc` and our `live_res_cdc.xdc`.

**HLS cores (optional).** The packaged cores are included. To rebuild them, run `vitis_hls -f csim_live.tcl`,
`csynth_live.tcl` and `export_live.tcl` in `neural_net4_live576s/`, and the `*_tlast.tcl` scripts in
`neural_net4_tlast/`.

`analysis_scripts/bd_delta/` regenerates the delta from the two block designs and repeats the check.

The v1.44 build meets all timing constraints at sign-off (WNS 0.000 ns, no failing endpoint; paper Table III). The
routed timing and utilization reports are in `paper_tables/table3/data/`.

## Running on a board

PYNQ image on the RFSoC 4x2, with QICK installed. Copy the bitstream and `.hwh` to the board and run
`host/nnqick_server.py` there, with `QICK_NS_HOST` set to the board's address. Host scripts reach the board through
Pyro4 (`QICK_NS_HOST`) and its Jupyter server (`PYNQ_JUPYTER_PASSWORD`, default `xilinx`). The replay channel is a coax
loopback from DAC tile to ADC tile on the same board (paper Sec. IV). Calibrate with `tools/run_calibration.py`, then follow
`campaign_v144/run_all.sh`.

We ran the scripts on Windows 11 with Git Bash. The campaign shell scripts change to `$REPO_ROOT` (default: the current
folder), so start them from this folder's root. Some scripts still have Windows paths from our machines hard-coded
(`C:/qick_projects/...`, where the Vivado projects lived); these mark where an input or output lives and need adjusting.

## Notebooks

`notebooks/` holds self-contained versions of the main workflows, written to be read, run and changed. Each one
carries its own code: nothing is imported from this repository, only numpy, scipy, matplotlib, qick and Pyro4, plus
TensorFlow, QKeras and hls4ml for the training ones. Run them from the `notebooks/` folder; they read data from the
repository root (`DATA_DIR = ".."`) and write to `notebooks/outputs/`.

| Notebook | Needs a board | What it does |
|---|---|---|
| `01_calibrate_replay_channel` | yes | Two-pass calibration of the DAC → coax → ADC loopback, (k, c) |
| `02_replay_recorded_traces` | yes | Shuffled, balanced test subset; predistort, replay, check the class structure survives |
| `03_train_and_export_classifier` | no | Domain map, ternary QKeras network with seed selection, hls4ml export with the firmware's precisions (reproduces the deployed models) |
| `04_bit_accurate_fixed_point` | no | Compile the hls4ml C++ model and compare it with floating point on replayed traces |
| `05_feed_forward_on_the_decision` | yes | The feed-forward loop: gate calibration, in-loop shots, pulse/gate/decision agreement, unsafe count, loop latency |
| `06_synthetic_readout_data` | no | The synthetic-data generator (reproduces the paper's set bit for bit) and what it misses |
| `07_readout_noise` | no | Noise statistics of the recorded data, matched-filter ceiling, decay test |
| `08_nn_live_and_nn_axi_results` | optional | NN_live and NN_axi fidelities from the campaign data, board and classifier comparisons, NN_live bit-exact check; optional live capture |


## Third-party components

QICK (MIT licence) is referenced, not copied. The NN_axi and NN_live sources include code generated by hls4ml
(Apache-2.0) and its `nnet_utils`, and the Xilinx `ap_types` headers (Apache-2.0). Their licences apply to those files; the MIT licence in `LICENSE` covers the rest.
