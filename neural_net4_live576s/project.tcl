# NN_live 1.5: 1.4 with the logit saturating (ap_fixed<16,6,AP_TRN,AP_SAT>, audit R.17 / C1; v1.41).
# NN_live 1.4: 1.3 with ACC_L as a free-running pipeline (style=frp: no global clock enable; v1.40L10 timing).
# NN_live 1.3: 1.2 (boxcar-path classifier, same weights) with the tail retimed: batchnorm1 scale applied per sample, 1-cycle weight ROM (task 4.1h, v1.40L9).
# NN_live 1.2: the 576-sample-window classifier retrained on the boxcar readout path (task 4.1h, model_signed_domain_seed20260901_w576_box.h5, seed 20260904). 1.1 was the 576-sample FIR-path classifier (task 4.1g). 1.0 was the v1.40 classifier on the live readout stream (task 4.1, architecture B).
variable project_name
set project_name "NN_live"
variable part
set part "xczu48dr-ffvg1517-2-e"
# Runs on clk_adc0_x2 beside axis_dyn_readout_v1_0: 552.96 MHz.
variable clock_period
set clock_period 1.808
variable clock_uncertainty
set clock_uncertainty 12.5%
variable version
set version "1.5"
