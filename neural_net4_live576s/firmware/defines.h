#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<32,20> input_t;
typedef ap_fixed<48,32> q_dense_1_accum_t;
typedef ap_fixed<48,32> layer2_t;
typedef ap_int<2> weight2_t;
typedef ap_int<2> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<32,16> layer4_t;
typedef ap_fixed<40,8> batchnorm1_scale_t;
typedef ap_fixed<40,8> batchnorm1_bias_t;
typedef ap_fixed<32,16> layer5_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<32,16> q_output_accum_t;
typedef ap_fixed<32,16> layer6_t;
typedef ap_int<2> weight6_t;
typedef ap_int<2> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<16,6,AP_TRN,AP_SAT> q_output_alpha_result_t;   // v1.5 (R.17): saturate, was ap_fixed<16,6> (wrap)
typedef struct exponent_scale10_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale10_t;
typedef ap_int<2> bias10_t;
typedef ap_fixed<18,8> result_t;
typedef ap_fixed<18,8> sigmoid_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
