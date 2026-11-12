#include <iostream>

#include "NN.h"
#include "parameters.h"


void NN(
    input_t input_1[1152],
    result_t layer8_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer8_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 4608>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<batchnorm1_scale_t, 4>(s4, "s4.txt");
        nnet::load_weights_from_txt<batchnorm1_bias_t, 4>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 4>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 1>(b6, "b6.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale10_t, 1>(s10, "s10.txt");
        nnet::load_weights_from_txt<bias10_t, 1>(b10, "b10.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer4_t layer4_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    q_output_alpha_result_t layer10_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    nnet::dense<input_t, layer2_t, config2>(input_1, layer2_out, w2, b2); // q_dense_1

    nnet::normalize<layer2_t, layer4_t, config4>(layer2_out, layer4_out, s4, b4); // batchnorm1

    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // relu1

    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // q_output

    nnet::normalize<layer6_t, q_output_alpha_result_t, config10>(layer6_out, layer10_out, s10, b10); // q_output_alpha

    nnet::sigmoid<q_output_alpha_result_t, result_t, sigmoid_config8>(layer10_out, layer8_out); // sigmoid

}

