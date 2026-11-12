#ifndef NN_H_
#define NN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void NN(
    input_t input_1[1152],
    result_t layer8_out[1]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
