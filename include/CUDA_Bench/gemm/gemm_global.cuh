#pragma once
#include <CUDA_Bench/util/precision_select.cuh>

// This is global variables that need to be used for NVBench :(
extern int gdim_M;
extern int gdim_N;
extern int gdim_K;
extern int gnum_iter;
extern int gdevice;
extern double galpha;
extern double gbeta;
extern Precision gmulprecision;
extern Precision gaccprecision;
extern bool gprint_result;
extern bool gtensor_cores;
extern bool guse_cublas;