#pragma once
#include <CUDA_Bench/util/precision_select.cuh>

// This is global variables that need to be used for NVBench :(
extern int gorigdim_M;
extern int gdim_M;
extern int gnum_iter;
extern Precision gmulprecision;
extern Precision gaccprecision;
extern bool gprint_result;
extern bool gtensor_cores;
extern bool guse_cublas;
extern bool gprofiling;
extern bool galternate;
extern const int gargc_nvbench;
extern const char *gargv_nvbench[];