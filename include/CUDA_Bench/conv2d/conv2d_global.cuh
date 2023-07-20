#pragma once
#include <CUDA_Bench/util/precision_select.cuh>

// This is global variables that need to be used for NVBench :(
extern long ginput_N;
extern long ginput_H;
extern long ginput_W;
extern long ginput_C;
extern long gfilter_K;
extern long gfilter_R;
extern long gfilter_S;
extern long gstride_H;
extern long gstride_V;
extern int gnum_iter;
extern Precision gmulprecision;
extern Precision gaccprecision;
extern bool gprint_result;
extern bool gtensor_cores;
extern bool gprofiling;
extern const int gargc_nvbench;
extern const char *gargv_nvbench[];