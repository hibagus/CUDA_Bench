// Matrix-matrix Multiplication using cutlass
// (C) 2022 Bagus Hanindhito

#include <CUDA_Bench/gemm/gemm_cutlass.cuh>
#include <CUDA_Bench/gemm/gemm_cutlass_launch_int.cuh>
#include <CUDA_Bench/gemm/gemm_cutlass_launch_fp.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/util/gpuinfo.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/util/precision_select.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

int gemm_cutlass()
{
     // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    if(gdevice<nDevices)
    {
        std::cout << "[INFO] Using GPU index " << gdevice << " to run the benchmark." << std::endl;
        gpuErrchk(cudaSetDevice(gdevice));
    }
    else
    {
        std::cout << "[ERR!] Invalid GPU index " << gdevice << "!" << std::endl;
        exit(1);
    }


    // Detect Device Capability
    cudaDeviceProp props;
    gpuErrchk(cudaGetDeviceProperties(&props, 0));
    GPUARCH gpuarch;

    switch(props.major * 10 + props.minor)
    {
        case 70: {gpuarch=GPUARCH_VOLTA;  break;}
        case 75: {gpuarch=GPUARCH_TURING; break;}
        case 80: {gpuarch=GPUARCH_AMPERE; break;}
        case 86: {gpuarch=GPUARCH_AMPERE; break;}
        case 89: {gpuarch=GPUARCH_AMPERE; break;} // Temporary fix for Ada Lovelace
        case 100: {gpuarch=GPUARCH_AMPERE; break;} // Temporary fix for blackwell
        case 120: {gpuarch=GPUARCH_AMPERE; break;} // Temporary fix for blackwell
        default: {gpuarch=GPUARCH_OTHER; break;}
    }
    
    
    // Precision Compability Check
    if(gmulprecision==PRECISION_FP64 && gaccprecision==PRECISION_FP64)
    {
        if(gtensor_cores)
        {
            std::cout << "[WARN] Currently Tensor Cores are not supporting FP64 multiplication and accumulation\n";            
        }
        
        switch(gpuarch)
        {
            case GPUARCH_VOLTA : 
            {
                gemm_cutlass_launch_volta_fp64_fp64_fp64_ntc();
                break;
            }
            case GPUARCH_TURING: 
            {
                gemm_cutlass_launch_turing_fp64_fp64_fp64_ntc();
                break;
            }
            case GPUARCH_AMPERE: 
            {
                gemm_cutlass_launch_ampere_fp64_fp64_fp64_ntc();
                break;
            }
            default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
        }
    }
    else if(gmulprecision==PRECISION_FP32 && gaccprecision==PRECISION_FP32)
    {
        if(gtensor_cores)
        {
            std::cout << "[WARN] Currently Tensor Cores are not supporting FP32 multiplication and accumulation\n";
            std::cout << "[WARN] Use CUBLAS for FP32 multiplication and accumulation on Tensor Cores with lossy precision\n";
        }
        switch(gpuarch)
        {
            case GPUARCH_VOLTA : 
            {
                gemm_cutlass_launch_volta_fp32_fp32_fp32_ntc();
                break;
            }
            case GPUARCH_TURING: 
            {
                gemm_cutlass_launch_turing_fp32_fp32_fp32_ntc();
                break;
            }
            case GPUARCH_AMPERE: 
            {
                gemm_cutlass_launch_ampere_fp32_fp32_fp32_ntc();
                break;
            }
            default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
        }
    }
    else if(gmulprecision==PRECISION_FP16 && gaccprecision==PRECISION_FP32)
    {
        if(gtensor_cores)
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : 
                {
                    gemm_cutlass_launch_volta_fp32_fp16_fp32_tc();
                    break;
                }
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_fp32_fp16_fp32_tc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_fp32_fp16_fp32_tc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
            
        }
        else
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : 
                {
                    gemm_cutlass_launch_volta_fp32_fp16_fp32_ntc();
                    break;
                }
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_fp32_fp16_fp32_ntc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_fp32_fp16_fp32_ntc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
        }

    }
    else if(gmulprecision==PRECISION_FP16 && gaccprecision==PRECISION_FP16)
    {
        if(gtensor_cores)
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : 
                {
                    gemm_cutlass_launch_volta_fp16_fp16_fp16_tc();
                    break;
                }
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_fp16_fp16_fp16_tc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_fp16_fp16_fp16_tc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
            
        }
        else
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : 
                {
                    gemm_cutlass_launch_volta_fp16_fp16_fp16_ntc();
                    break;
                }
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_fp16_fp16_fp16_ntc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_fp16_fp16_fp16_ntc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
        }

    }
    else if(gmulprecision==PRECISION_INT8 && gaccprecision==PRECISION_INT8)
    {
        std::cout << "[WARN] Promoting accumulation precision to int32 to maintain compability\n";
        if(gtensor_cores)
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : {std::cout << "[ERR!] Volta Tensor Cores do not support int8. Please use CUDA Cores instead\n"; exit(1); break;}
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_int32_int8_int32_tc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_int32_int8_int32_tc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
            
        }
        else
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : 
                {
                    gemm_cutlass_launch_volta_int32_int8_int32_ntc();
                    break;
                }
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_int32_int8_int32_ntc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_int32_int8_int32_ntc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
        }
    }
    else if(gmulprecision==PRECISION_INT4 && (gaccprecision==PRECISION_INT8 || gaccprecision==PRECISION_INT4))
    {
        std::cout << "[WARN] Promoting accumulation precision to int32 to maintain compability\n";
        if(gtensor_cores)
        {
            switch(gpuarch)
            {
                case GPUARCH_VOLTA : {std::cout << "[ERR!] Volta Tensor Cores do not support int4\n"; exit(1); break;}
                case GPUARCH_TURING: 
                {
                    gemm_cutlass_launch_turing_int32_int4_int32_tc();
                    break;
                }
                case GPUARCH_AMPERE: 
                {
                    gemm_cutlass_launch_ampere_int32_int4_int32_tc();
                    break;
                }
                default: {std::cout << "[ERR!] GPU Compute Capability is lower than it is required\n"; exit(1); break;}
            }
            
        }
        else
        {
            std::cout << "[ERR!] Operations involving int4 requires the use of Tensor Cores\n"; 
            exit(1);
        }
    }
    return 0;
}