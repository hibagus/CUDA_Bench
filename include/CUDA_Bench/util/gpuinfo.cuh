#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <CUDA_Bench/util/gpucheck.cuh>

template<typename T> inline void printDeviceTable(T t, const int& width)
{
    std::stringstream ss;
    ss << t;
    std::cout  << std::left << std::setw(width) << std::setfill(' ') << ss.str().substr(0,width);
}

inline void print_cuda_device_info(int nDevices)
{
    std::cout << "[INFO] Detected " << nDevices << " CUDA-capable device(s)\n";
    std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n"; 
    printDeviceTable("[INFO] |", 8); 
    printDeviceTable("#", 3);                  printDeviceTable("|", 1);
    printDeviceTable("Device Name", 24);       printDeviceTable("|", 1);
    printDeviceTable("CC", 4);      printDeviceTable("|", 1);
    printDeviceTable("#SM", 4);       printDeviceTable("|", 1);
    printDeviceTable("Freq. (MHz)", 11);     printDeviceTable("|", 1);
    printDeviceTable("Mem. Size (MB)", 14);  printDeviceTable("|", 1);
    printDeviceTable("Mem. BW (GB/s)", 14);  printDeviceTable("|", 1);
    printDeviceTable("\n", 1);
    std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n";  
    
    for (int i = 0; i < nDevices; i++) 
    {
        cudaDeviceProp prop;
        gpuErrchk(cudaGetDeviceProperties(&prop, i));
        printDeviceTable("[INFO] |", 8); 
        printDeviceTable(i, 3);                  printDeviceTable("|", 1);
        printDeviceTable(prop.name, 24);       printDeviceTable("|", 1);
        printDeviceTable(std::to_string(prop.major)+"."+std::to_string(prop.minor), 4);      printDeviceTable("|", 1);
        printDeviceTable(prop.multiProcessorCount, 4);   printDeviceTable("|", 1);
        printDeviceTable(prop.clockRate/1000, 11); printDeviceTable("|", 1);
        printDeviceTable(prop.totalGlobalMem/1048576, 14);  printDeviceTable("|", 1);
        printDeviceTable(2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6, 14);  printDeviceTable("|", 1);
        printDeviceTable("\n", 1);
        std::cout << "[INFO] +---+------------------------+----+----+-----------+--------------+--------------+\n";  
    }
}

inline void print_no_cuda_devices()
{
    std::cerr << "---------------------------------------------------------------"
           "--------------------\n";
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    std::cerr << "[ERR!]: No CUDA-capable devices are detected. Program will now exit.\n";
    std::cerr << "       Please check whether your system has CUDA-capable device installed"
                 " and the CUDA driver is installed correctly.\n";       
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    std::cerr << "---------------------------------------------------------------"
                 "--------------------\n";
    exit(1);
}