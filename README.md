<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">CUDA Tensor Cores Benchmark</h3>

  <p align="center">
    A collection of CUDA GPU Micro Benchmarks for research purposes. This benchmarks will make use of Tensor Cores available on NVIDIA Volta, NVIDIA Turing, and NVIDIA Ampere GPU architectures. Future release will include NVIDIA Ada Lovelace and NVIDIA Hopper (if I am able to get my hands on these new GPUs, hopefully)
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#about-nvidia-tensor-cores-gpu">About NVIDIA Tensor Cores GPU</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#gemm-general-matrix-matrix-multiplication">GEMM - General Matrix-Matrix Multiplication</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains a collection of microbenchmark written in C++ and CUDA for research purposes. They are designed to stress the Tensor Cores available on NVIDIA Volta, NVIDIA Turing, and NVIDIA Ampere GPU architectures. Support for newer NVIDIA Ada Lovelace and NVIDIA Hopper are planned (if I get the hardware to test for it).

The benchmarks are implemented with either cuBLAS or cutlass. The NVIDIA cuBLAS should be included with the NVIDIA CUDA Toolkit. I would recommend to use NVIDIA CUDA Toolkit with version higher than 11.4. **Avoid the use of NVIDIA CUDA Toolkit version 11.2** since they have bug with IMMA operations that use Tensor Cores for integer operation (int8/int4) on NVIDIA Turing and NVIDIA Ampere. The NVIDIA cutlass is included as submodule in this project. Currently, int4 IMMA operation is only supported on cutlass while the other HMMA (fp16) and IMMA (int8) are both supported by cuBLAS and cutlass. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### About NVIDIA Tensor Cores GPU
This benchmark is designed to stress the Tensor Cores unit on NVIDIA GPUs. The following list describes the NVIDIA GPU Architectures that have Tensor Cores and their respective supported precisions.
1. NVIDIA Volta (First generation of Tensor Cores)
   - SM70 Devices: Tesla V100, Titan V, and Quadro GV100
   - Precision supported with Tensor Cores: FP16
   - Precision supported with CUDA Cores: FP64, FP32, FP16, INT8

2. NVIDIA Turing (Second generation of Tensor Cores)
   - SM75 Devices: GTX 16xx, RTX 2xxx, Titan RTX, Quadro RTX xxxx, Quadro Txxxx, Tesla T4 
   - Precision supported with Tensor Cores: FP16, INT8, INT4, INT1
   - Precision supported with CUDA Cores: FP64, FP32, FP16, INT8

3. NVIDIA Ampere (Third generation of Tensor Cores)
   - SM80 Devices: A100
   - SM86 Devices: RTX 3xxx, RTX Axxxx, Axx, Ax, ...
   - Precision supported with Tensor Cores: FP64, TF32, BF16, FP16, INT8, INT4, INT1
   - Precision supported with CUDA Cores: FP64, FP32, BF16, FP16, INT8

4. NVIDIA Hopper (Fourth generation of Tensor Cores)
   - SM90 Devices: H100
   - Precision supported with Tensor Cores: FP64, TF32, BF16, FP16, FP8, INT8
   - Precision supported with CUDA Cores: FP64, FP32, BF16, FP16, INT8

5. NVIDIA Ada Lovelace (Fourth generation of Tensor Cores)
   - SM89 Devices: RTX 4xxx, RTX 6000 (Ada), L40
   - Precision supported with Tensor Cores: FP64, TF32, BF16, FP16, FP8, INT8
   - Precision supported with CUDA Cores: FP64, FP32, BF16, FP16, INT8

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

The following libraries/frameworks are used in this repository.

* [cuBLAS] [https://developer.nvidia.com/cublas]
* [cutlass] [https://github.com/NVIDIA/cutlass]
* [argparse] [https://github.com/p-ranav/argparse]
* [nvbench] [https://github.com/NVIDIA/nvbench]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

_TODO_

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* TODO
  ```sh
  TODO
  ```

### Installation

Installation can be done easily through the following steps. Make sure that you have all dependencies configured correctly on your system. 

1. Clone the CUDA_Bench Github repository
   ```sh
   git clone https://github.com/hibagus/CUDA_Bench.git
   ```
2. Change directory to CUDA_Bench
   ```sh
   cd CUDA_Bench
   ```
3. Clone the submodule
   ```sh
   git submodule update --init --recursive
   ```
4. Change the target GPU Architecture by setting `set(GPU_ARCHITECTURE_SUPPORT "XX")` using the following command, where `XX` is the CUDA Compute Capability (SM). This setting will be automated in future release.
   ```sh
   vi cmake/CUDASettings.cmake
   ```
5. Make build directory and go inside it
   ```sh
   mkdir build && cd build
   ```
6. Run cmake
   ```sh
   # Recommended Build
   cmake -DBUILD_MODE=Release ..

   # Build for Debugging
   cmake -DBUILD_MODE=Debug ..

   # Build for Profiling with Code Analysis
   cmake -DBUILD_MODE=Profile ..
   ```
7. Run make
   ```sh
   make
   ```
8. Binaries are available in bin directory
   ```sh
   cd ../bin
   ```
9. Run appropriate binary by following the instructions of each binary.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Microbenchmark Details -->
## FIR - Finite Impulse Response Filter Computation

<!-- Microbenchmark Details -->
## GEMM - General Matrix-Matrix Multiplication
This is general matrix-matrix multiplication on GPUs. It performs multiplication in the form of C = (alpha)x(AxB) + (beta)xC where A, B, and C are matrices with dimension MxK, KxN, and MxN, respectively. The scaling factor alpha and beta are set fixed to 1 and 0, respectively. By default, it uses cutlass as its library, but user can choose to use cuBLAS as well. 

### Supported precision
This benchmark is targetted to stress test the Tensor Cores, but it has the ability to use CUDA Cores as well. Supported operations are dependent on GPU hardware architectures. I would be more than happy to implement other precision as long as it is supported by the hardware and library.

#### CUDA Cores
It supports FP64, FP32, FP16, and INT8 precision on CUDA Cores. Both cuBLAS and cutlass implementation are available.

#### Tensor Cores
It supports FP32 with fast-lossy precision, FP16, and INT8 using cuBLAS. On cutlass, it supports FP16, INT8, and INT4 precisions. Cutlass implementation needs minimum matrices size to be completed successfully. If you encounter error like misaligned address, please try to use larger matrices size.

### Usage
The user guide can be obtained in help menu of the program.
```sh
./gemm_cuda_bench --help

Usage: ./gemm_cuda_bench [-h] [--result] [--cudacoresonly] [--usecublas] [--profile] [--mulprecision MULPREC] [--accprecision ACCPREC] [--iterations ITER] dim_M dim_N dim_K

Positional arguments:
  dim_M                         Positive integer that describes M dimension of the matrices A(MxK) and C(MxN) 
  dim_N                         Positive integer that describes N dimension of the matrices B(KxN) and C(MxN) 
  dim_K                         Positive integer that describes K dimension of the matrices A(MxK) and B(KxN) 

Optional arguments:
  -h, --help                    shows help message and exits 
  -R, --result                  Show result at the end of program 
  -C, --cudacoresonly           Use CUDA Cores only and do not use Tensor Cores 
  -B, --usecublas               Use NVIDIA CUBLAS library instead of NVIDIA CUTLASS for GEMM 
  -P, --profile                 Enable built-in kernel profiling with NVBench 
  -M, --mulprecision MULPREC    Select matrix multiplication precision: fp64, fp32, fp16, int8, or int4 [default: "fp16"]
  -A, --accprecision ACCPREC    Select matrix accumulation precision: fp64, fp32, fp16, int8, or int4 [default: "fp16"]
  -I, --iterations ITER         Number of iterations, useful for performance profiling [default: 1]
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

TODO!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Bagus Hanindhito - hanindhito [at] bagus [dot] my [dot] id

Project Link: [https://github.com/hibagus/CUDA_Bench](https://github.com/hibagus/CUDA_Bench)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

TODO!
