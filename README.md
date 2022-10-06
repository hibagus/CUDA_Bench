<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">CUDA Benchmark</h3>

  <p align="center">
    A collection of CUDA GPU Micro Benchmarks for research purposes.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains a collection of microbenchmark written in C++ and CUDA for research purposes.
They mostly use cuBLAS as linear algebra library optimized for NVIDIA GPUs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

The following libraries/frameworks are used in this repository.

* [cuBLAS][https://developer.nvidia.com/cublas]
* [nvbench][https://github.com/NVIDIA/nvbench]

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
