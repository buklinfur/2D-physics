# LBMSim: 2D Lattice‐Boltzmann CUDA Simulation

Current state: technical preview

A lightweight 2D Lattice‐Boltzmann Method (LBM) fluid simulator, accelerated on NVIDIA GPUs via CUDA, with:

- **App Facade**: thin controller (`main.cu`) that wires together config parsing, update checking, and CUDA engine  
- **Config Parser**: JSON-based simulation parameters (grid size, flow, display, obstacles) via nlohmann/json  
- **CUDA Facade**: handles obstacle setup, kernel launches, and OpenGL visualization  
- **Self-Update Loader**: checks GitHub Releases, downloads & verifies SHA256 checksums, and self-replaces the binary  

---

## Features

- **2D LBM core** with configurable Reynolds number, lattice velocity, and flow direction  
- **Obstacle support**: circles, rectangles, or custom polygons & more in perspective
- **Real-time OpenGL display** with vsync and color‐scaling limits  

To be added: custom configs support, visual interface & more.

---

## Prerequisites

- **Linux x86_64**  
- **CUDA Toolkit** (v12.0 recommended)  
- **CMake ≥ 3.18**, GCC / Clang  
- System packages:  
  ```bash
  sudo apt update
  sudo apt install -y \
    nvidia-cuda-toolkit \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev
    ```


## Build

- **Download**
* ```git clone https://github.com/your-org/LBMSim.git```
* ```cd LBMSim/ProjectSource```

- **Install**
* ```mkdir build && cd build```
* ```cmake -DCMAKE_BUILD_TYPE=Release ..```
* ```cmake --build . -- -j```