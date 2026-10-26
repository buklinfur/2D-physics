# CUDA modules

The simulation core, built as static libraries and linked into `executables` by `../CMakeLists.txt`.

| Library | Files | Contents |
|---|---|---|
| `CUDAGrid` | `CUDAGrid/` | `Field`: device buffers (`f_in`, `f_out`, velocity, density), D2Q9 constants, LBM kernels, `step()`, `get_visual()` |
| `CUDADraw` | `CUDADraw/` | `GLVisual`: GLFW window and GL texture registered with CUDA; `draw()` copies a device image into it |
| `Obstacles` | `Obstacles/` | `Obstacle` → `Circle`, `Rectangle`, `Custom` (list of solid cells); `ObstaclesFactory` keeps their union as a device mask |
| `CUDAFacade` | `CUDAFacade/` | `CUDAFacade`: owns the three above; `run()`, `benchmark()`, `dump_frames()`, `add_*()` |

## Minimal use

```cpp
#include "CUDAFacade/CUDAFacade.cuh"

CUDAFacade sim(/*width*/ 960, /*height*/ 540, /*block*/ 16, /*uLB*/ 0.04f, /*Re*/ 100.0f,
               FlowDirection::LEFT_TO_RIGHT, "LBMSim");
sim.add_circle(320.0f, 270.0f, 50.0f);
sim.add_rectangle(600.0f, 200.0f, 40.0f, 120.0f);
sim.run();  // until the window is closed
```

Pass `headless = true` (last constructor argument) to skip the window; then only
`benchmark(steps, warmup, /*render*/ false)` and `dump_frames()` are available.

## Dependencies

CUDA Toolkit 12.x, OpenGL, GLEW, GLFW 3.3+ — on Ubuntu:
`sudo apt install nvidia-cuda-toolkit libglew-dev libglfw3-dev`.
