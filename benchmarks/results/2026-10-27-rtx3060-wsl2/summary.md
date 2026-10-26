GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 572.60  
CPU: AMD Ryzen 7 7700 8-Core Processor  
OS: Ubuntu 24.04.2 LTS, kernel 6.18.54.1-microsoft-standard-WSL2; Build cuda_12.8.r12.8/compiler.35404655_0  
Commit 095ca1a, STEPS=2000 WARMUP=200 REPEATS=3 CPU_STEPS=30 SIZES='640x360 960x540 1280x720 1920x1080'

| Grid | Nodes | GPU ms/step | GPU MLUPS | CPU prototype ms/step | Speed-up vs CPU prototype |
|---|---:|---:|---:|---:|---:|
| 640×360 | 0.23 M | 0.68 (±4%) | 341 | 7.9 | 12× |
| 960×540 | 0.52 M | 1.17 (±1%) | 444 | 18.5 | 16× |
| 1280×720 | 0.92 M | 1.88 (±1%) | 489 | 33.2 | 18× |
| 1920×1080 | 2.07 M | 3.96 (±1%) | 523 | 75.6 | 19× |
