#ifndef CUDAGrid
#define CUDAGrid

#pragma once
#include <cuda_runtime.h>
#include <vector>

enum class FlowDirection { LEFT_TO_RIGHT, RIGHT_TO_LEFT, TOP_TO_DOWN, DOWN_TO_TOP };
// enum class BoundaryType { INFLOW, OUTFLOW, PERIODIC, WALL }; // maybe in perspective...

__device__ float equilibrium(float density, float weight, float e1, float e2, float vel1, float vel2);

__global__ void cuda_density_velocity(float* f_in, float* vel, float* density, float* e);

__global__ void cuda_inflow(float* vel, float* f_in, float* density, float* weights, float* e);

__global__ void cuda_outflow(float* f_in);

__global__ void cuda_collision(float* f_in, float* f_out, float* density, float* vel, float* weights, float* e);

__global__ void cuda_streaming(float* f_in, float* f_out, float* e);

__global__ void convert_visual(float *vel, unsigned char *visual_buffer, float max);

__global__ void bounce_back(float* f_in, float* f_out, float *obstacle, int obstacle_size);

class Field {
public:
    Field(size_t width, size_t height, int block_size, float uLB, float Re, 
          FlowDirection flow_dir, bool* obstacle_mask);
    ~Field();

    void step();
    unsigned char* get_visual(float max);
    void update_obstacle_mask(bool* obstacle_mask);
private:
    size_t width, height;
    dim3 blockSize;
    dim3 gridSize;
    float *f_in, *f_out, *vel, *density, *e, *weights;
    bool* obstacle_mask;
    unsigned char *visual_buffer;
    FlowDirection flow_dir;
};

#endif