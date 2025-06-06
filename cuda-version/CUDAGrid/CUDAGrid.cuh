#ifndef CUDAGrid
#define CUDAGrid

#define FIELD_WIDTH 1920
#define FIELD_HEIGHT 1080
#define BLOCK_X 32
#define BLOCK_Y 32
#define BLOCKS_WIDTH (FIELD_WIDTH / BLOCK_X)
#define BLOCKS_HEIGHT (FIELD_HEIGHT / BLOCK_Y)

__device__ float equilibrium(float density, float weight, float e1, float e2, float vel1, float vel2);

__global__ void cuda_density_velocity(float* f_in, float* vel, float* density, float* e);

__global__ void cuda_inflow(float* vel, float* f_in, float* density, float* weights, float* e);

__global__ void cuda_outflow(float* f_in);

__global__ void cuda_collision(float* f_in, float* f_out, float* density, float* vel, float* weights, float* e);

__global__ void cuda_streaming(float* f_in, float* f_out, float* e);

class field{
private:
    
public:

};


#endif