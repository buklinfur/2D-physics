#ifndef CUDAGrid
#define CUDAGrid


__device__ float equilibrium(float density, float weight, float e1, float e2, float vel1, float vel2);

__global__ void cuda_density_velocity(float* f_in, float* vel, float* density, float* e);

__global__ void cuda_inflow(float* vel, float* f_in, float* density, float* weights, float* e);

__global__ void cuda_outflow(float* f_in);

__global__ void cuda_collision(float* f_in, float* f_out, float* density, float* vel, float* weights, float* e);

__global__ void cuda_streaming(float* f_in, float* f_out, float* e);

__global__ void convert_visual(float *vel, unsigned char *visual_buffer, float max);

__global__ void bounce_back(float* f_in, float* f_out, float *obstacle, int obstacle_size);

class Field{
private:
    int obstacle_size;
    float *e, *weights, *obstacle;
    float *f_in, *f_out, *vel, *density;
    unsigned char *visual_buffer;
    dim3 blockSize, gridSize;
public:
    Field(size_t width, size_t height, int block_size=32);
    ~Field();
    
    void step();
    unsigned char* get_visual();
};


#endif