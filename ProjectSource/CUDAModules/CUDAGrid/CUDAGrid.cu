#include <chrono>
#include <iostream>
#include <random>
#include "CUDAGrid.cuh"

using namespace std;

__device__ size_t gpu_width;
__device__ size_t gpu_height;
__device__ float gpu_r;
__device__ float gpu_uLB;
__device__ float gpu_Re;
__device__ float gpu_nulb;
__device__ float gpu_omega;

#define CUDA_CHECK(err, msg) \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << msg << ": " << cudaGetErrorString(err) << endl; \
        exit(1); \
    }

__device__ float equilibrium(float density,float weight,float e1, float e2,float vel1, float vel2){
    float usqr = 3.0/2 * (vel1*vel1 + vel2*vel2);
    float cu = 3 * (e1*vel1 + e2*vel2);
    float f_eq = density * weight * (1 + cu + 0.5*cu*cu - usqr);
    return f_eq;
}

__global__ void cuda_density_velocity(float* f_in, float* vel, float* density, float* e) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= gpu_width || y >= gpu_height) return;

    int vel_cord = y*gpu_width*2 + x*2;
    int density_cord = y*gpu_width + x;
    int f_in_cord = y*gpu_width*9 + x*9;

    density[density_cord] = 0;
    vel[vel_cord + 0] = 0;
    vel[vel_cord + 1] = 0;

    for (int k = 0; k < 9; k++) {
        density[density_cord] += f_in[f_in_cord + k];
        vel[vel_cord + 0] += e[k*2 + 0] * f_in[f_in_cord + k];
        vel[vel_cord + 1] += e[k*2 + 1] * f_in[f_in_cord + k];
    }

    if (density[density_cord] != 0) {
        vel[vel_cord + 0] /= density[density_cord];
        vel[vel_cord + 1] /= density[density_cord];
    }
}

__global__ void cuda_inflow(float* vel, float* f_in, float* density, float* weights, float* e, int flow_dir){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (flow_dir < 2 ? gpu_height : gpu_width)) return;

    int vel_cord, f_in_cord, density_cord;
    float u, v;
    if (flow_dir == 0) {
        vel_cord = id * gpu_width * 2;
        f_in_cord = id * gpu_width * 9;
        density_cord = id * gpu_width;
        u = gpu_uLB * (1 + 1e-4f * sin(id * 2 * 3.14f / (gpu_height - 1)));
        v = 0;
    } else if (flow_dir == 1) {
        vel_cord = id * gpu_width * 2 + (gpu_width - 1) * 2;
        f_in_cord = id * gpu_width * 9 + (gpu_width - 1) * 9;
        density_cord = id * gpu_width + (gpu_width - 1);
        u = -gpu_uLB * (1 + 1e-4f * sin(id * 2 * 3.14f / (gpu_height - 1)));
        v = 0;
    } else if (flow_dir == 2) {
        vel_cord = id * 2;
        f_in_cord = id * 9;
        density_cord = id;
        u = 0;
        v = gpu_uLB * (1 + 1e-4f * sin(id * 2 * 3.14f / (gpu_width - 1)));
    } else {
        vel_cord = (gpu_height - 1) * gpu_width * 2 + id * 2;
        f_in_cord = (gpu_height - 1) * gpu_width * 9 + id * 9;
        density_cord = (gpu_height - 1) * gpu_width + id;
        u = 0;
        v = -gpu_uLB * (1 + 1e-4f * sin(id * 2 * 3.14f / (gpu_width - 1)));
    }

    vel[vel_cord + 0] = u;
    vel[vel_cord + 1] = v;
    float sum1 = f_in[f_in_cord + (flow_dir == 0 ? 3 : flow_dir == 1 ? 6 : flow_dir == 2 ? 1 : 7)] +
                 f_in[f_in_cord + (flow_dir == 0 ? 4 : flow_dir == 1 ? 7 : flow_dir == 2 ? 2 : 8)] +
                 f_in[f_in_cord + (flow_dir == 0 ? 5 : flow_dir == 1 ? 8 : flow_dir == 2 ? 3 : 6)];
    float sum2 = f_in[f_in_cord + (flow_dir == 0 ? 6 : flow_dir == 1 ? 3 : flow_dir == 2 ? 7 : 1)] +
                 f_in[f_in_cord + (flow_dir == 0 ? 7 : flow_dir == 1 ? 4 : flow_dir == 2 ? 8 : 2)] +
                 f_in[f_in_cord + (flow_dir == 0 ? 8 : flow_dir == 1 ? 5 : flow_dir == 2 ? 6 : 3)];
    density[density_cord] = 1.0f / (1.0f - (flow_dir < 2 ? u : v)) * (sum1 + 2 * sum2);

    int start_i = (flow_dir == 0 ? 0 : flow_dir == 1 ? 3 : flow_dir == 2 ? 2 : 0);
    int end_i = (flow_dir == 0 ? 3 : flow_dir == 1 ? 6 : flow_dir == 2 ? 5 : 2);
    for (int i = start_i; i < end_i; i++) {
        int opp_i = (flow_dir == 0 ? 8 - i : flow_dir == 1 ? 14 - i : flow_dir == 2 ? 10 - i : 8 - i);
        f_in[f_in_cord + i] = equilibrium(density[density_cord], weights[i], e[i*2+0], e[i*2+1], vel[vel_cord + 0], vel[vel_cord + 1]) +
                              f_in[f_in_cord + opp_i] -
                              equilibrium(density[density_cord], weights[opp_i], e[opp_i*2+0], e[opp_i*2+1], vel[vel_cord + 0], vel[vel_cord + 1]);
    }
}

__global__ void cuda_outflow(float* f_in, int flow_dir){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (flow_dir < 2 ? gpu_height : gpu_width)) return;

    int f_in_cord1, f_in_cord2;
    if (flow_dir == 0) { 
        f_in_cord1 = id * gpu_width * 9 + (gpu_width - 1) * 9;
        f_in_cord2 = id * gpu_width * 9 + (gpu_width - 2) * 9;
        f_in[f_in_cord1 + 6] = f_in[f_in_cord2 + 6];
        f_in[f_in_cord1 + 7] = f_in[f_in_cord2 + 7];
        f_in[f_in_cord1 + 8] = f_in[f_in_cord2 + 8];
    } else if (flow_dir == 1) {
        f_in_cord1 = id * gpu_width * 9;
        f_in_cord2 = id * gpu_width * 9 + 9;
        f_in[f_in_cord1 + 3] = f_in[f_in_cord2 + 3];
        f_in[f_in_cord1 + 4] = f_in[f_in_cord2 + 4];
        f_in[f_in_cord1 + 5] = f_in[f_in_cord2 + 5];
    } else if (flow_dir == 2) {
        f_in_cord1 = (gpu_height - 1) * gpu_width * 9 + id * 9;
        f_in_cord2 = (gpu_height - 2) * gpu_width * 9 + id * 9;
        f_in[f_in_cord1 + 6] = f_in[f_in_cord2 + 6];
        f_in[f_in_cord1 + 7] = f_in[f_in_cord2 + 7];
        f_in[f_in_cord1 + 8] = f_in[f_in_cord2 + 8];
    } else { 
        f_in_cord1 = id * 9;
        f_in_cord2 = gpu_width * 9 + id * 9;
        f_in[f_in_cord1 + 1] = f_in[f_in_cord2 + 1];
        f_in[f_in_cord1 + 2] = f_in[f_in_cord2 + 2];
        f_in[f_in_cord1 + 3] = f_in[f_in_cord2 + 3];
    }
}

__global__ void cuda_collision(float *f_in, float *f_out, float *density, float* vel, float *weights, float *e){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gpu_width || y >= gpu_height) return;
    int f_in_cord = y*gpu_width*9 + x*9;
    int vel_cord = y*gpu_width*2 + x*2;
    int density_cord = y*gpu_width + x;
    
    for(int i = 0; i < 9; i++){
        float usqr = 3.0/2 * (vel[vel_cord+0]*vel[vel_cord+0] + vel[vel_cord+1]*vel[vel_cord+1]);
        float cu = 3 * (e[i*2+0] * vel[vel_cord+0] + e[i*2+1] * vel[vel_cord+1]);
        float f_eq = density[density_cord] * weights[i] * (1 + cu + 0.5*cu*cu - usqr);
        f_out[f_in_cord + i] = f_in[f_in_cord + i] - gpu_omega * (f_in[f_in_cord + i] - f_eq);
    }
}

__global__ void cuda_streaming(float* f_in, float* f_out, float* e, int flow_dir){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gpu_width || y >= gpu_height) return;
    int f_out_cord = y*gpu_width*9 + x*9;

    for(int i = 0; i < 9; i++){
        int id1 = x + e[i*2 + 0];
        int id2 = y + e[i*2 + 1];

        if (flow_dir == 0) {
            if (id1 < 0 || id1 >= gpu_width || id2 == 0 || id2 == gpu_height - 1) continue;
        } else if (flow_dir == 1) {
            if (id1 < 0 || id1 >= gpu_width || id2 == 0 || id2 == gpu_height - 1) continue;
        } else if (flow_dir == 2) {
            if (id1 == 0 || id1 == gpu_width - 1 || id2 < 0 || id2 >= gpu_height) continue;
        } else {
            if (id1 == 0 || id1 == gpu_width - 1 || id2 < 0 || id2 >= gpu_height) continue;
        }

        f_in[id2*gpu_width*9 + id1*9 + i] = f_out[f_out_cord + i];
    }
}

__global__ void convert_visual(float *vel, unsigned char *visual_buffer, float max){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gpu_width || y >= gpu_height) return;
    int vel_cord = y*gpu_width*2 + x*2;
    int vis_cord = y*gpu_width + x;
    visual_buffer[vis_cord] = (sqrt(vel[vel_cord + 0]*vel[vel_cord + 0] + vel[vel_cord + 1]*vel[vel_cord + 1]) / max) * 255;
}

__global__ void bounce_back(float* f_in, float* f_out, bool* obstacle_mask){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gpu_width || y >= gpu_height) return;

    int idx = y * gpu_width + x;
    if (!obstacle_mask[idx]) return;

    int f_cord = y * gpu_width * 9 + x * 9;
    for (int i = 0; i < 9; i++)
        f_out[f_cord + i] = f_in[f_cord + 8 - i];
}

__global__ void set_device_constants(size_t width, size_t height, float r,
                                 float uLB, float Re, float nulb, float omega){
    gpu_width = width;
    gpu_height = height;
    gpu_r = r;
    gpu_uLB = uLB;
    gpu_Re = Re;
    gpu_nulb = nulb;
    gpu_omega = omega;
}

Field::Field(size_t width, size_t height, int block_size, float uLB, float Re, 
             FlowDirection flow_dir, bool* obstacle_mask) 
    : width(width), height(height), flow_dir(flow_dir), obstacle_mask(obstacle_mask) {

    blockSize = dim3(block_size, block_size);
    gridSize = dim3((width+(block_size-1))/block_size, (height+(block_size-1))/block_size);

    float r = ((height-1)/9);
    float nulb = (uLB*r/Re);
    float omega = (1/(3*nulb+0.5));

    cudaError_t err;
    err = cudaMalloc(&f_in, width * height * 9 * sizeof(float)); CUDA_CHECK(err, "cudaMalloc f_in");
    err = cudaMalloc(&f_out, width * height * 9 * sizeof(float)); CUDA_CHECK(err, "cudaMalloc f_out");
    err = cudaMalloc(&vel, width * height * 2 * sizeof(float)); CUDA_CHECK(err, "cudaMalloc vel");
    err = cudaMalloc(&density, width * height * sizeof(float)); CUDA_CHECK(err, "cudaMalloc density");
    err = cudaMalloc(&e, 9 * 2 * sizeof(float)); CUDA_CHECK(err, "cudaMalloc e");
    err = cudaMalloc(&weights, 9 * sizeof(float)); CUDA_CHECK(err, "cudaMalloc weights");
    err = cudaMalloc(&visual_buffer, width * height * sizeof(unsigned char)); CUDA_CHECK(err, "cudaMalloc visual_buffer");

    set_device_constants<<<1, 1>>>(width, height, r, uLB, Re, nulb, omega);
    err = cudaGetLastError(); CUDA_CHECK(err, "set_device_constants kernel");

    float *temp_f_in = new float[width * height * 9];
    float *temp_f_out = new float[width * height * 9];
    float *temp_vel = new float[width * height * 2];
    float *temp_density = new float[width * height];

    float host_e[9 * 2] = { 1,1, 1,0, 1,-1, 0,1, 0,0, 0,-1, -1,1, -1,0, -1,-1 };
    float host_weights[9] = {1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36};
    cout << "velocity\n";
    //velocity
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) { 
            temp_vel[i*2 + j*width*2 + 0] = (1-2) * uLB * (1 + 1e-4*sin(i/(width-1)*0.5))/10; 
            temp_vel[i*2 + j*width*2 + 1] = (1-2) * uLB * (1 + 1e-4*sin(j/(height-1)*0.5))/10;
            if (j == height - 1 || j == 0) {
                temp_vel[i*2 + j*width*2 + 0] = 0;
                temp_vel[i*2 + j*width*2 + 1] = 0;
            }
        }
    }

    //f_out
    for (size_t i = 0; i < width * height * 9; i++) {
        temp_f_out[i] = 0.0f;
    }

    //f_in
    cout << "f in\n";
    for (int i = 0; i < 9; i++) {
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {
                double usqr = 3.0/2 * (temp_vel[w*2 + h*width*2 + 0]*temp_vel[w*2 + h*width*2 + 0] + 
                                       temp_vel[w*2 + h*width*2 + 1]*temp_vel[w*2 + h*width*2 + 1]);
                double cu = 3 * (host_e[i*2 + 0]*temp_vel[w*2 + h*width*2 + 0] + 
                                 host_e[i*2 + 1]*temp_vel[w*2 + h*width*2 + 1]);
                temp_f_in[w*9 + h*width*9 + i] = 1 * host_weights[i] * (1 + cu + 0.5*cu*cu - usqr);
            }
        }
    }

    //density
    cout << "density\n";
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) temp_density[i + j*width] = 0.0001;
    }

    //obstacle
    cout << "obstacle\n";

    err = cudaMemcpy(f_in, temp_f_in, width * height * 9 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "cudaMemcpy f_in");
    err = cudaMemcpy(f_out, temp_f_out, width * height * 9 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "cudaMemcpy f_out");
    err = cudaMemcpy(vel, temp_vel, width * height * 2 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "cudaMemcpy vel");
    err = cudaMemcpy(density, temp_density, width * height * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "cudaMemcpy density");
    err = cudaMemcpy(e, host_e, 9 * 2 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "cudaMemcpy e");
    err = cudaMemcpy(weights, host_weights, 9 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "cudaMemcpy weights");

    delete [] temp_f_in;
    delete [] temp_f_out;
    delete [] temp_vel;
    delete [] temp_density;
}

Field::~Field() {
    if (f_in) cudaFree(f_in);
    if (f_out) cudaFree(f_out);
    if (vel) cudaFree(vel);
    if (density) cudaFree(density);
    if (e) cudaFree(e);
    if (weights) cudaFree(weights);
    if (visual_buffer) cudaFree(visual_buffer);
}

void Field::step() {
    cudaError_t err;
    cuda_outflow<<<34,32>>>(f_in, static_cast<int>(flow_dir));
    err = cudaGetLastError(); CUDA_CHECK(err, "cuda_outflow kernel");
    cudaDeviceSynchronize();
    cuda_density_velocity<<<gridSize, blockSize>>>(f_in, vel, density, e);
    err = cudaGetLastError(); CUDA_CHECK(err, "cuda_density_velocity");
    cudaDeviceSynchronize();
    cuda_inflow<<<34,32>>>(vel, f_in, density, weights, e, static_cast<int>(flow_dir));
    err = cudaGetLastError(); CUDA_CHECK(err, "cuda_inflow kernel");
    cudaDeviceSynchronize();
    cuda_collision<<<gridSize,blockSize>>>(f_in, f_out, density, vel, weights, e);
    err = cudaGetLastError(); CUDA_CHECK(err, "cuda_collision kernel");
    cudaDeviceSynchronize();
    bounce_back<<<gridSize, blockSize>>>(f_in, f_out, obstacle_mask);
    err = cudaGetLastError(); CUDA_CHECK(err, "bounce_back kernel");
    cudaDeviceSynchronize();
    cuda_streaming<<<gridSize,blockSize>>>(f_in, f_out, e, static_cast<int>(flow_dir));
    err = cudaGetLastError(); CUDA_CHECK(err, "cuda_streaming kernel");
    cudaDeviceSynchronize();
}

unsigned char* Field::get_visual(float max) {
    cudaError_t err;
    convert_visual<<<gridSize, blockSize>>>(vel, visual_buffer, max);
    err = cudaGetLastError(); CUDA_CHECK(err, "convert_visual kernel");
    cudaDeviceSynchronize();
    return visual_buffer;
}

void Field::update_obstacle_mask(bool* obstacle_mask) {
    this->obstacle_mask = obstacle_mask;
}