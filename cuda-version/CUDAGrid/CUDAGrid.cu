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

__global__ void cuda_inflow(float* vel, float* f_in, float* density, float* weights, float* e){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= gpu_height)return;
    int vel_cord = id*gpu_width*2 + 0;
    int f_in_cord = id*gpu_width*9 + 0;
    int density_cord = id*gpu_width + 0;

    vel[vel_cord + 0] = -(1-2) * gpu_uLB * (1 + 1e-4*sin(id/(gpu_height-1)*2*3.14));
    vel[vel_cord + 1] = -(1-2) * gpu_uLB * (1 + 1e-4*sin(id/(gpu_height-1)*2*3.14));
    float sum1 = f_in[f_in_cord + 3] + f_in[f_in_cord + 4] + f_in[f_in_cord + 5];
    float sum2 = f_in[f_in_cord + 6] + f_in[f_in_cord + 7] + f_in[f_in_cord + 8];
    density[density_cord] = 1/(1-vel[vel_cord + 0])*(sum1 + 2 * sum2);

    for(int i = 0; i < 3; i++)
        f_in[f_in_cord + i] = equilibrium(density[density_cord], weights[i], e[i*2+0],e[i*2+1], vel[vel_cord + 0],vel[vel_cord + 1]) +
                                          f_in[f_in_cord+8-i] - equilibrium(density[density_cord], weights[8-i], e[(8-i)*2+0],e[(8-i)*2+1], vel[vel_cord+0],vel[vel_cord+1]); 
}

__global__ void cuda_outflow(float* f_in){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= gpu_height)return;
    int f_in_cord1 = id*gpu_width*9 + (gpu_width-1)*9;
    int f_in_cord2 = id*gpu_width*9 + (gpu_width-2)*9;

    f_in[f_in_cord1 + 6] = f_in[f_in_cord2 + 6];
    f_in[f_in_cord1 + 7] = f_in[f_in_cord2 + 7];
    f_in[f_in_cord1 + 8] = f_in[f_in_cord2 + 8];
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

__global__ void cuda_streaming(float* f_in, float* f_out, float* e){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gpu_width || y >= gpu_height) return;
    int f_out_cord = y*gpu_width*9 + x*9;

    for(int i = 0; i < 9; i++){
        int id1 = x + e[i*2 + 0];
        int id2 = y + e[i*2 + 1];
        if(id1 == gpu_width)id1 = 0;
        if(id1 == -1)id1 = gpu_width - 1;
        if(id2 == gpu_height)id2 = 0;
        if(id2 == -1)id2 = gpu_height - 1;
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


__global__ void bounce_back(float* f_in, float* f_out, float *obstacle, int obstacle_size){
    int obstacle_cord = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    if(obstacle_cord >= obstacle_size*2)return;
    int f_cord_x = obstacle[obstacle_cord + 0];
    int f_cord_y = obstacle[obstacle_cord + 1];
    for(int i = 0; i < 9; i++)
        f_out[f_cord_y*gpu_width*9 + f_cord_x*9 + i] = f_in[f_cord_y*gpu_width*9 + f_cord_x*9 + 8-i];
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

Field::Field(size_t width, size_t height, int block_size){


    blockSize = dim3(block_size, block_size);
    gridSize = dim3((width+(block_size-1))/block_size, (height+(block_size-1))/block_size);

    float r = ((height-1)/9);
    float uLB = 0.04;
    float Re = 10;
    float nulb = (uLB*r/Re);
    float omega = (1/(3*nulb+0.5));

    set_device_constants<<<1, 1>>>(width, height, r, uLB, Re, nulb, omega);

    float *temp_f_in = new float[width*height*9];
    float *temp_f_out = new float[width*height*9];
    float *temp_vel = new float[width*height*2];
    float *temp_density = new float[width*height];

    float host_e[9 * 2] = { 1,1, 1,0, 1,-1, 0,1, 0,0, 0,-1, -1,1, -1,0, -1,-1 };
    float host_weights[9] = {1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36};
    cout << "velocity\n";
    //velocity
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) { 
            temp_vel[i*2 + j*width*2 + 0] = (1-2) * uLB * (1 + 1e-4*sin(i/(width-1)*0.5))/10; 
            temp_vel[i*2 + j*width*2 + 1] = (1-2) * uLB * (1 + 1e-4*sin(j/(height-1)*0.5))/10;
            if(j == width - 1 || j == 0){
                temp_vel[i*2 + j*width*2 + 0] = 0;
                temp_vel[i*2 + j*width*2 + 1] = 0;
            }
            }
    }

    //f_in
    cout << "f in\n";
    for (int i = 0; i < 9; i++) {
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++){
                double usqr = 3.0/2 * (temp_vel[w*2 + h*width*2 + 0]*temp_vel[w*2 + h*width*2 + 0] + temp_vel[w*2 + h*width*2 + 1]*temp_vel[w*2 + h*width*2 + 1]);
                double cu = 3 * (host_e[i*2 + 0]*temp_vel[w*2 + h*width*2 + 0] + host_e[i*2 + 1]*temp_vel[w*2 + h*width*2 + 1]);
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
    vector<vector<int>> vec_obstacle;
    for(int w = 0; w < width; w++){
        for(int h = 0; h < height; h++){
            if((w - width/3)*(w - width/3) + (h-height/2)*(h-height/2) < r*r){
                vec_obstacle.push_back({w, h});
            }
        }
    }
    float *arr_obstacle = new float[vec_obstacle.size() * 2]; 
    for(int i = 0; i < vec_obstacle.size(); i++){
        arr_obstacle[i*2 + 0] = vec_obstacle[i][0];
        arr_obstacle[i*2 + 1] = vec_obstacle[i][1];
    }
    obstacle_size = vec_obstacle.size();

    cudaMalloc(&f_in, width * height * 9 * sizeof(float));
    cudaMalloc(&f_out, width * height * 9 * sizeof(float));
    cudaMalloc(&vel, width * height * 2 * sizeof(float));
    cudaMalloc(&density, width * height * sizeof(float));
    cudaMalloc(&e, 9 * 2 * sizeof(float));
    cudaMalloc(&weights, 9*sizeof(float));
    cudaMalloc(&visual_buffer, width*height*sizeof(unsigned char));
    cudaMalloc(&obstacle, obstacle_size*2*sizeof(float));

    cudaMemcpy(f_in, temp_f_in, width * height * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(f_out, temp_f_out, width * height * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vel, temp_vel, width * height * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(density, temp_density, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(e, host_e, 9 * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights, host_weights, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(obstacle, arr_obstacle, obstacle_size*2*sizeof(float), cudaMemcpyHostToDevice);

    delete [] temp_f_in;
    delete [] temp_f_out;
    delete [] temp_vel;
    delete [] temp_density;
}

Field::~Field(){
    if (f_in) cudaFree(f_in);
    if (f_out) cudaFree(f_out);
    if (vel) cudaFree(vel);
    if (density) cudaFree(density);
    if (e) cudaFree(e);
    if (weights) cudaFree(weights);
    if (visual_buffer) cudaFree(visual_buffer);
    if (obstacle) cudaFree(obstacle);
}

void Field::step(){
    cuda_outflow<<<34,32>>>(f_in);
    cudaDeviceSynchronize();
    cuda_density_velocity<<<gridSize, blockSize>>>(f_in, vel, density, e);
    cudaDeviceSynchronize();
    cuda_inflow<<<34,32>>>(vel, f_in, density, weights, e);
    cudaDeviceSynchronize();
    cuda_collision<<<gridSize,blockSize>>>(f_in, f_out, density, vel, weights, e);
    cudaDeviceSynchronize();
    bounce_back<<<(obstacle_size+31)/32, 32>>>(f_in, f_out, obstacle, obstacle_size);
    cudaDeviceSynchronize();
    cuda_streaming<<<gridSize,blockSize>>>(f_in, f_out, e);
    cudaDeviceSynchronize();
}

unsigned char* Field::get_visual(){
    convert_visual<<<gridSize, blockSize>>>(vel, visual_buffer, 0.1);
    cudaDeviceSynchronize();
    return visual_buffer;
}