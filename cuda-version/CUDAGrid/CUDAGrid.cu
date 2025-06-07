#include <chrono>
#include <iostream>
#include <random>
#include "CUDAGrid.cuh"

using namespace std;

#define uLB 0.04
#define Re 10
#define r ((FIELD_HEIGHT-1)/9)
#define nulb (uLB*r/Re)
#define omega (1/(3*nulb+0.5))

__device__ float equilibrium(float density,float weight,float e1, float e2,float vel1, float vel2){
    float usqr = 3.0/2 * (vel1*vel1 + vel2*vel2);
    float cu = 3 * (e1*vel1 + e2*vel2);
    float f_eq = density * weight * (1 + cu + 0.5*cu*cu - usqr);
    return f_eq;
}

__global__ void cuda_density_velocity(float* f_in, float* vel, float* density, float* e) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= FIELD_WIDTH || y >= FIELD_HEIGHT) return;

    int vel_cord = y*FIELD_WIDTH*2 + x*2;
    int density_cord = y*FIELD_WIDTH + x;
    int f_in_cord = y*FIELD_WIDTH*9 + x*9;

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
    if(id >= FIELD_HEIGHT)return;
    int vel_cord = id*FIELD_WIDTH*2 + 0;
    int f_in_cord = id*FIELD_WIDTH*9 + 0;
    int density_cord = id*FIELD_WIDTH + 0;

    vel[vel_cord + 0] = -(1-2) * uLB * (1 + 1e-4*sin(id/(FIELD_HEIGHT-1)*2*3.14));
    vel[vel_cord + 1] = -(1-2) * uLB * (1 + 1e-4*sin(id/(FIELD_HEIGHT-1)*2*3.14));
    float sum1 = f_in[f_in_cord + 3] + f_in[f_in_cord + 4] + f_in[f_in_cord + 5];
    float sum2 = f_in[f_in_cord + 6] + f_in[f_in_cord + 7] + f_in[f_in_cord + 8];
    density[density_cord] = 1/(1-vel[vel_cord + 0])*(sum1 + 2 * sum2);

    for(int i = 0; i < 3; i++)
        f_in[f_in_cord + i] = equilibrium(density[density_cord], weights[i], e[i*2+0],e[i*2+1], vel[vel_cord + 0],vel[vel_cord + 1]) +
                                          f_in[f_in_cord+8-i] - equilibrium(density[density_cord], weights[8-i], e[(8-i)*2+0],e[(8-i)*2+1], vel[vel_cord+0],vel[vel_cord+1]); 
}

__global__ void cuda_outflow(float* f_in){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= FIELD_HEIGHT)return;
    int f_in_cord1 = id*FIELD_WIDTH*9 + (FIELD_WIDTH-1)*9;
    int f_in_cord2 = id*FIELD_WIDTH*9 + (FIELD_WIDTH-2)*9;

    f_in[f_in_cord1 + 6] = f_in[f_in_cord2 + 6];
    f_in[f_in_cord1 + 7] = f_in[f_in_cord2 + 7];
    f_in[f_in_cord1 + 8] = f_in[f_in_cord2 + 8];
}

__global__ void cuda_collision(float *f_in, float *f_out, float *density, float* vel, float *weights, float *e){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f_in_cord = y*FIELD_WIDTH*9 + x*9;
    int vel_cord = y*FIELD_WIDTH*2 + x*2;
    int density_cord = y*FIELD_WIDTH + x;
    
    for(int i = 0; i < 9; i++){
        float usqr = 3.0/2 * (vel[vel_cord+0]*vel[vel_cord+0] + vel[vel_cord+1]*vel[vel_cord+1]);
        float cu = 3 * (e[i*2+0] * vel[vel_cord+0] + e[i*2+1] * vel[vel_cord+1]);
        float f_eq = density[density_cord] * weights[i] * (1 + cu + 0.5*cu*cu - usqr);
        f_out[f_in_cord + i] = f_in[f_in_cord + i] - omega * (f_in[f_in_cord + i] - f_eq);
    }
}

__global__ void cuda_streaming(float* f_in, float* f_out, float* e){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f_out_cord = y*FIELD_WIDTH*9 + x*9;

    for(int i = 0; i < 9; i++){
        int id1 = x + e[i*2 + 0];
        int id2 = y + e[i*2 + 1];
        if(id1 == FIELD_WIDTH)id1 = 0;
        if(id1 == -1)id1 = FIELD_WIDTH - 1;
        if(id2 == FIELD_HEIGHT)id2 = 0;
        if(id2 == -1)id2 = FIELD_HEIGHT - 1;
        f_in[id2*FIELD_WIDTH*9 + id1*9 + i] = f_out[f_out_cord + i];
    }
}

__global__ void convert_visual(float *vel, unsigned char *visual_buffer, float max){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int vel_cord = y*FIELD_WIDTH*2 + x*2;
    int vis_cord = y*FIELD_WIDTH + x;
    visual_buffer[vis_cord] = (sqrt(vel[vel_cord + 0]*vel[vel_cord + 0] + vel[vel_cord + 1]*vel[vel_cord + 1]) / max) * 255;
}


__global__ void bounce_back(float* f_in, float* f_out, float *obstacle, int obstacle_size){
    int obstacle_cord = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    if(obstacle_cord >= obstacle_size*2)return;
    int f_cord_x = obstacle[obstacle_cord + 0];
    int f_cord_y = obstacle[obstacle_cord + 1];
    for(int i = 0; i < 9; i++)
        f_out[f_cord_y*FIELD_WIDTH*9 + f_cord_x*9 + i] = f_in[f_cord_y*FIELD_WIDTH*9 + f_cord_x*9 + 8-i];
}

Field::Field(){
    float *temp_f_in = new float[FIELD_WIDTH*FIELD_HEIGHT*9];
    float *temp_f_out = new float[FIELD_WIDTH*FIELD_HEIGHT*9];
    float *temp_vel = new float[FIELD_WIDTH*FIELD_HEIGHT*2];
    float *temp_density = new float[FIELD_WIDTH*FIELD_HEIGHT];

    float host_e[9 * 2] = { 1,1, 1,0, 1,-1, 0,1, 0,0, 0,-1, -1,1, -1,0, -1,-1 };
    float host_weights[9] = {1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36};
    cout << "velocity\n";
    //velocity
    for (int i = 0; i < FIELD_WIDTH; i++) {
        for (int j = 0; j < FIELD_HEIGHT; j++) { 
            temp_vel[i*2 + j*FIELD_WIDTH*2 + 0] = (1-2) * uLB * (1 + 1e-4*sin(i/(FIELD_WIDTH-1)*0.5))/10; 
            temp_vel[i*2 + j*FIELD_WIDTH*2 + 1] = (1-2) * uLB * (1 + 1e-4*sin(j/(FIELD_HEIGHT-1)*0.5))/10;
            if(j == FIELD_WIDTH - 1 || j == 0){
                temp_vel[i*2 + j*FIELD_WIDTH*2 + 0] = 0;
                temp_vel[i*2 + j*FIELD_WIDTH*2 + 1] = 0;
            }
            }
    }

    //f_in
    cout << "f in\n";
    for (int i = 0; i < 9; i++) {
        for (int w = 0; w < FIELD_WIDTH; w++) {
            for (int h = 0; h < FIELD_HEIGHT; h++){
                double usqr = 3.0/2 * (temp_vel[w*2 + h*FIELD_WIDTH*2 + 0]*temp_vel[w*2 + h*FIELD_WIDTH*2 + 0] + temp_vel[w*2 + h*FIELD_WIDTH*2 + 1]*temp_vel[w*2 + h*FIELD_WIDTH*2 + 1]);
                double cu = 3 * (host_e[i*2 + 0]*temp_vel[w*2 + h*FIELD_WIDTH*2 + 0] + host_e[i*2 + 1]*temp_vel[w*2 + h*FIELD_WIDTH*2 + 1]);
                temp_f_in[w*9 + h*FIELD_WIDTH*9 + i] = 1 * host_weights[i] * (1 + cu + 0.5*cu*cu - usqr);
            }
        }
    }

    //density
    cout << "density\n";
    for (int i = 0; i < FIELD_WIDTH; i++) {
        for (int j = 0; j < FIELD_HEIGHT; j++) temp_density[i + j*FIELD_WIDTH] = 0.0001;
    }

    //obstacle
    cout << "obstacle\n";
    vector<vector<int>> vec_obstacle;
    for(int w = 0; w < FIELD_WIDTH; w++){
        for(int h = 0; h < FIELD_HEIGHT; h++){
            if((w - FIELD_WIDTH/3)*(w - FIELD_WIDTH/3) + (h-FIELD_HEIGHT/2)*(h-FIELD_HEIGHT/2) < r*r){
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

    cudaMalloc(&f_in, 1920 * 1080 * 9 * sizeof(float));
    cudaMalloc(&f_out, 1920 * 1080 * 9 * sizeof(float));
    cudaMalloc(&vel, 1920 * 1080 * 2 * sizeof(float));
    cudaMalloc(&density, 1920 * 1080 * sizeof(float));
    cudaMalloc(&e, 9 * 2 * sizeof(float));
    cudaMalloc(&weights, 9*sizeof(float));
    cudaMalloc(&visual_buffer, FIELD_WIDTH*FIELD_HEIGHT*sizeof(unsigned char));
    cudaMalloc(&obstacle, obstacle_size*2*sizeof(float));

    cudaMemcpy(f_in, temp_f_in, 1920 * 1080 * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(f_out, temp_f_out, 1920 * 1080 * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vel, temp_vel, 1920 * 1080 * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(density, temp_density, 1920 * 1080 * sizeof(float), cudaMemcpyHostToDevice);
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
    dim3 blockSize(BLOCK_X, BLOCK_Y);
    dim3 gridSize(BLOCKS_WIDTH, BLOCKS_HEIGHT);
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
    dim3 blockSize(BLOCK_X, BLOCK_Y);
    dim3 gridSize(BLOCKS_WIDTH, BLOCKS_HEIGHT);
    convert_visual<<<gridSize, blockSize>>>(vel, visual_buffer, 0.1);
    cudaDeviceSynchronize();
    return visual_buffer;
}