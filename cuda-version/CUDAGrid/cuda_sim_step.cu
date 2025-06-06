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
    
    //if (x >= FIELD_WIDTH || y >= FIELD_HEIGHT) return;

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
