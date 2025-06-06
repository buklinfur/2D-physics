#include "../CUDADraw/CUDADraw.cuh"

const unsigned int WIDTH = 512;
const unsigned int HEIGHT = 512;

void *cuda_buffer;
void initCUDA() {

    cudaMalloc(&cuda_buffer, WIDTH * HEIGHT * sizeof(unsigned char));
    

    unsigned char host_buffer[WIDTH * HEIGHT];
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            host_buffer[y * WIDTH + x] = x % 256;
        }
    }
    cudaMemcpy(cuda_buffer, host_buffer, WIDTH * HEIGHT, cudaMemcpyHostToDevice);
}

int main() {
   
    initCUDA();
    
	GLVisual riba("ok computer", 512, 512);
	while(riba.alive()){
		riba.draw(cuda_buffer);
	}    

    return 0;
}