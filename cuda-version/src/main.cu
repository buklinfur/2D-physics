#include "../CUDADraw/CUDADraw.cuh"
#include "../CUDAGrid/CUDAGrid.cuh"


int main() {
   
    Field pivo;
	GLVisual riba("ok computer", 1920, 1080);
    void *cuda_buffer;

	while(riba.alive()){
        pivo.step();
        cuda_buffer = pivo.get_visual();
		riba.draw(cuda_buffer);
	}    

    return 0;
}