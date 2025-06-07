#include "CUDADraw.cuh"
#include <iostream>
GLVisual::GLVisual(char name[], int width, int height) : 
    WIDTH(width), HEIGHT(height), window(nullptr), cuda_tex_resource(nullptr) {
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }

    // Create window and OpenGL context
    window = glfwCreateWindow(WIDTH, HEIGHT, name, NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window\n";
        return;
    }
    
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        return;
    }

    // Create texture
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, WIDTH, HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

    // Register with CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_tex_resource, textureID, 
                                                GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register texture: " << cudaGetErrorString(err) << "\n";
    }
}

GLVisual::~GLVisual() {
    if (cuda_tex_resource) {
        cudaGraphicsUnregisterResource(cuda_tex_resource);
        cuda_tex_resource = nullptr;
    }

    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
        window = nullptr;
    }
}

void GLVisual::draw(void *cuda_buffer){
    glClear(GL_COLOR_BUFFER_BIT);
    // Подключаем ресурс CUDA
    cudaGraphicsMapResources(1, &cuda_tex_resource, 0);
        
    // Получаем указатель на текстуру
    cudaArray_t texture_ptr;
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0);
    
    // Копируем данные из CUDA буфера в текстуру
    cudaMemcpyToArray(texture_ptr, 0, 0, cuda_buffer, WIDTH * HEIGHT, cudaMemcpyDeviceToDevice);
    
    // Отключаем ресурс
    cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0);
        
    // Рисуем квад с текстурой
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
        
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool GLVisual::alive(){
    return !glfwWindowShouldClose(window);
}