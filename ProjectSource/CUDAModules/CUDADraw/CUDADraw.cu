#include "CUDADraw.cuh"
#include <iostream>

GLVisual::GLVisual(const char* name, int width, int height, int vsync)
    : width(width), height(height), window(nullptr), cuda_tex_resource(nullptr) 
{
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }

    window = glfwCreateWindow(width, height, name, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(vsync);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        return;
    }

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

    cudaError_t err = cudaGraphicsGLRegisterImage(
        &cuda_tex_resource,
        textureID,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (err != cudaSuccess) {
        std::cerr << "Failed to register texture: " 
                  << cudaGetErrorString(err) << "\n";
    }
}

GLVisual::~GLVisual() {
    if (cuda_tex_resource) {
        cudaGraphicsUnregisterResource(cuda_tex_resource);
    }
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

void GLVisual::draw(unsigned char* cuda_buffer) {
    glClear(GL_COLOR_BUFFER_BIT);

    cudaGraphicsMapResources(1, &cuda_tex_resource, 0);

    cudaArray_t texture_ptr;
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0);

    cudaMemcpy2DToArray(
        texture_ptr,
        0, 0,
        cuda_buffer,
        width,
        width, height,
        cudaMemcpyDeviceToDevice
    );

    cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
      glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
      glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();
}


bool GLVisual::alive() {
    return !glfwWindowShouldClose(window);
}