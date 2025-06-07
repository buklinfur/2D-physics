#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

class GLVisual {
private:
    GLuint textureID;
    cudaGraphicsResource* cuda_tex_resource = nullptr;
    GLFWwindow* window = nullptr;
    int width, height; // Renamed from WIDTH, HEIGHT

public:
    GLVisual(const char* name, int width, int height, int vsync = 1); // Changed char name[] to const char*
    ~GLVisual();

    void draw(unsigned char* cuda_buffer); // Changed void* to unsigned char*
    bool alive();
};