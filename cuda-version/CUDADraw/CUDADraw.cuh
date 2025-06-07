#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>


class GLVisual{
private:
    GLuint textureID;
    cudaGraphicsResource* cuda_tex_resource = nullptr;
    GLFWwindow* window = nullptr;
    int WIDTH, HEIGHT;

public:
    GLVisual(char name[], int width, int height,int vsync=1);
    ~GLVisual();

    void draw(void *cuda_buffer);
    bool alive();
};
