#include "CUDADraw.cuh"
#include <iostream>
/** 
 * @class GLVisual
 * @brief Manages an OpenGL window and CUDA‐registered texture for drawing.
 */

/**
 * @brief Initializes GLFW, creates a window, sets up a CUDA‐registered texture.
 * @param name   Window title.
 * @param width  Width of the window and texture.
 * @param height Height of the window and texture.
 * @param vsync  VSync flag (0 = off, >0 = on).
 */
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

/**
 * @brief Cleans up CUDA resources and destroys the GLFW window.
 */
GLVisual::~GLVisual() {
    if (cuda_tex_resource) {
        cudaGraphicsUnregisterResource(cuda_tex_resource);
    }
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

/**
 * @brief Renders the given CUDA buffer to the window texture and displays it.
 * @param cuda_buffer Device pointer containing pixel data (one byte per texel).
 */
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

/**
 * @brief Checks whether the window is still open.
 * @return True if the window has not been signaled to close.
 */
bool GLVisual::alive() {
    return !glfwWindowShouldClose(window);
}