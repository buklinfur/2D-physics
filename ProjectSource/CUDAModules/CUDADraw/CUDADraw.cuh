#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

/**
 * @class GLVisual
 * @brief Manages an OpenGL window and CUDA‐registered texture for drawing.
 */
class GLVisual {
private:
    GLuint textureID;
    cudaGraphicsResource* cuda_tex_resource = nullptr;
    GLFWwindow* window = nullptr;
    int width, height;

public:
    /**
     * @brief Initializes GLFW, creates a window, sets up a CUDA‐registered texture.
     * @param name   Window title.
     * @param width  Width of the window and texture.
     * @param height Height of the window and texture.
     * @param vsync  VSync flag (0 = off, >0 = on).
     */
    GLVisual(const char* name, int width, int height, int vsync = 1);

    /**
     * @brief Cleans up CUDA resources and destroys the GLFW window.
     */
    ~GLVisual();

    /**
     * @brief Renders the given CUDA buffer to the window texture and displays it.
     * @param cuda_buffer Device pointer containing pixel data (one byte per texel).
     */
    void draw(unsigned char* cuda_buffer);

    /**
     * @brief Checks whether the window is still open.
     * @return True if the window has not been signaled to close.
     */
    bool alive();
};