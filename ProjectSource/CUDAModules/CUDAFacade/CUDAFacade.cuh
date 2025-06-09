#pragma once

#include "../CUDADraw/CUDADraw.cuh"
#include "../CUDAGrid/CUDAGrid.cuh"
#include "../Obstacles/ObstaclesFactory.cuh"


class CUDAFacade {
public:
    /**
     * @brief Construct a CUDAFacade object that manages simulation, obstacles, and visualization.
     *
     * Initializes and links together:
     * - The Field object (computational backend)
     * - The GLVisual object (OpenGL-based frontend)
     * - The ObstaclesFactory for defining domain geometry
     *
     * @param width Width of the simulation grid.
     * @param height Height of the simulation grid.
     * @param block_size CUDA block size used in kernels.
     * @param uLB Lattice velocity for simulation scaling.
     * @param Re Reynolds number for viscosity computation.
     * @param flow_dir Flow direction for inflow boundary condition.
     * @param window_name Title of the OpenGL visualization window.
     * @param vsync Vertical sync setting (1 = enabled, 0 = disabled).
     * @param lim Velocity magnitude limit for visualization scaling.
     */
    CUDAFacade(size_t width, size_t height, int block_size, float uLB, float Re,
              FlowDirection flow_dir, const char* window_name, int vsync = 1, float lim = 0.1f);

    /**
     * @brief Destructor for CUDAFacade.
     *
     * Cleans up and deallocates memory for:
     * - The Field simulation object
     * - The GLVisual visualization object
     * - The ObstaclesFactory geometry generator
     */
    ~CUDAFacade();

    /**
     * @brief Add a circular obstacle to the simulation domain.
     *
     * Updates the obstacle mask and uploads it to the GPU.
     *
     * @param center_x X-coordinate of the circle center.
     * @param center_y Y-coordinate of the circle center.
     * @param radius Radius of the circle.
     */
    void add_circle(float center_x, float center_y, float radius);

    /**
     * @brief Add a rectangular obstacle to the simulation domain.
     *
     * Updates the obstacle mask and uploads it to the GPU.
     *
     * @param center_x X-coordinate of the rectangle center.
     * @param center_y Y-coordinate of the rectangle center.
     * @param width Width of the rectangle.
     * @param height Height of the rectangle.
     */
    void add_rectangle(float center_x, float center_y, float width, float height);

    /**
    * @brief Add a custom obstacle defined by specific grid points.
    *
    * Updates the obstacle mask and uploads it to the GPU.
    *
    * @param points A vector of (x, y) pairs defining solid cells.
    */
    void add_custom(const std::vector<std::pair<int, int>>& points);

    /**
     * @brief Remove all obstacles from the simulation domain.
     *
     * Clears the obstacle mask and uploads the updated state to the GPU.
     */
    void clear_obstacles();

    /**
     * @brief Start and run the main simulation loop with visualization.
     *
     * While the visualization window is active:
     * - Perform one simulation step (collision, streaming)
     * - Retrieve velocity field and render it using OpenGL
     */
    void run();

private:
    size_t width_, height_;
    GLVisual* visuals_; 
    Field* computing_;
    ObstaclesFactory* obstacles_factory_;
    float lim_;
};