#include "CUDAFacade.cuh"
#include <stdexcept>

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
CUDAFacade::CUDAFacade(size_t width, size_t height, int block_size, float uLB, float Re,
                       FlowDirection flow_dir, const char* window_name, int vsync, float lim)
    : width_(width), height_(height) {

    obstacles_factory_ = new ObstaclesFactory(width, height);
    computing_ = new Field(width, height, block_size, uLB, Re, flow_dir,
                           obstacles_factory_->get_obstacle_mask());
    visuals_ = new GLVisual(window_name, width, height, vsync);
    lim_ = lim;
}

/**
 * @brief Destructor for CUDAFacade.
 *
 * Cleans up and deallocates memory for:
 * - The Field simulation object
 * - The GLVisual visualization object
 * - The ObstaclesFactory geometry generator
 */
CUDAFacade::~CUDAFacade() {
    delete visuals_;
    delete computing_;
    delete obstacles_factory_;
}

/**
 * @brief Add a circular obstacle to the simulation domain.
 *
 * Updates the obstacle mask and uploads it to the GPU.
 *
 * @param center_x X-coordinate of the circle center.
 * @param center_y Y-coordinate of the circle center.
 * @param radius Radius of the circle.
 */
void CUDAFacade::add_circle(float center_x, float center_y, float radius) {
    obstacles_factory_->add_circle(center_x, center_y, radius);
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

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
void CUDAFacade::add_rectangle(float center_x, float center_y, float width, float height) {
    obstacles_factory_->add_rectangle(center_x, center_y, width, height);
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

/**
 * @brief Add a custom obstacle defined by specific grid points.
 *
 * Updates the obstacle mask and uploads it to the GPU.
 *
 * @param points A vector of (x, y) pairs defining solid cells.
 */
void CUDAFacade::add_custom(const std::vector<std::pair<int, int>>& points) {
    obstacles_factory_->add_custom(points);
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

/**
 * @brief Remove all obstacles from the simulation domain.
 *
 * Clears the obstacle mask and uploads the updated state to the GPU.
 */
void CUDAFacade::clear_obstacles() {
    obstacles_factory_->clear_obstacles();
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

/**
 * @brief Start and run the main simulation loop with visualization.
 *
 * While the visualization window is active:
 * - Perform one simulation step (collision, streaming)
 * - Retrieve velocity field and render it using OpenGL
 */
void CUDAFacade::run() {
    while (visuals_->alive()) {
        computing_->step();
        unsigned char* visual_data = computing_->get_visual(lim_); 
        if (visual_data) visuals_->draw(visual_data);
    }
}