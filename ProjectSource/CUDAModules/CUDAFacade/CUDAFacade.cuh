#pragma once

#include "../CUDADraw/CUDADraw.cuh"
#include "../CUDAGrid/CUDAGrid.cuh"
#include "../Obstacles/ObstaclesFactory.cuh"
#include <string>

/**
 * @brief Timing of a fixed-length benchmark run (see CUDAFacade::benchmark).
 */
struct BenchmarkResult {
    size_t width = 0, height = 0; ///< Grid size in lattice nodes.
    int steps = 0;                ///< Timed steps (warm-up excluded).
    bool render = false;          ///< True if every step was also drawn to the window.
    double total_s = 0.0;         ///< Wall-clock time of the timed steps, seconds.
    double ms_per_step = 0.0;     ///< Average wall-clock time of one step, milliseconds.
    double mlups = 0.0;           ///< Million lattice-node updates per second.
    double steps_per_s = 0.0;     ///< Steps per second (equals FPS when render is true).
    bool finite = false;          ///< True if the velocity field has no NaN/Inf after the run.
};

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
     * @param headless If true, no window or OpenGL context is created; only
     *                 benchmark(render = false) and dump_frames() are available.
     */
    CUDAFacade(size_t width, size_t height, int block_size, float uLB, float Re,
              FlowDirection flow_dir, const char* window_name, int vsync = 1, float lim = 0.1f,
              bool headless = false);

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

    /**
     * @brief Run a fixed number of steps and measure wall-clock throughput.
     *
     * The device is synchronized before and after the timed loop. Each step is
     * the same Field::step() used by run(), so the numbers describe the
     * application as shipped, not an isolated kernel.
     *
     * @param steps  Number of timed steps.
     * @param warmup Untimed steps executed first (JIT, caches, clocks).
     * @param render If true, every step is also converted to an image and drawn
     *               to the window (requires a non-headless facade).
     * @return Timing summary, including a NaN/Inf check of the final field.
     */
    BenchmarkResult benchmark(int steps, int warmup, bool render);

    /**
     * @brief Run the simulation without a window and save frames as PGM images.
     *
     * Frames are the same velocity-magnitude image that run() displays,
     * written as 8-bit grayscale `frame_00000.pgm`, `frame_00001.pgm`, ...
     * with the image origin at the bottom, as on screen.
     *
     * @param steps Total number of simulation steps.
     * @param every Save one frame every `every` steps.
     * @param dir   Output directory (created if missing).
     * @return Number of frames written.
     */
    int dump_frames(int steps, int every, const std::string& dir);

private:
    size_t width_, height_;
    GLVisual* visuals_; 
    Field* computing_;
    ObstaclesFactory* obstacles_factory_;
    float lim_;
};