#pragma once

#include "Obstacles.cuh"
#include <memory>
#include <vector>

class ObstaclesFactory {
public:
    /**
     * @brief Constructs a factory for managing and applying obstacles to the simulation domain.
     *
     * Allocates a device-side boolean mask representing solid cells for obstacles.
     *
     * @param width Width of the simulation grid.
     * @param height Height of the simulation grid.
     *
     * @throws std::runtime_error if CUDA memory allocation for the obstacle mask fails.
     */
    ObstaclesFactory(size_t width, size_t height);

    /**
     * @brief Destructor that frees the CUDA device memory for the obstacle mask.
     */
    ~ObstaclesFactory();

    /**
     * @brief Adds a circular obstacle to the domain and updates the device obstacle mask.
     *
     * @param center_x X-coordinate of the circle's center.
     * @param center_y Y-coordinate of the circle's center.
     * @param radius Radius of the circle.
     *
     * @throws std::runtime_error if the mask update fails during CUDA memory copy.
     */
    void add_circle(float center_x, float center_y, float radius);

    /**
     * @brief Adds a rectangular obstacle to the domain and updates the device obstacle mask.
     *
     * @param center_x X-coordinate of the rectangle's center.
     * @param center_y Y-coordinate of the rectangle's center.
     * @param width Width of the rectangle.
     * @param height Height of the rectangle.
     *
     * @throws std::runtime_error if the mask update fails during CUDA memory copy.
     */
    void add_rectangle(float center_x, float center_y, float width, float height);

    /**
     * @brief Adds a custom obstacle defined by a set of grid points and updates the device mask.
     *
     * @param points A vector of (x, y) pairs marking the solid grid points.
     *
     * @throws std::runtime_error if the mask update fails during CUDA memory copy.
     */
    void add_custom(const std::vector<std::pair<int, int>>& points);

    /**
     * @brief Retrieves the device pointer to the current obstacle mask.
     *
     * @return A device pointer to the boolean mask indicating obstacle cells.
     */
    bool* get_obstacle_mask() const;

    /**
     * @brief Removes all previously added obstacles and resets the obstacle mask on the device.
     *
     * @throws std::runtime_error if clearing the CUDA mask fails.
     */
    void clear_obstacles();

private:
    size_t width_, height_;
    std::vector<std::unique_ptr<Obstacle>> obstacles_;
    bool* obstacle_mask_;
};