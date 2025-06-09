#include "ObstaclesFactory.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

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
ObstaclesFactory::ObstaclesFactory(size_t width, size_t height) : width_(width), height_(height) {
    cudaError_t err = cudaMalloc(&obstacle_mask_, width * height * sizeof(bool));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate obstacle_mask: " + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(obstacle_mask_, 0, width * height * sizeof(bool));
}

/**
 * @brief Destructor that frees the CUDA device memory for the obstacle mask.
 */
ObstaclesFactory::~ObstaclesFactory() {
    if (obstacle_mask_) {
        cudaFree(obstacle_mask_);
    }
}

/**
 * @brief Adds a circular obstacle to the domain and updates the device obstacle mask.
 *
 * @param center_x X-coordinate of the circle's center.
 * @param center_y Y-coordinate of the circle's center.
 * @param radius Radius of the circle.
 *
 * @throws std::runtime_error if the mask update fails during CUDA memory copy.
 */
void ObstaclesFactory::add_circle(float center_x, float center_y, float radius) {
    obstacles_.push_back(std::make_unique<Circle>(center_x, center_y, radius));
    bool* host_mask = new bool[width_ * height_]();
    obstacles_.back()->apply_to_mask(host_mask, width_, height_);
    cudaError_t err = cudaMemcpy(obstacle_mask_, host_mask, width_ * height_ * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        delete[] host_mask;
        throw std::runtime_error("Failed to copy circle mask: " + std::string(cudaGetErrorString(err)));
    }
    delete[] host_mask;
}

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
void ObstaclesFactory::add_rectangle(float center_x, float center_y, float width, float height) {
    obstacles_.push_back(std::make_unique<Rectangle>(center_x, center_y, width, height));
    bool* host_mask = new bool[width_ * height_]();
    obstacles_.back()->apply_to_mask(host_mask, width_, height_);
    cudaError_t err = cudaMemcpy(obstacle_mask_, host_mask, width_ * height_ * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        delete[] host_mask;
        throw std::runtime_error("Failed to copy rectangle mask: " + std::string(cudaGetErrorString(err)));
    }
    delete[] host_mask;
}

/**
 * @brief Adds a custom obstacle defined by a set of grid points and updates the device mask.
 *
 * @param points A vector of (x, y) pairs marking the solid grid points.
 *
 * @throws std::runtime_error if the mask update fails during CUDA memory copy.
 */
void ObstaclesFactory::add_custom(const std::vector<std::pair<int, int>>& points) {
    obstacles_.push_back(std::make_unique<Custom>(points));
    bool* host_mask = new bool[width_ * height_]();
    obstacles_.back()->apply_to_mask(host_mask, width_, height_);
    cudaError_t err = cudaMemcpy(obstacle_mask_, host_mask, width_ * height_ * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        delete[] host_mask;
        throw std::runtime_error("Failed to copy custom mask: " + std::string(cudaGetErrorString(err)));
    }
    delete[] host_mask;
}

/**
 * @brief Retrieves the device pointer to the current obstacle mask.
 *
 * @return A device pointer to the boolean mask indicating obstacle cells.
 */
bool* ObstaclesFactory::get_obstacle_mask() const {
    return obstacle_mask_;
}

/**
 * @brief Removes all previously added obstacles and resets the obstacle mask on the device.
 *
 * @throws std::runtime_error if clearing the CUDA mask fails.
 */
void ObstaclesFactory::clear_obstacles() {
    obstacles_.clear();
    cudaError_t err = cudaMemset(obstacle_mask_, 0, width_ * height_ * sizeof(bool));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to clear obstacle_mask: " + std::string(cudaGetErrorString(err)));
    }
}