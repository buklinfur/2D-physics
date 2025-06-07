#include "ObstaclesFactory.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

ObstaclesFactory::ObstaclesFactory(size_t width, size_t height) : width_(width), height_(height) {
    cudaError_t err = cudaMalloc(&obstacle_mask_, width * height * sizeof(bool));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate obstacle_mask: " + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(obstacle_mask_, 0, width * height * sizeof(bool));
}

ObstaclesFactory::~ObstaclesFactory() {
    if (obstacle_mask_) {
        cudaFree(obstacle_mask_);
    }
}

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

bool* ObstaclesFactory::get_obstacle_mask() const {
    return obstacle_mask_;
}

void ObstaclesFactory::clear_obstacles() {
    obstacles_.clear();
    cudaError_t err = cudaMemset(obstacle_mask_, 0, width_ * height_ * sizeof(bool));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to clear obstacle_mask: " + std::string(cudaGetErrorString(err)));
    }
}