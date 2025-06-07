#pragma once

#include "../CUDADraw/CUDADraw.cuh"
#include "../CUDAGrid/CUDAGrid.cuh"
#include "../Obstacles/ObstaclesFactory.cuh"

class CUDAFacade {
public:
    CUDAFacade(size_t width, size_t height, int block_size, float uLB, float Re,
              FlowDirection flow_dir, const char* window_name, int vsync = 1);

    void add_circle(float center_x, float center_y, float radius);
    void add_rectangle(float center_x, float center_y, float width, float height);
    void add_custom(const std::vector<std::pair<int, int>>& points);

    void clear_obstacles();

    void run();

    ~CUDAFacade();

private:
    size_t width_, height_;
    GLVisual* visuals_; 
    Field* computing_;
    ObstaclesFactory* obstacles_factory_;
};