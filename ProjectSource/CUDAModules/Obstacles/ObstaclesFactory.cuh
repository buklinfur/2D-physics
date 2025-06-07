#pragma once

#include "Obstacles.cuh"
#include <memory>
#include <vector>

class ObstaclesFactory {
public:
    ObstaclesFactory(size_t width, size_t height);
    ~ObstaclesFactory();

    void add_circle(float center_x, float center_y, float radius);
    void add_rectangle(float center_x, float center_y, float width, float height);
    void add_custom(const std::vector<std::pair<int, int>>& points);

    bool* get_obstacle_mask() const;
    void clear_obstacles();

private:
    size_t width_, height_;
    std::vector<std::unique_ptr<Obstacle>> obstacles_;
    bool* obstacle_mask_;
};