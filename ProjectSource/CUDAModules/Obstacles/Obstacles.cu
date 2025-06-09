#include "Obstacles.cuh"
#include <stdexcept>
#include <cmath>

Circle::Circle(float center_x, float center_y, float radius)
    : center_x_(center_x), center_y_(center_y), radius_(radius) {
    if (radius <= 0) {
        throw std::runtime_error("Circle radius must be positive");
    }
}

void Circle::apply_to_mask(bool* mask, size_t width, size_t height) const {
    for (size_t w = 0; w < width; w++) {
        for (size_t h = 0; h < height; h++) {
            if ((w - center_x_) * (w - center_x_) + (h - center_y_) * (h - center_y_) < radius_ * radius_) {
                mask[h * width + w] = true;
            }
        }
    }
}

Rectangle::Rectangle(float center_x, float center_y, float width, float height)
    : center_x_(center_x), center_y_(center_y), width_(width), height_(height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Rectangle dimensions must be positive");
    }
}

void Rectangle::apply_to_mask(bool* mask, size_t width, size_t height) const {
    for (size_t w = std::max(0, int(center_x_ - width_ / 2)); 
         w < std::min(width, size_t(center_x_ + width_ / 2)); w++) {
        for (size_t h = std::max(0, int(center_y_ - height_ / 2)); 
             h < std::min(height, size_t(center_y_ + height_ / 2)); h++) {
            mask[h * width + w] = true;
        }
    }
}

Custom::Custom(const std::vector<std::pair<int, int>>& points) : points_(points) {
    if (points.empty()) {
        throw std::runtime_error("Custom obstacle must have at least one point");
    }
}

void Custom::apply_to_mask(bool* mask, size_t width, size_t height) const {
    for (const auto& point : points_) {
        if (point.first >= 0 && point.first < width && point.second >= 0 && point.second < height) {
            mask[point.second * width + point.first] = true;
        }
    }
}