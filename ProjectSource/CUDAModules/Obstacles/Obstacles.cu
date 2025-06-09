#include "Obstacles.cuh"
#include <stdexcept>
#include <cmath>

/**
 * @brief Construct a circular obstacle.
 *
 * @param center_x X-coordinate of the circle's center.
 * @param center_y Y-coordinate of the circle's center.
 * @param radius Radius of the circle (must be positive).
 *
 * @throws std::runtime_error if radius is non-positive.
 */
Circle::Circle(float center_x, float center_y, float radius)
    : center_x_(center_x), center_y_(center_y), radius_(radius) {
    if (radius <= 0) {
        throw std::runtime_error("Circle radius must be positive");
    }
}

/**
 * @brief Apply the circle shape to a boolean obstacle mask.
 *
 * For each grid cell, if it lies within the circle, sets the corresponding mask value to true.
 *
 * @param mask Pointer to the obstacle mask array (flattened 2D grid).
 * @param width Width of the simulation domain.
 * @param height Height of the simulation domain.
 */
void Circle::apply_to_mask(bool* mask, size_t width, size_t height) const {
    for (size_t w = 0; w < width; w++) {
        for (size_t h = 0; h < height; h++) {
            if ((w - center_x_) * (w - center_x_) + (h - center_y_) * (h - center_y_) < radius_ * radius_) {
                mask[h * width + w] = true;
            }
        }
    }
}

/**
 * @brief Construct a rectangular obstacle.
 *
 * @param center_x X-coordinate of the rectangle's center.
 * @param center_y Y-coordinate of the rectangle's center.
 * @param width Width of the rectangle (must be positive).
 * @param height Height of the rectangle (must be positive).
 *
 * @throws std::runtime_error if width or height is non-positive.
 */
Rectangle::Rectangle(float center_x, float center_y, float width, float height)
    : center_x_(center_x), center_y_(center_y), width_(width), height_(height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Rectangle dimensions must be positive");
    }
}

/**
 * @brief Apply the rectangle shape to a boolean obstacle mask.
 *
 * For each grid cell, if it lies within the rectangle bounds, sets the corresponding mask value to true.
 *
 * @param mask Pointer to the obstacle mask array (flattened 2D grid).
 * @param width Width of the simulation domain.
 * @param height Height of the simulation domain.
 */
void Rectangle::apply_to_mask(bool* mask, size_t width, size_t height) const {
    for (size_t w = std::max(0, int(center_x_ - width_ / 2)); 
         w < std::min(width, size_t(center_x_ + width_ / 2)); w++) {
        for (size_t h = std::max(0, int(center_y_ - height_ / 2)); 
             h < std::min(height, size_t(center_y_ + height_ / 2)); h++) {
            mask[h * width + w] = true;
        }
    }
}

/**
 * @brief Construct a custom obstacle from a list of specific grid points.
 *
 * @param points A vector of (x, y) integer grid coordinates marking solid cells.
 *
 * @throws std::runtime_error if the point list is empty.
 */
Custom::Custom(const std::vector<std::pair<int, int>>& points) : points_(points) {
    if (points.empty()) {
        throw std::runtime_error("Custom obstacle must have at least one point");
    }
}

/**
 * @brief Apply the custom shape to a boolean obstacle mask.
 *
 * Marks each explicitly listed (x, y) point as a solid cell in the obstacle mask.
 * Skips points that fall outside the simulation domain.
 *
 * @param mask Pointer to the obstacle mask array (flattened 2D grid).
 * @param width Width of the simulation domain.
 * @param height Height of the simulation domain.
 */
void Custom::apply_to_mask(bool* mask, size_t width, size_t height) const {
    for (const auto& point : points_) {
        if (point.first >= 0 && point.first < width && point.second >= 0 && point.second < height) {
            mask[point.second * width + point.first] = true;
        }
    }
}