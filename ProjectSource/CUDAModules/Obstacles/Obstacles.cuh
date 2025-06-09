#pragma once

#include <vector>
#include <string>

class Obstacle {
public:
    virtual ~Obstacle() = default;
    virtual void apply_to_mask(bool* mask, size_t width, size_t height) const = 0;
};

class Circle : public Obstacle {
public:
    /**
     * @brief Construct a circular obstacle.
     *
     * @param center_x X-coordinate of the circle's center.
     * @param center_y Y-coordinate of the circle's center.
     * @param radius Radius of the circle (must be positive).
     *
     * @throws std::runtime_error if radius is non-positive.
     */
    Circle(float center_x, float center_y, float radius);

    /**
     * @brief Apply the circle shape to a boolean obstacle mask.
     *
     * For each grid cell, if it lies within the circle, sets the corresponding mask value to true.
     *
     * @param mask Pointer to the obstacle mask array (flattened 2D grid).
     * @param width Width of the simulation domain.
     * @param height Height of the simulation domain.
     */
    void apply_to_mask(bool* mask, size_t width, size_t height) const override;

private:
    float center_x_, center_y_, radius_;
};

class Rectangle : public Obstacle {
public:
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
    Rectangle(float center_x, float center_y, float width, float height);

    /**
     * @brief Apply the rectangle shape to a boolean obstacle mask.
     *
     * For each grid cell, if it lies within the rectangle bounds, sets the corresponding mask value to true.
     *
     * @param mask Pointer to the obstacle mask array (flattened 2D grid).
     * @param width Width of the simulation domain.
     * @param height Height of the simulation domain.
     */
    void apply_to_mask(bool* mask, size_t width, size_t height) const override;

private:
    float center_x_, center_y_, width_, height_;
};

class Custom : public Obstacle {
public:
    /**
     * @brief Construct a custom obstacle from a list of specific grid points.
     *
     * @param points A vector of (x, y) integer grid coordinates marking solid cells.
     *
     * @throws std::runtime_error if the point list is empty.
     */
    Custom(const std::vector<std::pair<int, int>>& points);

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
    void apply_to_mask(bool* mask, size_t width, size_t height) const override;

private:
    std::vector<std::pair<int, int>> points_;
};