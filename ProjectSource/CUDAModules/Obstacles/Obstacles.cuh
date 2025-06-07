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
    Circle(float center_x, float center_y, float radius);
    void apply_to_mask(bool* mask, size_t width, size_t height) const override;

private:
    float center_x_, center_y_, radius_;
};

class Rectangle : public Obstacle {
public:
    Rectangle(float center_x, float center_y, float width, float height);
    void apply_to_mask(bool* mask, size_t width, size_t height) const override;

private:
    float center_x_, center_y_, width_, height_;
};

class Custom : public Obstacle {
public:
    Custom(const std::vector<std::pair<int, int>>& points);
    void apply_to_mask(bool* mask, size_t width, size_t height) const override;

private:
    std::vector<std::pair<int, int>> points_;
};