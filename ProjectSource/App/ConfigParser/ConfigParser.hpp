#pragma once

#include <string>
#include <vector>
#include <utility>

struct ObstacleSpec {
    std::string type;
    float centerX, centerY;
    float radius;
    float width, height;
    std::vector<std::pair<int,int>> points;
};

struct SimulationConfig {
    size_t width, height;
    int    blockSize;
    float  uLB, Re;
    std::string flowDirection;
    std::string windowName;
    bool   vsync;
    float  lim;
    bool   enableObstacles;
    std::vector<ObstacleSpec> obstacles;
};

class ConfigParser {
public:
    static SimulationConfig loadFromFile(const std::string& filename);
};
