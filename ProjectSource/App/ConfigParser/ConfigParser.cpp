#include "ConfigParser.hpp"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

SimulationConfig ConfigParser::loadFromFile(const std::string& filename) {
    std::ifstream in{filename};
    if (!in) throw std::runtime_error("Cannot open config file: " + filename);

    nlohmann::json j;
    in >> j;

    SimulationConfig cfg;
    // (same parsing code as before, but here in .cpp)
    cfg.width     = j["simulation"]["grid"]["width"];
    cfg.height    = j["simulation"]["grid"]["height"];
    cfg.blockSize = j["simulation"]["grid"]["blockSize"];

    cfg.uLB           = j["simulation"]["flow"]["uLB"];
    cfg.Re            = j["simulation"]["flow"]["Re"];
    cfg.flowDirection = j["simulation"]["flow"]["direction"];

    cfg.windowName = j["simulation"]["display"]["windowName"];
    cfg.vsync      = j["simulation"]["display"]["vsync"];
    cfg.lim        = j["simulation"]["display"]["lim"];

    auto& obst = j["obstacles"];
    cfg.enableObstacles = obst.value("enable", false);
    if (cfg.enableObstacles && obst.contains("shapes")) {
        for (auto& item : obst["shapes"]) {
            ObstacleSpec s;
            s.type = item["type"];
            if (s.type == "circle") {
                s.centerX = item["centerX"];
                s.centerY = item["centerY"];
                s.radius  = item["radius"];
            } else if (s.type == "rectangle") {
                s.centerX = item["centerX"];
                s.centerY = item["centerY"];
                s.width   = item["width"];
                s.height  = item["height"];
            } else if (s.type == "custom") {
                for (auto& p : item["points"])
                    s.points.emplace_back(p[0].get<int>(), p[1].get<int>());
            } else {
                throw std::runtime_error("Unknown obstacle type: " + s.type);
            }
            cfg.obstacles.push_back(std::move(s));
        }
    }

    if (cfg.width == 0 || cfg.height == 0)
        throw std::runtime_error("Grid dimensions must be > 0");
    if (cfg.blockSize <= 0)
        throw std::runtime_error("blockSize must be > 0");

    return cfg;
}