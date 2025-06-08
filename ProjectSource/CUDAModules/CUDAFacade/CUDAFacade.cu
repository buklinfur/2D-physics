#include "CUDAFacade.cuh"
#include <stdexcept>

CUDAFacade::CUDAFacade(size_t width, size_t height, int block_size, float uLB, float Re,
                       FlowDirection flow_dir, const char* window_name, int vsync, float lim)
    : width_(width), height_(height) {

    obstacles_factory_ = new ObstaclesFactory(width, height);
    computing_ = new Field(width, height, block_size, uLB, Re, flow_dir,
                           obstacles_factory_->get_obstacle_mask());
    visuals_ = new GLVisual(window_name, width, height, vsync);
    lim_ = lim;
}

CUDAFacade::~CUDAFacade() {
    delete visuals_;
    delete computing_;
    delete obstacles_factory_;
}

void CUDAFacade::add_circle(float center_x, float center_y, float radius) {
    obstacles_factory_->add_circle(center_x, center_y, radius);
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

void CUDAFacade::add_rectangle(float center_x, float center_y, float width, float height) {
    obstacles_factory_->add_rectangle(center_x, center_y, width, height);
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

void CUDAFacade::add_custom(const std::vector<std::pair<int, int>>& points) {
    obstacles_factory_->add_custom(points);
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

void CUDAFacade::clear_obstacles() {
    obstacles_factory_->clear_obstacles();
    computing_->update_obstacle_mask(obstacles_factory_->get_obstacle_mask());
}

void CUDAFacade::run() {
    while (visuals_->alive()) {
        computing_->step();
        unsigned char* visual_data = computing_->get_visual(lim_); 
        if (visual_data) visuals_->draw(visual_data);
    }
}