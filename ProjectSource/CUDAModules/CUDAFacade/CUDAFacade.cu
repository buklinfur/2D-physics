#include "CUDAFacade.cuh"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

CUDAFacade::CUDAFacade(size_t width, size_t height, int block_size, float uLB, float Re,
                       FlowDirection flow_dir, const char* window_name, int vsync, float lim,
                       bool headless)
    : width_(width), height_(height), visuals_(nullptr) {

    obstacles_factory_ = new ObstaclesFactory(width, height);
    computing_ = new Field(width, height, block_size, uLB, Re, flow_dir,
                           obstacles_factory_->get_obstacle_mask());
    if (!headless) {
        visuals_ = new GLVisual(window_name, width, height, vsync);
        if (!visuals_->ok()) {
            delete visuals_;
            delete computing_;
            delete obstacles_factory_;
            throw std::runtime_error("Could not create an OpenGL window or register it with CUDA "
                                     "(no display? headless runs: --benchmark without --render, or --dump-frames)");
        }
    }
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
    if (!visuals_) {
        throw std::runtime_error("run() needs a window; this facade was created headless");
    }
    while (visuals_->alive()) {
        computing_->step();
        unsigned char* visual_data = computing_->get_visual(lim_); 
        if (visual_data) visuals_->draw(visual_data);
    }
}

BenchmarkResult CUDAFacade::benchmark(int steps, int warmup, bool render) {
    if (steps <= 0) {
        throw std::invalid_argument("benchmark: steps must be positive");
    }
    if (render && !visuals_) {
        throw std::runtime_error("benchmark: render mode needs a window; this facade was created headless");
    }

    auto one_step = [&]() {
        computing_->step();
        if (render) {
            visuals_->draw(computing_->get_visual(lim_));
        }
    };

    for (int i = 0; i < warmup; ++i) {
        one_step();
    }
    cudaDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        one_step();
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("benchmark: CUDA error: ") + cudaGetErrorString(err));
    }

    BenchmarkResult r;
    r.width = width_;
    r.height = height_;
    r.steps = steps;
    r.render = render;
    r.total_s = std::chrono::duration<double>(t1 - t0).count();
    r.ms_per_step = r.total_s * 1e3 / steps;
    r.mlups = static_cast<double>(width_) * height_ * steps / r.total_s / 1e6;
    r.steps_per_s = steps / r.total_s;
    r.finite = computing_->velocity_is_finite();
    return r;
}

int CUDAFacade::dump_frames(int steps, int every, const std::string& dir) {
    if (steps <= 0 || every <= 0) {
        throw std::invalid_argument("dump_frames: steps and every must be positive");
    }
    std::filesystem::create_directories(dir);

    std::vector<unsigned char> frame(width_ * height_);
    int written = 0;
    for (int i = 1; i <= steps; ++i) {
        computing_->step();
        if (i % every != 0) {
            continue;
        }
        unsigned char* device_frame = computing_->get_visual(lim_);
        cudaError_t err = cudaMemcpy(frame.data(), device_frame, frame.size(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("dump_frames: cudaMemcpy failed: ") + cudaGetErrorString(err));
        }

        char name[32];
        std::snprintf(name, sizeof(name), "frame_%05d.pgm", written);
        const std::string path = (std::filesystem::path(dir) / name).string();
        FILE* out = std::fopen(path.c_str(), "wb");
        if (!out) {
            throw std::runtime_error("dump_frames: cannot open " + path);
        }
        std::fprintf(out, "P5\n%zu %zu\n255\n", width_, height_);
        // Row 0 of the buffer is the bottom of the window (OpenGL texture origin),
        // while PGM rows go top to bottom, so write rows in reverse.
        for (size_t y = height_; y-- > 0;) {
            std::fwrite(frame.data() + y * width_, 1, width_, out);
        }
        std::fclose(out);
        ++written;
    }
    return written;
}
