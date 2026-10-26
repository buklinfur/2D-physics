#include "CUDAModules/CUDAFacade/CUDAFacade.cuh"
#include "UpdateLoader/UpdateLoader.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

/** @brief Command-line options. Defaults reproduce the original demo scene. */
struct Options {
    size_t width = 1920;
    size_t height = 1080;
    int block = 16;
    float uLB = 0.04f;
    float Re = 10.0f;
    float lim = 0.1f;
    int vsync = 1;

    bool no_update = false;

    bool benchmark = false;
    bool render = false;
    bool csv = false;
    int steps = 1000;
    int warmup = 100;

    std::string frames_dir;
    int every = 10;
};

void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "\n"
        "Interactive mode (default): opens a window and runs until it is closed.\n"
        "\n"
        "Simulation:\n"
        "  --size WxH          grid size in lattice nodes (default 1920x1080)\n"
        "  --re R              Reynolds number (default 10)\n"
        "  --ulb U             inflow lattice velocity (default 0.04)\n"
        "  --block B           CUDA block edge, B x B threads (default 16)\n"
        "  --lim L             velocity mapped to full brightness (default 0.1)\n"
        "  --vsync 0|1         interactive mode only (default 1)\n"
        "  --no-update         skip the GitHub release check\n"
        "\n"
        "Benchmark (implies --no-update):\n"
        "  --benchmark         run a fixed number of steps and report timing\n"
        "  --steps N           timed steps (default 1000)\n"
        "  --warmup N          untimed steps before timing (default 100)\n"
        "  --render            also draw every step to a window, vsync off\n"
        "                      (default: compute only, no window is created)\n"
        "  --csv               print one CSV line instead of a report\n"
        "\n"
        "Frame export (headless, implies --no-update):\n"
        "  --dump-frames DIR   write PGM frames to DIR, then exit\n"
        "  --steps N           total steps (default 1000)\n"
        "  --every K           save every K-th step (default 10)\n"
        "\n"
        "The scene is one circular obstacle at (W/3, H/2) with radius H*100/1080.\n";
}

/** @brief Parse "1920x1080" into width and height; throws on malformed input. */
void parse_size(const std::string& text, size_t& w, size_t& h) {
    const size_t x = text.find('x');
    if (x == std::string::npos) {
        throw std::invalid_argument("--size expects WxH, got '" + text + "'");
    }
    const long pw = std::stol(text.substr(0, x));
    const long ph = std::stol(text.substr(x + 1));
    if (pw < 16 || ph < 16) {
        throw std::invalid_argument("--size: both dimensions must be at least 16");
    }
    w = static_cast<size_t>(pw);
    h = static_cast<size_t>(ph);
}

Options parse_args(int argc, char** argv) {
    Options o;
    auto value = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string(argv[i]) + " needs a value");
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--no-update")    { o.no_update = true; }
        else if (a == "--size")         { parse_size(value(i), o.width, o.height); }
        else if (a == "--re")           { o.Re = std::stof(value(i)); }
        else if (a == "--ulb")          { o.uLB = std::stof(value(i)); }
        else if (a == "--block")        { o.block = std::stoi(value(i)); }
        else if (a == "--lim")          { o.lim = std::stof(value(i)); }
        else if (a == "--vsync")        { o.vsync = std::stoi(value(i)); }
        else if (a == "--benchmark")    { o.benchmark = true; }
        else if (a == "--render")       { o.render = true; }
        else if (a == "--csv")          { o.csv = true; }
        else if (a == "--steps")        { o.steps = std::stoi(value(i)); }
        else if (a == "--warmup")       { o.warmup = std::stoi(value(i)); }
        else if (a == "--dump-frames")  { o.frames_dir = value(i); }
        else if (a == "--every")        { o.every = std::stoi(value(i)); }
        else { throw std::invalid_argument("unknown option '" + a + "' (see --help)"); }
    }

    if (o.benchmark && !o.frames_dir.empty()) {
        throw std::invalid_argument("--benchmark and --dump-frames cannot be combined");
    }
    if (o.render && !o.benchmark) {
        throw std::invalid_argument("--render only applies to --benchmark");
    }
    if (o.block <= 0 || o.block > 32) {
        throw std::invalid_argument("--block must be in 1..32 (B x B threads, at most 1024)");
    }
    if (o.benchmark || !o.frames_dir.empty()) {
        o.no_update = true;
    }
    return o;
}

/** @brief Offer to install a newer release; only returns if no update was installed. */
void offer_update(char** argv) {
    if (!UpdateManager::CheckForUpdate()) {
        return;
    }
    std::cout << "\n\033[1;32mA new version is available!\033[0m\n"
              << "Current version: " << UpdateManager::GetCurrentVersion() << "\n"
              << "Latest version : " << UpdateManager::GetLatestVersion() << "\n"
              << "\nWould you like to update now? [Y/n]: ";

    std::string choice;
    while (std::getline(std::cin, choice)) {
        if (choice.empty() || choice == "y" || choice == "Y") {
            std::cout << "\nDownloading update...\n";
            if (UpdateManager::DownloadUpdate()) {
                UpdateManager::ApplyUpdate(argv);  // execv()s the new binary on success
            }
            std::cerr << "Update failed, continuing with the current version.\n\n";
            return;
        }
        if (choice == "n" || choice == "N") {
            std::cout << "Skipping update. Continuing with current version.\n\n";
            return;
        }
        std::cout << "Please enter 'Y' (yes) or 'N' (no): ";
    }
}

void print_device_info() {
    int device = 0;
    cudaDeviceProp prop{};
    int driver = 0, runtime = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("# GPU: %s (CC %d.%d, %d SMs, %.1f GiB)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("# CUDA driver %d.%d, runtime %d.%d\n", driver / 1000, (driver % 1000) / 10,
                runtime / 1000, (runtime % 1000) / 10);
}

void add_default_scene(CUDAFacade& sim, const Options& o) {
    // At 1920x1080 this is the original demo: circle at (640, 540), radius 100.
    sim.add_circle(o.width / 3.0f, o.height / 2.0f, o.height * (100.0f / 1080.0f));
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }

    if (!o.csv) {
        std::cout << "LBM Simulation (PID: " << getpid() << ")" << std::endl;
    }
    if (!o.no_update) {
        offer_update(argv);
    }

    try {
        const bool headless = !o.frames_dir.empty() || (o.benchmark && !o.render);
        const int vsync = o.benchmark ? 0 : o.vsync;  // never cap benchmark FPS by the display

        CUDAFacade sim(o.width, o.height, o.block, o.uLB, o.Re, FlowDirection::LEFT_TO_RIGHT,
                       "LBMSim", vsync, o.lim, headless);
        add_default_scene(sim, o);

        if (o.benchmark) {
            const BenchmarkResult r = sim.benchmark(o.steps, o.warmup, o.render);
            if (o.csv) {
                std::printf("%zu,%zu,%d,%s,%.6f,%.4f,%.2f,%.2f,%s\n", r.width, r.height, r.steps,
                            r.render ? "render" : "compute", r.total_s, r.ms_per_step, r.mlups,
                            r.steps_per_s, r.finite ? "ok" : "nan");
            } else {
                print_device_info();
                std::printf("grid        : %zu x %zu (%.2f M nodes)\n", r.width, r.height,
                            r.width * r.height / 1e6);
                std::printf("mode        : %s\n", r.render ? "compute + render (vsync off)" : "compute only");
                std::printf("steps       : %d timed, %d warm-up\n", r.steps, o.warmup);
                std::printf("total       : %.3f s\n", r.total_s);
                std::printf("per step    : %.3f ms\n", r.ms_per_step);
                std::printf("throughput  : %.1f MLUPS, %.1f steps/s\n", r.mlups, r.steps_per_s);
                std::printf("field check : %s\n", r.finite ? "finite" : "NaN/Inf detected (unstable run)");
            }
            return r.finite ? 0 : 3;
        }

        if (!o.frames_dir.empty()) {
            const int n = sim.dump_frames(o.steps, o.every, o.frames_dir);
            std::cout << "Wrote " << n << " frames to " << o.frames_dir << std::endl;
            return 0;
        }

        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
