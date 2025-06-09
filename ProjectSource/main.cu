#include "CUDAModules/CUDAFacade/CUDAFacade.cuh"
#include "UpdateLoader/UpdateLoader.hpp"
#include <iostream>
#include <chrono>
#include <unistd.h>

using namespace std;

int main() {
    std::cout << "LBM Simulation Version: " << UpdateManager::GetCurrentVersion() 
              << " (PID: " << getpid() << ")" << std::endl;

    // Update check with cooldown
    static std::chrono::steady_clock::time_point lastCheck;
    auto now = std::chrono::steady_clock::now();
    
    if (lastCheck.time_since_epoch().count() == 0 || 
        now - lastCheck > std::chrono::hours(24)) {
        
        lastCheck = now;
        if (UpdateManager::CheckForUpdate()) {
            std::cout << "Update available. Downloading..." << std::endl;
            if (UpdateManager::DownloadUpdate()) {
                std::cout << "Update ready. Restarting..." << std::endl;
                UpdateManager::ApplyUpdate();
                // Should never reach here
                return 1;
            }
        }
    }
    try {
        CUDAFacade sim(1240, 1080, 16, 0.04f, 10.0f, FlowDirection::LEFT_TO_RIGHT, "ok computer");

        sim.add_circle(640.0f, 540.0f, 100.0f); // Circle at (640, 540), radius 100
        // sim.add_rectangle(1280.0f, 540.0f, 200.0f, 100.0f); // Rectangle at (1280, 540) 
        // sim.add_custom({{1000, 1000}, {1000, 101}, {101, 1000}, {101, 101}}); // 4 dots

        sim.run();
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}