#include "CUDAModules/CUDAFacade/CUDAFacade.cuh"
#include "UpdateLoader/UpdateLoader.hpp"
#include <iostream>
#include <chrono>
#include <unistd.h>

using namespace std;

int main(int argc, char** argv) {
    bool skipUpdate = (argc > 1 && std::string(argv[1]) == "--no-update");

    std::cout << "LBM Simulation (PID: " << getpid() << ")" << std::endl;


    if (!skipUpdate) {
        static std::chrono::steady_clock::time_point lastCheck;
        auto now = std::chrono::steady_clock::now();

        if (lastCheck.time_since_epoch().count() == 0 ||
            now - lastCheck > std::chrono::hours(24)) {

            lastCheck = now;

            if (UpdateManager::CheckForUpdate()) {
                std::cout << "\n\033[1;32m🚀 A new version is available!\033[0m\n";
                std::cout << "Current version: " << UpdateManager::GetCurrentVersion() << "\n";
                std::cout << "Latest version : " << UpdateManager::GetLatestVersion() << "\n";
                std::cout << "\nWould you like to update now? [Y/n]: ";

                std::string choice;
                while (true) {
                    std::getline(std::cin, choice);
                    if (choice.empty() || choice == "y" || choice == "Y") {
                        std::cout << "\nDownloading update...\n";
                        if (UpdateManager::DownloadUpdate()) {
                            std::cout << "✅ Update ready. Restarting...\n";
                            UpdateManager::ApplyUpdate();
                            return 1;
                        } else {
                            std::cerr << "❌ Update failed.\n";
                            break;
                        }
                    } else if (choice == "n" || choice == "N") {
                        std::cout << "ℹ️  Skipping update. Continuing with current version.\n\n";
                        break;
                    } else {
                        std::cout << "Please enter 'Y' (yes) or 'N' (no): ";
                    }
                }
            }
        }
    }


    try {
        CUDAFacade sim(1920, 1080, 16, 0.04f, 10.0f, FlowDirection::LEFT_TO_RIGHT, "ok computer");

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