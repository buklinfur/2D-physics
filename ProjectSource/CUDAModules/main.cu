#include "CUDAFacade/CUDAFacade.cuh"
#include <iostream>

using namespace std;

int main() {
    try {
        CUDAFacade sim(1920, 1080, 16, 0.04f, 10.0f, FlowDirection::LEFT_TO_RIGHT, "ok computer");

        sim.add_circle(640.0f, 540.0f, 100.0f); // Circle at (640, 540), radius 100s
        // sim.add_rectangle(1280.0f, 540.0f, 200.0f, 100.0f); // Rectangle at (1280, 540) 
        // sim.add_custom({{1000, 1000}, {1000, 101}, {101, 1000}, {101, 101}}); // 4 dots

        sim.run();
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}