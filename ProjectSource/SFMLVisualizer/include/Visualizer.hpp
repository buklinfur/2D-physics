#ifndef VISUALIZE
#define VISUALIZE

#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>

namespace Visualizer {
    class Visualizer {
    private:
        sf::RenderWindow window_object;
        uint window_width, window_height;
        uint grid_width, grid_height;
        grid_info grid_x;
        grid_info grid_y;
        bool shrinked;

    public:

        Visualier();

        bool isShrinked();

        void makeWindow(uint width, uint height);
        void makeGrid(uint grid_width, uint grid_height);

        void drawField(const vec& density);
        void drawShrinked_field(const vec& density);

        void closeWindow();

        ~Visualizer();
    };
}

#endif //VISUALIZE