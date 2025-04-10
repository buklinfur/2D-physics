#ifndef VISUALIZE
#define VISUALIZE
#include <SFML/Graphics.hpp>
#include <SFML/Window/Event.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
typedef std::vector<double> vec;
typedef unsigned int uint;

typedef struct {
    uint left_padding, right_padding;
    uint step;
    uint kernel_size;
} grid_info;

class visualize{
    private:
    sf::RenderWindow window_object;
    uint window_width, window_height;
    uint grid_width, grid_height;
    grid_info grid_x;
    grid_info grid_y;
    bool shrinked;
    public:
    
    visualize();
    bool is_shrinked();
    void make_window(uint width, uint height);
    void make_grid(uint grid_width, uint grid_height); 
    void draw_field(const vec& density);
    void draw_shrinked_field(const vec& density);
    void close();
};

class visual_facade{
    private:
    visualize object;
    public:
    visual_facade(uint window_width, uint window_height,uint grid_width,uint grid_height);
    void show(const vec& density);
    void close();
};


void throwException(char text[]);
grid_info calculate_grid(uint w_size,uint g_size);


#endif //VISUALIZE