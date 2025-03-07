#include "visualize.hpp"



void throwException(char text[]){
    try{
        throw(text);
    }
    catch(char what[]){
        std::cout << what;
    }
}

grid_info calculate_grid(uint w_size,uint g_size,bool& shrinked){
    grid_info res;

    if(w_size >= g_size){
        res.kernel_size = 1;
        
        res.step = w_size / g_size;
        res.left_padding = (w_size - (res.step * g_size)) / 2;
        res.right_padding = (w_size - (res.step * g_size)) - res.left_padding;
        //printf("step %d, left %d, right %d\n",res.step,res.left_padding,res.right_padding);
        
    }
    else{
        shrinked = true;

        if(g_size % w_size != 0)throwException("if the grid size is bigger than window size it should have the size of (integer * window_size)");
        res.kernel_size = g_size / w_size;
        res.step = 1;
        //printf("kernel size %d\n",res.kernel_size);
    }
    return res;
}

visualize::visualize(){ shrinked = false; }

bool visualize::is_shrinked(){
    return shrinked;
}

void visualize::make_window(uint width,uint height){
    window_object = sf::RenderWindow(sf::VideoMode({width, height}), "ok computer");
    window_width = width;
    window_height = height;
}

void visualize::make_grid(uint g_width, uint g_height){
    grid_x = calculate_grid(window_width, g_width, shrinked);
    grid_y = calculate_grid(window_height, g_height, shrinked);
    grid_width = g_width;
    grid_height = g_height;
}

void visualize::draw_field(const vec& density){
    sf::RectangleShape cell({(float)grid_x.step, (float)grid_y.step});
    uint shade;
    for(int x = 0; x < grid_width; x++){
        for(int y = 0; y < grid_height; y++){
            shade = density[y * grid_width + x] * 255;
            cell.setPosition({(float)(grid_x.left_padding + x * grid_x.step),(float)(grid_y.left_padding + y * grid_y.step)});
            cell.setFillColor(sf::Color(shade, shade, shade));
            window_object.draw(cell);
        }
    }
    window_object.display();
}

void visualize::draw_shrinked_field(const vec& density){
    sf::RectangleShape cell({(float)grid_x.step, (float)grid_y.step});
    double shade;
    uint gx = 0, gy = 0;
    for(int x = 0; x < grid_width/grid_x.kernel_size; x++){
        for(int y = 0; y < grid_height/grid_y.kernel_size; y++){
            shade = 0;
            for(int kx = 0; kx < grid_x.kernel_size; kx++){
                for(int ky = 0; ky < grid_y.kernel_size; ky++){
                    shade += density[(gy + ky) * grid_width + (gx + kx)];
                }
            }
            gx += grid_x.kernel_size;
            if(gx == grid_width){
                gx = 0;
                gy += grid_y.kernel_size;
            }

            shade /= grid_x.kernel_size * grid_y.kernel_size;
            //std::cout << shade << " ";
            shade *= 255;

            cell.setPosition({(float)(grid_x.left_padding + x * grid_x.step),(float)(grid_y.left_padding + y * grid_y.step)});
            cell.setFillColor(sf::Color(shade, shade, shade));
            window_object.draw(cell); 
        }
        //std::cout<< "\n";
    }
    window_object.display();
}

void visualize::close(){
    window_object.close();
}



visual_facade::visual_facade(uint window_width, uint window_height,uint grid_width,uint grid_height){
    object.make_window(window_width, window_height);
    object.make_grid(grid_width, grid_height);
}

void visual_facade::show(const vec& density){
    if(object.is_shrinked()){
        object.draw_shrinked_field(density);
    }
    else{
        object.draw_field(density);
    }
}

void visual_facade::close(){
    object.close();
}



