
#include "visualize.hpp"
#include <conio.h>



int main()
{
    vec field{
              0.5,0.0,0.0,0.0,0.5,
              0.6,0.0,0.0,0.0,0.6,
              0.7,0.7,0.7,0.0,0.7,
              0.8,0.0,0.8,0.0,0.8,
              0.9,0.9,0.9,0.0,0.9
            };
    visual_facade riba(500,500,5,5);
    riba.show(field);
    getch();
    riba.close();
}
