класс visual_facade

visual_facade(uint window_width, uint window_height,uint grid_width,uint grid_height)
конструктор класса создаёт окно window_width на window_height пикселей и сетку grid_width на grid_height элементов
если grid_width <= window_width и grid_height <= window_height, то размеры сетки могут быть любыми целыми числами больше 0.
иначе, размеры сетки должны быть пропорциональны размерам окна т.е. делиться без остатка

void show(const vec& density)
принимает выпрямленную матрицу размером grid_width*grid_height
выводит в окно значения сетки

void close()
закрывает окно