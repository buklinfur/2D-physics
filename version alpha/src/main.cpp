
#include "visualize.hpp"
#include <conio.h>


#include <vector>

using namespace std;

typedef vector<double> vec;
typedef vector<vector<double>> matrix;
typedef vector<vector<vector<double>>> f_matrix;

typedef struct {
	int width, height;
	double sound_speed;
	double tao;
	matrix e;       // (9,2)
	vec weights;    // (9)
	f_matrix f_in;  // (width,height,9)
	f_matrix f_out; // (width,height,9)
	f_matrix speed; // (width,height,2)
	matrix density; // (width,height)
} field_info;

class field_class {
private:
	field_info field;
	void(*custom_function)(field_info&);

public:
	field_class(const matrix& e_val, const vec& weights_val, const double& sound_speed_val, const double& tao_val, const size_t& width_val, const size_t& height_val) {
		field.e.resize(9);
		for (int i = 0; i < 9; i++) {
			field.e[i].resize(2);
			field.e[i][0] = e_val[i][0];
			field.e[i][1] = e_val[i][1];
		}

		field.weights.resize(9);
		for (int i = 0; i < 9; i++) field.weights[i] = weights_val[i];

		field.sound_speed = sound_speed_val;
		field.tao = tao_val;
		field.width = width_val;
		field.height = height_val;

		field.f_in.resize(field.width);
		for (int i = 0; i < field.width; i++) {
			field.f_in[i].resize(field.height);
			for (int j = 0; j < field.height; j++) {
				field.f_in[i][j].resize(9);
				for (int k = 0; k < 9; k++) field.f_in[i][j][k] = 0;
			}
		}

		field.f_out.resize(field.width);
		for (int i = 0; i < field.width; i++) {
			field.f_out[i].resize(field.height);
			for (int j = 0; j < field.height; j++) {
				field.f_out[i][j].resize(9);
				for (int k = 0; k < 9; k++) field.f_out[i][j][k] = 0;
			}
		}

		field.speed.resize(field.width);
		for (int i = 0; i < field.width; i++) {
			field.speed[i].resize(field.height);
			for (int j = 0; j < field.height; j++) { field.speed[i][j].resize(2); field.speed[i][j][0] = 0; field.speed[i][j][1] = 0; }
		}

		field.density.resize(field.width);
		for (int i = 0; i < field.width; i++) {
			field.density[i].resize(field.height);
			for (int j = 0; j < field.height; j++) field.density[i][j] = 0;
		}


		custom_function = NULL;
	}

	void add_custom_rule(void(*function)(field_info&)) {
		custom_function = function;
	}

	void sim_step() {

		//считаем плотность и скорость для клеток
		for (int i = 0; i < field.width; i++) {
			for (int j = 0; j < field.height; j++) {
				field.density[i][j] = 0.0001;
				field.speed[i][j] = { 0, 0 };
				for (int k = 0; k < 9; k++) {
					field.density[i][j] += field.f_in[i][j][k];
					field.speed[i][j][0] += field.f_in[i][j][k] * field.e[k][0];
					field.speed[i][j][1] += field.f_in[i][j][k] * field.e[k][1];
				}
				field.speed[i][j][0] /= field.density[i][j];
				field.speed[i][j][1] /= field.density[i][j];
			}
		}
		//обновляем f 
		double f_eq;
		for (int i = 0; i < field.width; i++) {
			for (int j = 0; j < field.height; j++) {
				for (int k = 0; k < 9; k++) {
					f_eq = field.weights[k] * field.density[i][j] * (1 + (field.e[k][0] * field.speed[i][j][0] + field.e[k][1] * field.speed[i][j][1]) / 3 +
						(field.e[k][0] * field.speed[i][j][0] + field.e[k][1] * field.speed[i][j][1]) * (field.e[k][0] * field.speed[i][j][0] + field.e[k][1] * field.speed[i][j][1]) * 2.0/9
						- (field.speed[i][j][0] * field.speed[i][j][0] + field.speed[i][j][1] * field.speed[i][j][1])*2.0/3);
					if (k == 8) { field.f_out[i][j][8] = field.f_in[i][j][k] - (field.f_in[i][j][k] - f_eq) / field.tao; continue; }
					if (i + field.e[k][0] >= 0 && j + field.e[k][1] >= 0 && i + field.e[k][0] < field.width && j + field.e[k][1] < field.height) {
						field.f_out[i + field.e[k][0]][j + field.e[k][1]][(k + 4) % 8] = field.f_in[i][j][k] - (field.f_in[i][j][k] - f_eq) / field.tao;
					}
					else {
						//bounce-back
						field.f_out[i][j][k] = field.f_in[i][j][(k + 4) % 8];
					}
				}
			}
		}
		if (custom_function != NULL) custom_function(field);
		field.f_in.swap(field.f_out);
	}

	vec get_field() {
		
		
		vec visual(field.width*field.height);
		for (int j = 0; j < field.height; j++) {
			for (int i = 0; i < field.width; i++) {
				//visual[j * field.width + i] = field.density[i][j];
				visual[j * field.width + i] = field.speed[i][j][0] * field.speed[i][j][0] + field.speed[i][j][1] * field.speed[i][j][1];
				//if (visual[i * field.height + j] > 0)cout << i << " " << j << " " << visual[i * field.height + j] << "\n";
			}
		}
		return visual;
	}

	void set_density_field(vec new_density) {
		for (int i = 0; i < field.width; i++) {
			for (int j = 0; j < field.height; j++) {
				field.density[i][j] = new_density[i + j * field.width];
			}
		}
	}

};

#include <omp.h>
#include <iostream>
vec viprimlator(const matrix& density) {
	vec res(density.size() * density[0].size());
	for (int i = 0; i < density.size();i++) {
		for (int j = 0; j < density[i].size(); j++) {
			res[i * density[0].size() + j] = density[i][j];
		}
	}
	return res;
}

void stream_function(field_info& field) {
	cout << field.height << "\n";
	for (int j = 0; j < field.height; j++) {
		field.speed[0][j] = { 0.04 *  (1 + 1e-4 * sin((j*1.0) / field.height * 2 * 3.14)),0 };
		field.speed[field.width-1][j] = { 0,0 };
		field.density[field.width - 1][j] = 0.0001;
	}
	
}
#include <thread>
#include <chrono>
int main() {
	matrix e = { {0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,0} };
	vec weights = { 4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 };
	double sound_speed = 1 / sqrt(3);
	field_class riba(e, weights, sound_speed, 0.6, 100, 100);
	riba.add_custom_rule(stream_function);
	vec speed_matrix;
	vec field{
			  0.5,0.0,0.0,0.0,0.5,
			  0.6,0.0,0.0,0.0,0.6,
			  0.7,0.7,0.7,0.0,0.7,
			  0.8,0.0,0.8,0.0,0.8,
			  0.9,0.9,0.9,0.0,0.9
	};
	//riba.set_density_field(field);
	visual_facade renderer(500,500,100,100);
	for (int i = 0; i < 100;) {
		riba.sim_step();
		speed_matrix = riba.get_field();
		renderer.show(speed_matrix);
		//getch();
		//std::this_thread::sleep_for(std::chrono::nanoseconds(500000));
		cout << "step\n";
	}
}