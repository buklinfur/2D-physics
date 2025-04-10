
#include "visualize.hpp"
#include <conio.h>


#include <vector>

using namespace std;

typedef vector<double> vec;
typedef vector<vector<double>> matrix;
typedef vector<vector<vector<double>>> f_matrix;

typedef struct {
	int width, height;
	double tao;
	double omega;
	double uLB;
	vector<vector<int>> e;       // (9,2)
	vec weights;    // (9)
	f_matrix f_in;  // (9,width,height)
	f_matrix f_out; // (9,width,height)
	f_matrix vel; // (width,height,2)
	matrix density; // (width,height)
} field_info;

class field_class {
private:
	field_info field;
	void(*custom_function)(field_info&);
	vector<vector<int>> obstacle; // (points,2)
	double equilibrium(double density,double weight,vector<int> e,vec velocity){
		double usqr = 3.0/2 * (velocity[0]*velocity[0] + velocity[1]*velocity[1]);
		double cu = 3 * (e[0]*velocity[0] + e[1]*velocity[1]);
		double f_eq = density * weight * (1 + cu + 0.5*cu*cu - usqr);
		return f_eq;
	}
public:
	field_class(const vector<vector<int>>& e_val, const vec& weights_val, const size_t& width_val, const size_t& height_val,const double& omega_val, const double& uLB_val) {
		std::cout << "e\n";
		field.e.resize(9);
		for (int i = 0; i < 9; i++) {
			field.e[i].resize(2);
			field.e[i][0] = e_val[i][0];
			field.e[i][1] = e_val[i][1];
		}
		std::cout << "w\n";
		field.weights.resize(9);
		for (int i = 0; i < 9; i++) field.weights[i] = weights_val[i];

		field.width = width_val;
		field.height = height_val;
		field.omega = omega_val;
		field.uLB = uLB_val;
		std::cout << "vel\n";
		field.vel.resize(field.width);
		for (int i = 0; i < field.width; i++) {
			field.vel[i].resize(field.height);
			for (int j = 0; j < field.height; j++) { 
				field.vel[i][j].resize(2);
				field.vel[i][j][0] = (1-2) * field.uLB * (1 + 1e-4*sin(i/(field.width-1)*0.5))/10; 
				field.vel[i][j][1] = (1-2) * field.uLB * (1 + 1e-4*sin(j/(field.height-1)*0.5))/10;
			 }
		}
		std::cout << "fin\n";
		field.f_in.resize(9);
		for (int i = 0; i < 9; i++) {
			field.f_in[i].resize(field.width);
			for (int w = 0; w < field.width; w++) {
				field.f_in[i][w].resize(field.height);
				for (int h = 0; h < field.height; h++){
					double usqr = 3.0/2 * (field.vel[w][h][0]*field.vel[w][h][0] + field.vel[w][h][1]*field.vel[w][h][1]);
					double cu = 3 * (field.e[i][0]*field.vel[w][h][0] + field.e[i][1]*field.vel[w][h][1]);
					field.f_in[i][w][h] = 1 * field.weights[i] * (1 + cu + 0.5*cu*cu - usqr);
				}
			}
		}
		std::cout << "fout\n";
		field.f_out.resize(9);
		for (int i = 0; i < 9; i++) {
			field.f_out[i].resize(field.width);
			for (int j = 0; j < field.width; j++) {
				field.f_out[i][j].resize(field.height);
				for (int k = 0; k < field.height; k++) field.f_out[i][j][k] = 0;
			}
		}
		
		std::cout << "dens\n";
		field.density.resize(field.width);
		for (int i = 0; i < field.width; i++) {
			field.density[i].resize(field.height);
			for (int j = 0; j < field.height; j++) field.density[i][j] = 0.0001;
		}

		int r = 10;
		std::cout << "obstacle\n";
		for(int w = 0; w < field.width; w++){
			for(int h = 0; h < field.height; h++){
				if((w - field.width/3)*(w - field.width/3) + (h-field.height/2)*(h-field.height/2) < r*r){
					obstacle.push_back({w, h});
				}
			}
		}

		custom_function = NULL;
	}

	void add_custom_rule(void(*function)(field_info&)) {
		custom_function = function;
	}

	void sim_step() {

		// Right wall: outflow condition
		for(int y = 1; y < field.height-1; y++){
			field.f_in[6][field.width-1][y] = field.f_in[6][field.width-2][y];
			field.f_in[7][field.width-1][y] = field.f_in[7][field.width-2][y];
			field.f_in[8][field.width-1][y] = field.f_in[8][field.width-2][y];
		}
		
		/*
		//Up and down outflow
		for(int x = 0; x < field.width; x++){
			field.f_in[6][field.height-1][x] = field.f_in[6][field.height-2][x];
			field.f_in[3][field.height-1][x] = field.f_in[3][field.height-2][x];
			field.f_in[0][field.height-1][x] = field.f_in[0][field.height-2][x];
		}
		for(int x = 0; x < field.width; x++){
			field.f_in[8][0][x] = field.f_in[8][1][x];
			field.f_in[5][0][x] = field.f_in[5][1][x];
			field.f_in[2][0][x] = field.f_in[2][1][x];
		}
			*/
		// Compute density and velocity
		for(int i = 0; i < field.width; i++){
			for(int j = 0; j < field.height; j++){
				field.density[i][j] = 0;
				field.vel[i][j][0] = 0;
				field.vel[i][j][1] = 0;
				for(int k = 0; k < 9; k++) {
					field.density[i][j] += field.f_in[k][i][j];
					field.vel[i][j][0] += field.e[k][0] * field.f_in[k][i][j];
					field.vel[i][j][1] += field.e[k][1] * field.f_in[k][i][j];
				}
				field.vel[i][j][0] /= field.density[i][j];
				field.vel[i][j][1] /= field.density[i][j];
			}
		}
		// Left wall: inflow condition
		for(int y = 0; y < field.height; y++){
			field.vel[0][y][0] = (1-2) * field.uLB * (1 + 1e-4*sin(y/(field.height-1)*2*3.14));
			//field.vel[0][y][1] = (1-2) * field.uLB * (1 + 1e-4*sin(y/(field.height-1)*2*3.14));
			//field.vel[0][y][0] = 0.02 + (y%10)*0.002;
			field.vel[0][y][1] = (1-2) * field.uLB * (1 + 1e-4*sin(y/(field.height-1)*2*3.14));
			double sum1 = field.f_in[3][0][y] + field.f_in[4][0][y] + field.f_in[5][0][y];
			double sum2 = field.f_in[6][0][y] + field.f_in[7][0][y] + field.f_in[8][0][y];
			field.density[0][y] = 1/(1-field.vel[0][y][0]) * (sum1 + 2 * sum2);
		}
		// Compute equilibrium + collision
		for(int h = 0; h < field.height; h++){
			for(int i = 0; i < 3; i++){
				field.f_in[i][0][h] = equilibrium(field.density[0][h],field.weights[i],field.e[i],field.vel[0][h]) + 
									  field.f_in[8-i][0][h] - equilibrium(field.density[0][h],field.weights[8-i],field.e[8-i],field.vel[0][h]);
			}
		}
		for(int w = 0; w < field.width; w++){
			for(int h = 0; h < field.height;h++){
				for(int i = 0; i < 9; i++){
					double usqr = 3.0/2 * (field.vel[w][h][0]*field.vel[w][h][0] + field.vel[w][h][1]*field.vel[w][h][1]);
					double cu = 3 * (field.e[i][0]*field.vel[w][h][0] + field.e[i][1]*field.vel[w][h][1]);
					double f_eq = field.density[w][h] * field.weights[i] * (1 + cu + 0.5*cu*cu - usqr);
					field.f_out[i][w][h] = field.f_in[i][w][h] - field.omega * (field.f_in[i][w][h] - f_eq);
				}
			}
		}
		//bounce-back
		for(auto& cords : obstacle){
			for(int i = 0; i < 9; i++){
				field.f_out[i][cords[0]][cords[1]] = field.f_in[8-i][cords[0]][cords[1]];
			}
		}
		// Streaming step
		for(int i = 0; i < 9; i++){
			for(int w = 0; w < field.width; w++){
				for(int h = 0; h < field.height; h++){
					int index1 = w + field.e[i][0];
					int index2 = h + field.e[i][1];
					if(index1 == field.width)index1 = 0;
					if(index1 == -1)index1 = field.width - 1;
					if(index2 == field.height)index2 = 0;
					if(index2 == -1)index2 = field.height - 1;
					field.f_in[i][index1][index2] = field.f_out[i][w][h]; 
				}
			}
		}
	}

	vec get_field() {
		vec visual(field.width*field.height);
		double max = 1e-8;
		double min = 1000;
		for (int j = 0; j < field.height; j++) {
			for (int i = 0; i < field.width; i++) {
				//visual[j * field.width + i] = field.density[i][j];
				
				visual[j * field.width + i] = sqrt(field.vel[i][j][0] * field.vel[i][j][0] + field.vel[i][j][1] * field.vel[i][j][1]) * 10;
				if(visual[j * field.width + i] > max) max = visual[j * field.width + i];
				if(visual[j * field.width + i] < min) min = visual[j * field.width + i];
				//if (visual[i * field.height + j] > 0)cout << i << " " << j << " " << visual[i * field.height + j] << "\n";
				//cout << i << " " << j << " " << visual[j * field.width + i] << "\n";
			}
		}
		for(int i = 0; i < field.width*field.height; i++){
			//visual[i] = (visual[i] - min) / (max-min);
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
		field.vel[0][j] = { 0.04 *  (1 + 1e-4 * sin((j*1.0) / field.height * 2 * 3.14)),0 };
		field.vel[field.width-1][j] = { 0,0 };
		field.density[field.width - 1][j] = 0.0001;
	}
	
}
#include <thread>
#include <chrono>
int main() {
	vector<vector<int>> e = { {1,1},{1,0},{1,-1},{0,1},{0,0},{0,-1},{-1,1},{-1,0},{-1,-1} };
	vec weights = {1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36};
	int width = 80;
	int height = 80;
	double uLB = 0.04;
	int r = (height-1)/9;
	double Re = 10;
	double nulb = uLB*r/Re;
	double omega = 1 / (3*nulb+0.5);
	field_class riba(e, weights, width, height, omega, uLB);
	vec speed_matrix;
	//riba.set_density_field(field);
	visual_facade renderer(500,500,width,height);
	for (int i = 0; i < 1000;i++) {
		riba.sim_step();
		speed_matrix = riba.get_field();
		renderer.show(speed_matrix);
		//getch();
		//std::this_thread::sleep_for(std::chrono::nanoseconds(500000));
		cout << "step\n";
	}
	getch();
}