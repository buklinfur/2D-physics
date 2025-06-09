#ifndef CUDAGrid
#define CUDAGrid

#pragma once
#include <cuda_runtime.h>
#include <vector>

enum class FlowDirection { LEFT_TO_RIGHT, RIGHT_TO_LEFT, TOP_TO_DOWN, DOWN_TO_TOP };

/**
 * @brief Compute equilibrium distribution function for a D2Q9 model.
 *
 * @param density Fluid density at a node.
 * @param weight Lattice weight for the direction.
 * @param e1 x-component of discrete velocity direction.
 * @param e2 y-component of discrete velocity direction.
 * @param vel1 x-component of macroscopic velocity.
 * @param vel2 y-component of macroscopic velocity.
 * @return Equilibrium distribution function value for the given direction.
 */
__device__ float equilibrium(float density, float weight, float e1, float e2, float vel1, float vel2);

/**
 * @brief Compute fluid density and velocity at each node.
 *
 * @param f_in Input distribution functions.
 * @param vel Output buffer for velocities (2 components per node).
 * @param density Output buffer for scalar densities.
 * @param e Discrete velocity vectors (D2Q9 model, 18 elements).
 */
__global__ void cuda_density_velocity(float* f_in, float* vel, float* density, float* e);

/**
 * @brief Apply inflow boundary condition using specified direction.
 *
 * @param vel Velocity field (to update).
 * @param f_in Input distribution functions (to modify).
 * @param density Local density buffer.
 * @param weights Lattice weights.
 * @param e Discrete velocity directions.
 * @param flow_dir Direction of flow (0:left, 1:right, 2:bottom, 3:top).
 */
__global__ void cuda_inflow(float* vel, float* f_in, float* density, float* weights, float* e);

/**
 * @brief Apply zero-gradient outflow boundary condition.
 *
 * @param f_in Distribution functions (to modify at boundary).
 * @param flow_dir Direction of flow (0:left, 1:right, 2:bottom, 3:top).
 */
__global__ void cuda_outflow(float* f_in);

/**
 * @brief Perform BGK collision step to relax f_in toward equilibrium.
 *
 * @param f_in Input distribution functions.
 * @param f_out Output distribution functions after collision.
 * @param density Local density.
 * @param vel Local velocity.
 * @param weights Lattice weights.
 * @param e Discrete velocity directions.
 */
__global__ void cuda_collision(float* f_in, float* f_out, float* density, float* vel, float* weights, float* e);

/**
 * @brief Propagate distribution functions along discrete velocity directions.
 *
 * @param f_in Updated input distribution functions.
 * @param f_out Post-collision distribution functions.
 * @param e Discrete velocity vectors.
 * @param flow_dir Flow direction used to wrap around (periodic).
 */
__global__ void cuda_streaming(float* f_in, float* f_out, float* e);

/**
 * @brief Convert velocity magnitude to grayscale for visualization.
 *
 * @param vel Velocity buffer.
 * @param visual_buffer Output buffer (unsigned char) for grayscale visualization.
 * @param max Maximum velocity magnitude used for normalization.
 */
__global__ void convert_visual(float *vel, unsigned char *visual_buffer, float max);

/**
 * @brief Apply bounce-back boundary condition for solid obstacles.
 *
 * @param f_in Input distribution functions.
 * @param f_out Output buffer after reflection.
 * @param obstacle_mask Boolean mask indicating solid nodes.
 */
__global__ void bounce_back(float* f_in, float* f_out, float *obstacle, int obstacle_size);

/**
 * @brief Initialize device-side simulation constants.
 *
 * @param width Simulation width.
 * @param height Simulation height.
 * @param r Characteristic radius.
 * @param uLB Lattice velocity.
 * @param Re Reynolds number.
 * @param nulb Kinematic viscosity.
 * @param omega Relaxation parameter.
 */
__global__ void set_device_constants(size_t width, size_t height, float r, float uLB, float Re, float nulb, float omega);

class Field {
public:
    /**
     * @brief Construct a Field object managing LBM buffers and parameters.
     * Allocates GPU memory and calculates physical parameters for the LBM simulation.
     *
     * @param width Width of the simulation grid.
     * @param height Height of the simulation grid.
     * @param block_size Size of CUDA blocks (e.g., 16 for 16x16 threads).
     * @param uLB Characteristic lattice velocity.
     * @param Re Reynolds number.
     * @param flow_dir Direction of inflow (left, right, top, bottom).
     * @param obstacle_mask Host-side boolean mask for solid cells (used for bounce-back).
     */
    Field(size_t width, size_t height, int block_size, float uLB, float Re, 
          FlowDirection flow_dir, bool* obstacle_mask);

    /**
     * @brief Destructor for the Field object.
     *
     * Frees all allocated GPU memory associated with distribution functions, velocities,
     * densities, weights, and visualization buffers.
     */
    ~Field();

    /**
     * @brief Perform one LBM step: collision followed by streaming.
     *
     * This method invokes device kernels to:
     * - Compute macroscopic density and velocity
     * - Apply inflow and outflow boundary conditions
     * - Perform BGK collision step
     * - Stream distribution functions to neighbor nodes
     * - Apply bounce-back conditions for solid obstacles
     */
    void step();

    /**
     * @brief Launch a kernel to generate a velocity-magnitude grayscale image.
     *
     * Converts the velocity magnitude at each node to a grayscale value and stores
     * the result in a host-accessible visualization buffer.
     */
    unsigned char* get_visual(float max);

    /**
     * @brief Upload obstacle mask from host to device.
     *
     * This method transfers a boolean array indicating solid (bounce-back) cells
     * to device memory for use in bounce-back boundary condition kernels.
     *
     * @param obstacle_mask Host-side boolean array (true for solid cells).
     */
    void update_obstacle_mask(bool* obstacle_mask);

private:
    size_t width, height;
    dim3 blockSize;
    dim3 gridSize;
    float *f_in, *f_out, *vel, *density, *e, *weights;
    bool* obstacle_mask;
    unsigned char *visual_buffer;
    FlowDirection flow_dir;
};

#endif