/**
 * TorchRayGrid.h
 *
 * This file provides the libtorch-based equivalent of RayGrid.h.
 * It defines the geometry of the tomographic views (layers) using
 * torch::Tensor for all numerical data. This enables efficient, batched
 * computations suitable for both CPU and GPU execution.
 *
 * Major changes from the original:
 * - All std::vector, boost::multi_array, and custom Vector classes are replaced
 * with torch::Tensor.
 * - Calculations are designed to be performed in batches, eliminating slow,
 * iterative loops.
 * - The main Coordinates class is initialized with a torch::Device to specify
 * whether computations should occur on the CPU or a CUDA-enabled GPU.
 */

#ifndef WIRECELL_TORCHRAYGRID_H
#define WIRECELL_TORCHRAYGRID_H

#include <torch/torch.h>
#include <vector>
#include <utility>

namespace WireCell {
namespace TorchRayGrid {

    // A ray is identified by its layer and grid indices.
    // This remains a simple struct for clarity.
    struct coordinate_t {
        int layer;
        int grid;
    };

    // A crossing is identified by the coordinates of two intersecting rays.
    typedef std::pair<coordinate_t, coordinate_t> crossing_t;
    typedef std::vector<crossing_t> crossings_t;

    /**
     * @class Coordinates
     * @brief Manages the geometric relationships between layers in the tomographic setup.
     *
     * This class pre-computes various geometric quantities and stores them as tensors.
     * These tensors allow for extremely fast, batched calculation of ray crossings
     * and pitch locations, which are fundamental operations for the reconstruction.
     */
    class Coordinates {
    public:
        /**
         * @brief Constructs a Coordinates object.
         * @param rays A tensor defining the seed rays for each layer.
         * Shape: [nlayers, 2 (rays per pair), 2 (points per ray), 2 (x,y coords)].
         * It's assumed the rays have been projected onto a 2D plane.
         * @param device The torch::Device (e.g., torch::kCPU or torch::kCUDA) on which
         * to perform computations.
         */
        Coordinates(const torch::Tensor& rays, const torch::Device& device);

        /**
         * @brief Calculates the crossing point of a batch of ray pairs.
         * @param crossings A Long tensor identifying the pairs of rays.
         * Shape: [N, 2 (rays per crossing), 2 (layer, grid)].
         * @return A tensor of crossing point coordinates. Shape: [N, 2 (x,y)].
         */
        torch::Tensor ray_crossing_batch(const torch::Tensor& crossings) const;

        /**
         * @brief Calculates the location of ray crossings projected onto a third layer's pitch axis.
         * @param crossings A Long tensor identifying the pairs of rays forming the crossings.
         * Shape: [N, 2, 2], with the last dimension being (layer, grid).
         * @param other_layers A Long tensor of the layer indices onto which to project.
         * Shape: [N].
         * @return A tensor of pitch locations. Shape: [N].
         */
        torch::Tensor pitch_location_batch(const torch::Tensor& crossings, const torch::Tensor& other_layers) const;

        // Accessors for geometric tensors
        int nlayers() const { return m_nlayers; }
        const torch::Device& device() const { return m_device; }
        const torch::Tensor& pitch_mags() const { return m_pitch_mag; }
        const torch::Tensor& pitch_dirs() const { return m_pitch_dir; }
        const torch::Tensor& centers() const { return m_center; }
        const torch::Tensor& ray_jumps() const { return m_ray_jump; }
        const torch::Tensor& a() const { return m_a; }
        const torch::Tensor& b() const { return m_b; }

    private:
        int m_nlayers;
        torch::Device m_device;

        // Geometric quantities stored as tensors for batched operations.
        // All tensors reside on m_device.
        torch::Tensor m_pitch_mag;      // Shape: [nlayers]
        torch::Tensor m_pitch_dir;      // Shape: [nlayers, 2]
        torch::Tensor m_center;         // Shape: [nlayers, 2]
        torch::Tensor m_zero_crossing;  // Shape: [nlayers, nlayers, 2]
        torch::Tensor m_ray_jump;       // Shape: [nlayers, nlayers, 2]
        torch::Tensor m_a, m_b;         // Shape: [nlayers, nlayers, nlayers]
    };

} // namespace TorchRayGrid
} // namespace WireCell

#endif // WIRECELL_TORCHRAYGRID_H
