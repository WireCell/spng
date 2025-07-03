/**
 * TorchRayGrid.h
 *
 * This file provides the libtorch-based equivalent of RayGrid.h.
 * It defines the geometry of the tomographic views (layers) using
 * torch::Tensor for all numerical data. This enables efficient, batched
 * computations suitable for both CPU and GPU execution.
 *
 * Major changes from the original:
 *
 * - The data model is fully 2D and there is no concept of a 3rd "drift"
 *   direction.
 *
 * - All std::vector, boost::multi_array, and custom Vector classes are replaced
 *   with torch::Tensor.
 * 
 * - Calculations are designed to be performed in batches, eliminating slow,
 *   iterative loops.
 *
 * - The main Coordinates class provides a convention .to(device) method to move
 *   its data to a given device.
 */

#ifndef WIRECELL_TORCHRAYGRID_H
#define WIRECELL_TORCHRAYGRID_H

#include <torch/torch.h>
#include <vector>
#include <utility>
#include <cmath>                // for pi

namespace WireCell {
namespace Spng {
namespace TorchRayGrid {

    /** Ray grid taxonomy
     *
     * As this code uses torch tensors for all data there are no special C++
     * types to describe various taxons as in original RayGrid.  They are
     * described here:
     *
     * ray - a segment of a conceptually infinite line.  When ray grid describes
     * LArTPC wires/strips, a ray is the lower bounds of the region centered on
     * the electrode.
     *
     * coordinate - a pair of indices giving the "layer" and the (1D) "grid"
     * location of a ray.
     *
     * crossing - a pair of ray grid coordinates specifying rays in different
     * views.
     *
     * crossing point - a point in the mother 3D Cartesian space at a crossing.
     */

    class Coordinates {
    public:
        // Constructor
        Coordinates(const torch::Tensor& views);

        /// Return ray crossings.  This comes in three flavors: scalar, hybrid
        /// scalar/batched and full batched.
        torch::Tensor ray_crossing(int64_t view1, int64_t ray1,
                                   int64_t view2, int64_t ray2) const;
        torch::Tensor ray_crossing(int64_t view1, const torch::Tensor& ray1,
                                   int64_t view2, const torch::Tensor& ray2) const;
        torch::Tensor ray_crossing(const torch::Tensor& view1, 
                                   const torch::Tensor& ray1,
                                   const torch::Tensor& view2,
                                   const torch::Tensor& ray2) const;
        
        /// Return pitch locations.  This comes in three flavors: scalar, hybrid
        /// scalar/batched and full batched.
        torch::Tensor pitch_location(int64_t view1, int64_t ray1,
                                     int64_t view2, int64_t ray2,
                                     int64_t view_idx) const;
        torch::Tensor pitch_location(int64_t view1, const torch::Tensor& ray1,
                                     int64_t view2, const torch::Tensor& ray2,
                                     int64_t view_idx) const;
        torch::Tensor pitch_location(const torch::Tensor& view1, const torch::Tensor& ray1,
                                     const torch::Tensor& view2, const torch::Tensor& ray2,
                                     const torch::Tensor& view_idx) const;
        
        
        // Public tensors (equivalent to Python attributes)
        torch::Tensor pitch_mag;
        torch::Tensor pitch_dir;
        torch::Tensor center;
        torch::Tensor zero_crossings;
        torch::Tensor ray_jump;
        torch::Tensor a;
        torch::Tensor b;
        
    private:
        void init(const torch::Tensor& views);
        int64_t nviews; // Store nviews as a member
    };


    /**
       Return a tensor with ray pairs defining a number of views.

       If bounds is true (default is false) the first two layers give a
       rectangular horizontal and vertical bounds.

       The last three layers consist of first two layers symmetric with the last
       and at some angle.

       Strictly speaking, ray grid is not defined if any views are parallel.
       The default value of bounds=true indeed produces a parallel pair of
       views.  A Coordinates object can still be constructed on such ill-formed
       views.  Any crossings between these views are undefined.
     */
    torch::Tensor symmetric_views(double width=100, double height=100, double pitch_mag=3,
                                  double angle=60*M_PI/180.0, bool bounds = true);

    namespace funcs {
    /**
     * @brief Return the perpendicular vector giving the separation between two
     * parallel rays.
     * @param r0 A tensor representing the first ray (2 endpoints, 2 coordinates).
     * @param r1 A tensor representing the second ray (2 endpoints, 2 coordinates).
     * @return A 2D tensor representing the perpendicular vector.
     */
    torch::Tensor pitch(const torch::Tensor& r0, const torch::Tensor& r1);

    /**
     * @brief Return the vector along ray direction.
     * @param ray A tensor representing the a ray (2 endpoints, 2 coordinates).
     * @return A 2D tensor representing the vector along the ray.
     */
    torch::Tensor vector(const torch::Tensor& ray);

    /**
     * @brief Return the unit vector along ray direction.
     * @param ray A tensor representing the a ray (2 endpoints, 2 coordinates).
     * @return A 2D tensor representing the unit vector along the ray.
     */
    torch::Tensor direction(const torch::Tensor& ray);

    /**
     * @brief Return point where two non-parallel rays cross.
     * @param r0 A tensor representing the first ray (2 endpoints, 2 coordinates).
     * @param r1 A tensor representing the second ray (2 endpoints, 2 coordinates).
     * @return A 2D tensor representing the intersection point.
     * @throws std::runtime_error if lines are parallel.
     */
    torch::Tensor crossing(const torch::Tensor& r0, const torch::Tensor& r1);
    }

} // namespace TorchRayGrid
} // namespace Spng
} // namespace WireCell

#endif // WIRECELL_TORCHRAYGRID_H
