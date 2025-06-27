/**
 * TorchRayTiling.h
 *
 * This file provides the libtorch-based equivalent of RayTiling.h.
 * It defines the classes and functions for performing the tiling algorithm,
 * which finds regions of interest (Blobs) from 1D activity measurements.
 */

#ifndef WIRECELL_TORCHRAYTILING_H
#define WIRECELL_TORCHRAYTILING_H

#include "WireCellSpng/TorchRayGrid.h"
#include <string>
#include <set>

namespace WireCell {
namespace TorchRayGrid {

    // Represents a contiguous set of active rays in a single layer.
    // The structure remains simple, as it represents metadata.
    struct Strip {
        int layer;
        std::pair<int, int> bounds; // [min_grid_idx, max_grid_idx)

        size_t width() const {
            return bounds.second - bounds.first;
        }

        // Returns the two bounding rays of the strip as a crossing_t.
        // Note: these rays are parallel and do not cross.
        crossing_t addresses() const {
            return std::make_pair(coordinate_t{layer, bounds.first}, coordinate_t{layer, bounds.second});
        }
    };
    typedef std::vector<Strip> strips_t;

    /**
     * @class Activity
     * @brief Represents the 1D measurement data for a single layer.
     * This class wraps a 1D torch::Tensor.
     */
    class Activity {
    public:
        /**
         * @brief Constructs an Activity object.
         * @param layer The layer index.
         * @param data A 1D tensor of activity values.
         * @param offset The grid index corresponding to the first element of the data tensor.
         * @param threshold The value above which an activity is considered significant.
         */
        Activity(int layer, const torch::Tensor& data, int offset = 0, double threshold = 0.0);

        // Creates strips for all contiguous regions of activity above the threshold.
        strips_t make_strips() const;

        int layer() const { return m_layer; }
        int offset() const { return m_offset; }
        const torch::Tensor& span() const { return m_span; }
        bool empty() const { return m_span.size(0) == 0; }
        torch::Device device() const { return m_span.device(); }

    private:
        int m_layer;
        int m_offset;
        double m_threshold;
        torch::Tensor m_span; // 1D Tensor of activity values
    };
    typedef std::vector<Activity> activities_t;

    /**
     * @class Blob
     * @brief Represents a polygonal region in 2D space consistent with a set of Strips.
     */
    class Blob {
    public:
        // Adds a strip to the blob, updating the set of corners that define its boundary.
        // This is where the core intersection logic resides.
        void add(const Coordinates& coords, const Strip& strip, double nudge = 0.0);

        const strips_t& strips() const { return m_strips; }
        strips_t& strips() { return m_strips; }
        const crossings_t& corners() const { return m_corners; }

        // A blob is valid if it has at least 3 corners, implying a non-zero area.
        bool valid() const {
            return m_strips.size() >= 2 && m_corners.size() >= 3;
        }

    private:
        strips_t m_strips;
        crossings_t m_corners;
    };
    typedef std::vector<Blob> blobs_t;

    /**
     * @class Tiling
     * @brief Orchestrates the process of creating Blobs from Activities.
     */
    class Tiling {
    public:
        Tiling(const Coordinates& coords, double nudge = 0.0)
            : m_coords(coords), m_nudge(nudge) {}

        // Creates initial blobs from the first activity.
        blobs_t operator()(const Activity& activity);

        // Refines a set of existing blobs with a new activity.
        blobs_t operator()(const blobs_t& prior, const Activity& activity);

    private:
        const Coordinates& m_coords;
        double m_nudge;
    };


    // --- Free Functions ---

    // The main entry point to the algorithm.
    blobs_t make_blobs(const Coordinates& coords,
                       const activities_t& activities,
                       double nudge = 1e-3);

    // Removes invalid blobs (e.g., those with no area) from a collection.
    size_t drop_invalid(blobs_t& blobs);

    // Refines the bounds of strips in each blob to tightly fit the blob's corners.
    void prune(const Coordinates& coords, blobs_t& blobs, double nudge=1e-3);

} // namespace TorchRayGrid
} // namespace WireCell

#endif // WIRECELL_TORCHRAYTILING_H
