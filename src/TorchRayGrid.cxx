/**
 * TorchRayGrid.cxx
 *
 * Implementation of the torch-based Coordinates class.
 * The constructor contains the core logic for pre-calculating the geometric
 * tensors. This logic is vectorized to replace the original loops with
 * efficient tensor operations.
 */
#include "WireCellSpng/TorchRayGrid.h"
#include <iostream>

namespace WireCell {
namespace TorchRayGrid {

// Helper function to solve for the closest approach between batches of non-parallel rays.
// This is the core of calculating crossing points.
// It solves the 2x2 system of linear equations described in the design thoughts.
// ray_dirs shape: [N, 2, 2], ray_points shape: [N, 2, 2]
torch::Tensor find_crossing_points(const torch::Tensor& ray_dirs, const torch::Tensor& ray_points) {
    auto p1 = ray_points.slice(1, 0, 1).squeeze(1); // Shape: [N, 2]
    auto p2 = ray_points.slice(1, 1, 2).squeeze(1); // Shape: [N, 2]
    auto v1 = ray_dirs.slice(1, 0, 1).squeeze(1);   // Shape: [N, 2]
    auto v2 = ray_dirs.slice(1, 1, 2).squeeze(1);   // Shape: [N, 2]

    auto p2_minus_p1 = p2 - p1;

    // Build the matrix A for the system A * [t, s]^T = B
    auto v1_dot_v1 = torch::sum(v1 * v1, -1);
    auto v2_dot_v2 = torch::sum(v2 * v2, -1);
    auto v1_dot_v2 = torch::sum(v1 * v2, -1);

    auto A11 = v1_dot_v1;
    auto A12 = -v1_dot_v2;
    auto A21 = v1_dot_v2;
    auto A22 = -v2_dot_v2;

    // Stack to create the matrix A of shape [N, 2, 2]
    auto A = torch::stack({
        torch::stack({A11, A12}, -1),
        torch::stack({A21, A22}, -1)
    }, -2);

    // Build the vector B
    auto B1 = torch::sum(p2_minus_p1 * v1, -1);
    auto B2 = torch::sum(p2_minus_p1 * v2, -1);
    auto B = torch::stack({B1, B2}, -1).unsqueeze(-1); // Shape [N, 2, 1]

    // Solve for t and s
    auto ts = torch::linalg::solve(A, B, true).squeeze(-1); // Shape [N, 2]
    auto t = ts.slice(1, 0, 1); // Shape [N, 1]
    
    // Calculate crossing point on the first ray of each pair
    // crossing_point = p1 + t * v1
    return p1 + t * v1;
}


Coordinates::Coordinates(const torch::Tensor& rays, const torch::Device& device)
    : m_nlayers(rays.size(0)), m_device(device) {

    // Ensure all tensors are on the correct device and use double precision
    auto opts = torch::TensorOptions().device(m_device).dtype(torch::kF64);
    torch::Tensor proj_rays = rays.to(opts);

    // --- Step 1: Calculate per-layer quantities ---

    // Ray centers and direction vectors
    auto ray0_p1 = proj_rays.select(1, 0).select(1, 0); // Shape [nlayers, 2]
    auto ray0_p2 = proj_rays.select(1, 0).select(1, 1);
    auto ray1_p1 = proj_rays.select(1, 1).select(1, 0);
    auto ray1_p2 = proj_rays.select(1, 1).select(1, 1);

    m_center = (ray0_p1 + ray0_p2) / 2.0; // Center of ray 0 in each layer
    torch::Tensor ray_dirs = (ray0_p2 - ray0_p1).to(opts);
    torch::Tensor ray_dir_unit = ray_dirs / torch::linalg::norm(ray_dirs, c10::nullopt, 1, true, {});

    // Pitch vector is the component of vector between ray centers perpendicular to ray direction
    auto center1 = (ray1_p1 + ray1_p2) / 2.0;
    auto center_diff = center1 - m_center;
    auto proj_on_ray_dir = torch::sum(center_diff * ray_dir_unit, 1, true) * ray_dir_unit;
    auto pitch_vec = center_diff - proj_on_ray_dir;

    // fixme: no keepdim given, pass default false. check it is okay.
    m_pitch_mag = torch::linalg::norm(pitch_vec, c10::nullopt, 1, false, {});
    m_pitch_dir = pitch_vec / m_pitch_mag.unsqueeze(1);

    // --- Step 2: Calculate cross-layer quantities ---
    m_zero_crossing = torch::zeros({m_nlayers, m_nlayers, 2}, opts);
    m_ray_jump = torch::zeros({m_nlayers, m_nlayers, 2}, opts);

    // Create indices for all layer pairs (l, m)
    auto l_indices = torch::arange(m_nlayers, torch::kLong).view({-1, 1}).expand({-1, m_nlayers}).flatten();
    auto m_indices = torch::arange(m_nlayers, torch::kLong).view({1, -1}).expand({m_nlayers, -1}).flatten();
    auto non_diag_mask = l_indices != m_indices;
    
    // Filter to only non-diagonal pairs
    l_indices = l_indices.index({non_diag_mask});
    m_indices = m_indices.index({non_diag_mask});

    // Get ray0 centers and directions for all pairs
    auto r0_centers_l = m_center.index({l_indices});
    auto r0_centers_m = m_center.index({m_indices});
    auto r0_dirs_l = ray_dir_unit.index({l_indices});
    auto r0_dirs_m = ray_dir_unit.index({m_indices});
    
    // Solve for zero-crossing points in a batch
    auto batch_crossings = find_crossing_points(
        torch::stack({r0_dirs_l, r0_dirs_m}, 1),
        torch::stack({r0_centers_l, r0_centers_m}, 1)
    );
    m_zero_crossing.index_put_({l_indices, m_indices}, batch_crossings);

    // Calculate ray_jump: displacement along ray l0 for a one-unit step in ray m
    // This corresponds to the crossing of ray l0 with ray m1.
    auto r1_centers_m = center1.index({m_indices});
    auto r1_dirs_m = ray_dir_unit.index({m_indices}); // Assuming same direction for ray 0 and 1
    
    auto jump_crossings = find_crossing_points(
        torch::stack({r0_dirs_l, r1_dirs_m}, 1),
        torch::stack({r0_centers_l, r1_centers_m}, 1)
    );
    m_ray_jump.index_put_({l_indices, m_indices}, jump_crossings - m_zero_crossing.index({l_indices, m_indices}));
    
    // Diagonal elements of ray_jump are the unit direction vectors
    auto diag_indices = torch::arange(m_nlayers, torch::kLong);
    m_ray_jump.index_put_({diag_indices, diag_indices}, ray_dir_unit);

    // --- Step 3: Calculate triple-layer quantities (a and b tensors) ---
    // These are coefficients for fast pitch location calculation.
    m_a = torch::zeros({m_nlayers, m_nlayers, m_nlayers}, opts);
    m_b = torch::zeros({m_nlayers, m_nlayers, m_nlayers}, opts);

    // Expand tensors to 3D for broadcasting
    auto pn = m_pitch_dir.view({1, 1, m_nlayers, 2});
    auto cp = torch::sum(m_center * m_pitch_dir, 1).view({1, 1, m_nlayers}); // center dot pitch_dir
    auto zc = m_zero_crossing.view({m_nlayers, m_nlayers, 1, 2});
    auto rj_lm = m_ray_jump.view({m_nlayers, m_nlayers, 1, 2});

    // Batched dot products
    // b[l,m,n] = zero_crossing(l,m) . pitch_dir(n) - center(n) . pitch_dir(n)
    m_b = torch::sum(zc * pn, 3) - cp;

    // a[l,m,n] = ray_jump(l,m) . pitch_dir(n)
    m_a = torch::sum(rj_lm * pn, 3);
}

torch::Tensor Coordinates::ray_crossing_batch(const torch::Tensor& crossings) const {
    auto layer1 = crossings.slice(1, 0, 1).select(2, 0).squeeze(1); // Shape [N]
    auto grid1  = crossings.slice(1, 0, 1).select(2, 1).squeeze(1);
    auto layer2 = crossings.slice(1, 1, 2).select(2, 0).squeeze(1);
    auto grid2  = crossings.slice(1, 1, 2).select(2, 1).squeeze(1);
    
    // Fetch pre-computed values using advanced indexing
    auto zc = m_zero_crossing.index({layer1, layer2}); // Shape [N, 2]
    auto rj_lm = m_ray_jump.index({layer1, layer2});    // Shape [N, 2]
    auto rj_ml = m_ray_jump.index({layer2, layer1});    // Shape [N, 2]

    // Vectorized calculation: R = R00 + j*W_lm + i*W_ml
    auto i = grid1.to(zc.options()).unsqueeze(1);
    auto j = grid2.to(zc.options()).unsqueeze(1);

    return zc + j * rj_lm + i * rj_ml;
}

torch::Tensor Coordinates::pitch_location_batch(const torch::Tensor& crossings, const torch::Tensor& other_layers) const {
    auto layer1 = crossings.slice(1, 0, 1).select(2, 0).squeeze(1);
    auto grid1  = crossings.slice(1, 0, 1).select(2, 1).squeeze(1);
    auto layer2 = crossings.slice(1, 1, 2).select(2, 0).squeeze(1);
    auto grid2  = crossings.slice(1, 1, 2).select(2, 1).squeeze(1);

    // Fetch coefficients
    auto a_lmn = m_a.index({layer1, layer2, other_layers});
    auto a_mln = m_a.index({layer2, layer1, other_layers});
    auto b_lmn = m_b.index({layer1, layer2, other_layers});

    auto i = grid1.to(a_lmn.options());
    auto j = grid2.to(a_lmn.options());

    // Vectorized calculation: P = j*a_lmn + i*a_mln + b_lmn
    return j * a_lmn + i * a_mln + b_lmn;
}

} // namespace TorchRayGrid
} // namespace WireCell
