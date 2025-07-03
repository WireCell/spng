#include "WireCellSpng/TorchRayGrid.h"
#include <iostream>

// make .index() calls simpler
using namespace torch::indexing;

namespace WireCell {
namespace Spng {
namespace TorchRayGrid {

#include <iostream>             // remove, for debug only

namespace funcs {
torch::Tensor pitch(const torch::Tensor& r0, const torch::Tensor& r1) {
    // along ray 0
    torch::Tensor rdir = r0.index({1}) - r0.index({0}); // r0[1] - r0[0]
    
    // transpose to get unit perpendicular
    // torch.tensor([-rdir[1], rdir[0]]) / torch.norm(rdir)
    torch::Tensor uperp = torch::tensor({-rdir.index({1}).item<double>(),
            rdir.index({0}).item<double>()}, torch::kDouble);
    uperp = uperp / torch::norm(rdir);

    // connecting vector between points on either ray
    // r1[0]-r0[0]
    torch::Tensor cvec = r1.index({0}) - r0.index({0});
    
    // project onto the perpendicular
    // torch.dot(cvec, uperp)
    torch::Tensor pdist = torch::dot(cvec, uperp);
    return pdist * uperp;
}

torch::Tensor vector(const torch::Tensor& ray) {
    // ray[1] - ray[0]
    return ray.index({1}) - ray.index({0});
}

torch::Tensor direction(const torch::Tensor& ray) {
    torch::Tensor d = funcs::vector(ray);
    return d / torch::norm(d);
}

torch::Tensor crossing(const torch::Tensor& r0, const torch::Tensor& r1) {
    torch::Tensor p1 = r0.index({0});
    torch::Tensor p2 = r0.index({1});
    torch::Tensor p3 = r1.index({0});
    torch::Tensor p4 = r1.index({1});

    // x1, y1 = p1
    double x1 = p1.index({0}).item<double>();
    double y1 = p1.index({1}).item<double>();
    // x2, y2 = p2
    double x2 = p2.index({0}).item<double>();
    double y2 = p2.index({1}).item<double>();
    // x3, y3 = p3
    double x3 = p3.index({0}).item<double>();
    double y3 = p3.index({1}).item<double>();
    // x4, y4 = p4
    double x4 = p4.index({0}).item<double>();
    double y4 = p4.index({1}).item<double>();

    double denominator_val = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    
    // Using torch::isclose requires tensors, but for a single scalar,
    // a direct comparison with a small epsilon is often more straightforward
    // in C++ for performance, or create a scalar tensor to use torch::isclose.
    // For direct comparison, we'll use a small epsilon.
    // If you need the exact torch.isclose behavior, you'd do:
    // if (torch::isclose(torch::tensor(denominator_val), torch::tensor(0.0)).item<bool>())
    if (std::abs(denominator_val) < 1e-9) { // A common small epsilon
        throw std::runtime_error("parallel lines do not cross");
    }

    double t_numerator_val = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    double t_val = t_numerator_val / denominator_val;
    torch::Tensor t = torch::tensor(t_val, torch::kDouble);

    // intersection_point = p1 + t * (p2 - p1)
    torch::Tensor intersection_point = p1 + t * (p2 - p1);
    return intersection_point;
}
}

torch::Tensor symmetric_views(double width, double height, double pitch_mag, double angle, bool bounds)
{
    std::cerr << "symmetric_views\n";

    // 5 layers, 2 rays per layer, 2 points per ray, 2 coordinates per point
    int nlayers = 3;
    if (bounds) nlayers = 5;
    auto layers = torch::zeros({nlayers,2,2,2}, torch::kDouble);
    
    // "Y" and "Z" axes, names are historic from the old RayGrid assuming 3D.
    const auto why = torch::tensor({1.0, 0.0}, torch::kDouble);
    const auto zee = torch::tensor({0.0, 1.0}, torch::kDouble);
    
    const auto ll = torch::tensor({0.0, 0.0}, torch::kDouble);
    const auto lr = torch::tensor({0.0, width}, torch::kDouble);
    const auto ul = torch::tensor({height, 0.0}, torch::kDouble);
    const auto ur = torch::tensor({height, width}, torch::kDouble);

    int layer = -1;

    if (bounds) {
        // horizontal bounds
        ++layer;
        layers.index_put_({layer, 0, 0}, ll);
        layers.index_put_({layer, 0, 1}, lr);
        layers.index_put_({layer, 1, 0}, ul);
        layers.index_put_({layer, 1, 1}, ur);

        // vertical bounds
        ++layer;
        layers.index_put_({layer, 0, 0}, ll);
        layers.index_put_({layer, 0, 1}, ul);
        layers.index_put_({layer, 1, 0}, lr);
        layers.index_put_({layer, 1, 1}, ur);
    }

    // /-wires
    ++layer;
    {
        const auto d = torch::tensor({cos(angle), sin(angle)}, torch::kDouble);
        const auto p = torch::tensor({-d[1].item<double>(), d[0].item<double>()}, torch::kDouble);
        {
            const auto pjump = 0.5 * pitch_mag * p;
            const auto mjump2 = torch::dot(pjump, pjump);
            layers.index_put_({layer, 0, 0}, ul + why * mjump2 / torch::dot(why, pjump));
            layers.index_put_({layer, 0, 1}, ul + zee * mjump2 / torch::dot(zee, pjump));
        }
        {
            const auto pjump = 1.5 * pitch_mag * p;
            const auto mjump2 = torch::dot(pjump, pjump);
            layers.index_put_({layer, 1, 0}, ul + why * mjump2 / torch::dot(why, pjump));
            layers.index_put_({layer, 1, 1}, ul + zee * mjump2 / torch::dot(zee, pjump));
        }
    }

    // \-wires
    ++layer;
    {
        const auto d = torch::tensor({cos(angle), -sin(angle)}, torch::kDouble);
        const auto p = torch::tensor({-d[1].item<double>(), d[0].item<double>()}, torch::kDouble);
        {
            const auto pjump = 0.5 * pitch_mag * p;
            const auto mjump2 = torch::dot(pjump, pjump);
            layers.index_put_({layer, 0, 0}, ll + why * mjump2 / torch::dot(why, pjump));
            layers.index_put_({layer, 0, 1}, ll + zee * mjump2 / torch::dot(zee, pjump));
        }
        {
            const auto pjump = 1.5 * pitch_mag * p;
            const auto mjump2 = torch::dot(pjump, pjump);
            layers.index_put_({layer, 1, 0}, ll + why * mjump2 / torch::dot(why, pjump));
            layers.index_put_({layer, 1, 1}, ll + zee * mjump2 / torch::dot(zee, pjump));
        }
    }

    // |-wires
    ++layer;
    const auto pjumpw = pitch_mag * zee;
    layers.index_put_({layer, 0, 0}, ll + 0.0 * pjumpw);
    layers.index_put_({layer, 0, 1}, ul + 0.0 * pjumpw);
    layers.index_put_({layer, 1, 0}, ll + 1.0 * pjumpw);
    layers.index_put_({layer, 1, 1}, ul + 1.0 * pjumpw);
 
    return layers;
}

// Use torch::indexing namespace for cleaner code with tensor indexing
using namespace torch::indexing;

Coordinates::Coordinates(const torch::Tensor& views)
{
    // Ensure views is double precision
    if (views.dtype() != torch::kDouble) {
        // Optionally convert, or throw an error if input must strictly be double
        // views = views.to(torch::kDouble);
        std::cerr << "Warning: 'views' tensor is not of type torch::kDouble. Converting." << std::endl;
        const_cast<torch::Tensor&>(views) = views.to(torch::kDouble);
    }
    init(views);
}


torch::Tensor Coordinates::ray_crossing(int64_t view1, int64_t ray1, int64_t view2, int64_t ray2) const
{

    torch::Tensor r00 = zero_crossings.index({view1, view2});
    torch::Tensor w12 = ray_jump.index({view1, view2});
    torch::Tensor w21 = ray_jump.index({view2, view1});
    
    return r00 + static_cast<double>(ray2) * w12 + static_cast<double>(ray1) * w21;
}

torch::Tensor Coordinates::ray_crossing(int64_t view1, const torch::Tensor& ray1,
                                        int64_t view2, const torch::Tensor& ray2) const
{
    // Ensure ray batches are 1D tensors of the same size and type
    assert(ray1.dim() == 1 && ray2.dim() == 1);
    assert(ray1.sizes()[0] == ray2.sizes()[0]);
    assert(ray1.dtype() == torch::kLong && ray2.dtype() == torch::kLong);

    // r00 and w12, w21 are scalar tensors (2 coordinates),
    // they will be broadcasted to the batch size
    torch::Tensor r00_scalar_tensor = zero_crossings.index({view1, view2}); // Shape (2)
    torch::Tensor w12_scalar_tensor = ray_jump.index({view1, view2});       // Shape (2)
    torch::Tensor w21_scalar_tensor = ray_jump.index({view2, view1});       // Shape (2)

    // Expand scalars to (1, 2) to enable broadcasting with (batch_size, 1) * (1, 2)
    // Or let LibTorch handle broadcasting from (2) to (batch_size, 2) when combined with (batch_size, 1)
    // More explicitly, expand the 2-element vectors if they are used as part of a batch-wise operation.
    // However, if we just multiply with unsqueezed ray indices, the result will be (batch_size, 2)
    // which is the desired output.

    // ray1 and ray2 need to be unsqueezed to (batch_size, 1) for multiplication
    // with (2) resulting in (batch_size, 2) after broadcasting.
    return r00_scalar_tensor + ray2.unsqueeze(-1) * w12_scalar_tensor + ray1.unsqueeze(-1) * w21_scalar_tensor;
}

torch::Tensor Coordinates::ray_crossing(const torch::Tensor& view1, const torch::Tensor& ray1,
                                        const torch::Tensor& view2, const torch::Tensor& ray2) const
{
    // Ensure all input batches are 1D tensors of the same size and type
    assert(view1.dim() == 1 && ray1.dim() == 1 && view2.dim() == 1 && ray2.dim() == 1);
    assert(view1.sizes()[0] == ray1.sizes()[0] &&
           view1.sizes()[0] == view2.sizes()[0] &&
           view1.sizes()[0] == ray2.sizes()[0]);
    assert(view1.dtype() == torch::kLong && ray1.dtype() == torch::kLong &&
           view2.dtype() == torch::kLong && ray2.dtype() == torch::kLong);

    // Gather zero_crossings and ray_jump based on batched indices
    torch::Tensor r00_batched = zero_crossings.index({view1, view2, Ellipsis}); // Shape (batch_size, 2)
    torch::Tensor w12_batched = ray_jump.index({view1, view2, Ellipsis});       // Shape (batch_size, 2)
    torch::Tensor w21_batched = ray_jump.index({view2, view1, Ellipsis});       // Shape (batch_size, 2)

    // ray1 and ray2 need to be unsqueezed to (batch_size, 1) for correct broadcasting
    // with (batch_size, 2) tensors.
    return r00_batched + ray2.unsqueeze(-1) * w12_batched + ray1.unsqueeze(-1) * w21_batched;
}



torch::Tensor Coordinates::pitch_location(int64_t view1, int64_t ray1, int64_t view2, int64_t ray2,
                                          int64_t view_idx) const
{
    double b_val = b.index({view1, view2, view_idx}).item<double>();
    double a12_val = a.index({view1, view2, view_idx}).item<double>();
    double a21_val = a.index({view2, view1, view_idx}).item<double>();

    return torch::tensor(b_val + static_cast<double>(ray2) * a12_val + static_cast<double>(ray1) * a21_val, torch::kDouble);
}

torch::Tensor Coordinates::pitch_location(int64_t view1, const torch::Tensor& ray1,
                                          int64_t view2, const torch::Tensor& ray2,
                                          int64_t view_idx) const {
    // Ensure ray batches are 1D tensors of the same size and type
    assert(ray1.dim() == 1 && ray2.dim() == 1);
    assert(ray1.sizes()[0] == ray2.sizes()[0]);
    assert(ray1.dtype() == torch::kLong && ray2.dtype() == torch::kLong);

    // Get scalar values from the 'a' and 'b' tensors
    double b_val = b.index({view1, view2, view_idx}).item<double>();
    double a12_val = a.index({view1, view2, view_idx}).item<double>();
    double a21_val = a.index({view2, view1, view_idx}).item<double>();

    // Perform element-wise multiplication and addition.
    // ray1 and ray2 (kLong) need to be cast to kDouble for multiplication with doubles.
    return torch::tensor(b_val, torch::kDouble) + ray2.to(torch::kDouble) * a12_val + ray1.to(torch::kDouble) * a21_val;
}

torch::Tensor Coordinates::pitch_location(const torch::Tensor& view1, const torch::Tensor& ray1,
                                          const torch::Tensor& view2, const torch::Tensor& ray2,
                                          const torch::Tensor& view_idx) const {
    // Ensure all input batches are 1D tensors of the same size and type
    assert(view1.dim() == 1 && ray1.dim() == 1 && view2.dim() == 1 && ray2.dim() == 1 && view_idx.dim() == 1);
    assert(view1.sizes()[0] == ray1.sizes()[0] &&
           view1.sizes()[0] == view2.sizes()[0] &&
           view1.sizes()[0] == ray2.sizes()[0] &&
           view1.sizes()[0] == view_idx.sizes()[0]);
    assert(view1.dtype() == torch::kLong && ray1.dtype() == torch::kLong &&
           view2.dtype() == torch::kLong && ray2.dtype() == torch::kLong &&
           view_idx.dtype() == torch::kLong);

    // Gather 'b' and 'a' values using advanced indexing.
    // The result will be 1D tensors of shape (batch_size).
    torch::Tensor b_batched = b.index({view1, view2, view_idx});
    torch::Tensor a12_batched = a.index({view1, view2, view_idx});
    torch::Tensor a21_batched = a.index({view2, view1, view_idx});

    // Perform the calculation.
    // ray1 and ray2 (kLong) need to be cast to kDouble for multiplication.
    return b_batched + ray2.to(torch::kDouble) * a12_batched + ray1.to(torch::kDouble) * a21_batched;
}    



void Coordinates::init(const torch::Tensor& views)
{
    nviews = views.sizes()[0];

    // Initialize member tensors with torch::zeros and kDouble
    pitch_mag = torch::zeros({nviews}, torch::kDouble);
    pitch_dir = torch::zeros({nviews, 2}, torch::kDouble);
    center = torch::zeros({nviews, 2}, torch::kDouble);
    zero_crossings = torch::zeros({nviews, nviews, 2}, torch::kDouble);
    ray_jump = torch::zeros({nviews, nviews, 2}, torch::kDouble);
    a = torch::zeros({nviews, nviews, nviews}, torch::kDouble);
    b = torch::zeros({nviews, nviews, nviews}, torch::kDouble);

    // Per-view things
    // Loop through the first dimension of 'views' (N-views)
    for (int64_t layer = 0; layer < nviews; ++layer) {
        // views[layer] gives a (2, 2, 2) tensor, representing (2 rays, 2 endpoints, 2 coords) for this view.
        // So, r0 and r1 are (2, 2) tensors.
        torch::Tensor r0 = views.index({layer, 0}); // views[layer, 0]
        torch::Tensor r1 = views.index({layer, 1}); // views[layer, 1]

        torch::Tensor rpv = funcs::pitch(r0, r1);
        double rpl = torch::norm(rpv).item<double>();

        pitch_mag.index_put_({layer}, rpl);
        pitch_dir.index_put_({layer}, rpv / rpl);
        center.index_put_({layer}, 0.5 * (r0.index({0}) + r0.index({1}))); // 0.5 * (r0[0] + r0[1])
    }

    // Cross-view things
    for (int64_t il = 0; il < nviews; ++il) {
        torch::Tensor rl0 = views.index({il, 0}); // rl0 = views[il, 0]
        torch::Tensor rl1 = views.index({il, 1}); // rl1 = views[il, 1]

        for (int64_t im = 0; im < nviews; ++im) {
            torch::Tensor rm0 = views.index({im, 0}); // rm0 = views[im, 0]
            torch::Tensor rm1 = views.index({im, 1}); // rm1 = views[im, 1]

            // Special case diagonal values
            if (il == im) {
                zero_crossings.index_put_({il, im}, center.index({il}));
                ray_jump.index_put_({il, im}, funcs::direction(rl0)); // Equivalent to funcs.direction(views[il, 0])
                continue;
            }

            if (il < im) {
                torch::Tensor p;
                try {
                    p = funcs::crossing(rl0, rm0);
                    zero_crossings.index_put_({il, im}, p);
                    zero_crossings.index_put_({im, il}, p); // Exploit symmetry
                    ray_jump.index_put_({il, im}, funcs::crossing(rl0, rm1) - p);
                    ray_jump.index_put_({im, il}, funcs::crossing(rm0, rl1) - p);
                } catch (const std::runtime_error& e) {
                    std::cout << "skipping parallel view pair: il=" << il << " im=" << im << " (" << e.what() << ")" << std::endl;
                    // Python's `continue` here effectively means these elements remain zeros
                    // which is what our `torch::zeros` initialization does.
                }
            }
        }
    }

    // Triple layer things
    for (int64_t ik = 0; ik < nviews; ++ik) {
        torch::Tensor pk = pitch_dir.index({ik});
        double cp = torch::dot(center.index({ik}), pk).item<double>();

        for (int64_t il = 0; il < nviews; ++il) {
            if (il == ik) {
                continue;
            }

            for (int64_t im = 0; im < il; ++im) { // Note: original Python had `for im in range(il)`
                                                // which means im < il.
                if (im == ik) {
                    continue;
                }
                
                // zero_crossings[il, im] is a 2D vector, pk is a 2D vector. Dot product returns a scalar tensor.
                double rlmpk = torch::dot(zero_crossings.index({il, im}), pk).item<double>();
                double wlmpk = torch::dot(ray_jump.index({il, im}), pk).item<double>();
                double wmlpk = torch::dot(ray_jump.index({im, il}), pk).item<double>();
                
                a.index_put_({il, im, ik}, wlmpk);
                a.index_put_({im, il, ik}, wmlpk); // Python's a[im,il,ik] = wmlpk;
                b.index_put_({il, im, ik}, rlmpk - cp);
                b.index_put_({im, il, ik}, rlmpk - cp); // Python's b[im,il,ik] = rlmpk - cp
            }
        }
    }
}


} // namespace TorchRayGrid
} // namespace Spng
} // namespace WireCell
