#include "WireCellSpng/TorchRayGrid.h"
// Name collission for "CHECK" between torch and doctest.
#undef CHECK
#include "WireCellUtil/doctest.h"


// #include "WireCellUtil/svg.hpp"
// using namespace svg;

using namespace WireCell::Spng::TorchRayGrid;

// void do_coords_check_batched(const Coordinates& coords,
//                              const torch::Tensor& view1_batch,
//                              const torch::Tensor& view2_batch)
// {
//     // r0 = coords.zero_crossings[view1, view2] - equivalent batched access
//     torch::Tensor r0_batched = coords.zero_crossings.index({view1_batch, view2_batch, torch::indexing::Ellipsis});

//     // r1 = coords.ray_crossing_batched((view1,0), (view2,0))
//     torch::Tensor zero_ray_batch = torch::zeros_like(view1_batch, torch::kLong); // Batch of zeros for ray index
//     torch::Tensor r1_batched = coords.ray_crossing_batched(
//         {view1_batch, zero_ray_batch},
//         {view2_batch, zero_ray_batch}
//     );

//     std::cout << "\nBatched Check - r0_batched=\n" << r0_batched << std::endl;
//     std::cout << "Batched Check - r1_batched=\n" << r1_batched << std::endl;
//     assert(torch::allclose(r0_batched, r1_batched).item<bool>());

//     // dr0 = coords.ray_crossing_batched((view1,0), (view2,1)) - r0
//     torch::Tensor one_ray_batch = torch::ones_like(view1_batch, torch::kLong); // Batch of ones for ray index
//     torch::Tensor dr0_batched = coords.ray_crossing_batched(
//         {view1_batch, zero_ray_batch},
//         {view2_batch, one_ray_batch}
//     ) - r0_batched;

//     // dr1 = coords.ray_jump[view1, view2] - equivalent batched access
//     torch::Tensor dr1_batched = coords.ray_jump.index({view1_batch, view2_batch, torch::indexing::Ellipsis});

//     std::cout << "Batched Check - dr0_batched=\n" << dr0_batched << std::endl;
//     std::cout << "Batched Check - dr1_batched=\n" << dr1_batched << std::endl;
//     assert(torch::allclose(dr0_batched, dr1_batched).item<bool>());
// }

void do_coords_check_batched(const Coordinates& coords, const torch::Tensor& view1_idx, const torch::Tensor& view2_idx)
{
    int64_t nviews = view1_idx.sizes()[0];
    auto rays_idx = torch::zeros_like(view1_idx);

    // r0 = coords.zero_crossings[view1, view2]
    torch::Tensor r0_batched = coords.zero_crossings.index({view1_idx, view2_idx});
    for (int64_t ind = 0; ind<nviews; ++ind) {
        auto r0 = coords.zero_crossings.index({view1_idx[ind].item<int64_t>(), view2_idx[ind].item<int64_t>()});
        assert ( torch::all(r0 == r0_batched[ind]).item<bool>() );

        rays_idx[ind] = ind;
    }


    torch::Tensor r1_batched = coords.ray_crossing(view1_idx, rays_idx, view2_idx, rays_idx);
    for (int64_t ind = 0; ind<nviews; ++ind) {
        auto r1 = coords.ray_crossing(view1_idx[ind].item<int64_t>(),
                                      rays_idx[ind].item<int64_t>(),
                                      view2_idx[ind].item<int64_t>(),
                                      rays_idx[ind].item<int64_t>());
        assert ( torch::all(r1 == r1_batched[ind]).item<bool>() );
    }
}

void do_coords_check_scalar(const Coordinates& coords, int64_t view1_idx, int64_t view2_idx) {
    // r0 = coords.zero_crossings[view1, view2]
    torch::Tensor r0 = coords.zero_crossings.index({view1_idx, view2_idx});

    // r1 = coords.ray_crossing((view1,0), (view2,0))
    torch::Tensor r1 = coords.ray_crossing(view1_idx, 0, view2_idx, 0);

    std::cout << "r0=" << r0 << std::endl;
    // assert torch.all(r0 == r1)
    assert(torch::all(r0 == r1).item<bool>());

    // dr0 = coords.ray_crossing((view1,0), (view2,1)) - r0
    torch::Tensor dr0 = coords.ray_crossing(view1_idx, 0, view2_idx, 1) - r0;

    // dr1 = coords.ray_jump[view1, view2]
    torch::Tensor dr1 = coords.ray_jump.index({view1_idx, view2_idx});

    std::cout << "dr0=" << dr0 << std::endl;
    // assert torch.all(dr0 == dr1)
    assert(torch::all(dr0 == dr1).item<bool>());
}


TEST_CASE("spng torch raygrid coordinates") {
    const auto views = symmetric_views();
    
    // Python: for iview, view in enumerate(views):
    // In C++, iterating through tensor slices directly is less idiomatic.
    // We access elements/slices by index.
    int64_t nviews = views.sizes()[0];
    for (int64_t iview = 0; iview < nviews; ++iview) {
        torch::Tensor view = views.index({iview}); // Get the (2,2,2) slice for this view
        std::cout << "iview=" << iview << std::endl;

        for (int64_t iray = 0; iray < view.sizes()[0]; ++iray) { // Iterate 2 rays
            torch::Tensor ray = view.index({iray}); // Get the (2,2) slice for this ray
            std::cout << "\tiray=" << iray << std::endl;

            for (int64_t ipt = 0; ipt < ray.sizes()[0]; ++ipt) { // Iterate 2 points
                torch::Tensor pt = ray.index({ipt}); // Get the (2) slice for this point
                std::cout << "\t\tipt=" << ipt << "," << pt << std::endl;
            }
        }
    }

    Coordinates coords(views);

    // Scalar checks
    std::vector<int64_t> view1_indices = {2, 3, 4};
    std::vector<int64_t> view2_indices = {3, 4, 2};

    // Python: for v1, v2 in zip(view1, view2): do_coords_check(coords, v1, v2)
    for (size_t i = 0; i < view1_indices.size(); ++i) {
        do_coords_check_scalar(coords, view1_indices[i], view2_indices[i]);
    }
    torch::Tensor p0 = coords.pitch_location(2, 3, 0, 0, 4);
    std::cout << "p0=" << p0 << std::endl;


    // Batched
    // Create 1D tensors from the vectors for batched calls
    torch::Tensor view1_batch = torch::tensor(view1_indices, torch::kLong);
    torch::Tensor view2_batch = torch::tensor(view2_indices, torch::kLong);


    do_coords_check_batched(coords, view1_batch, view2_batch);    

    torch::Tensor p0_batched = coords.pitch_location(
        2, torch::tensor({0, 1}, torch::kLong),
        3, torch::tensor({0, 1}, torch::kLong),
        4
    );
    std::cout << "Batched p0=" << p0_batched << std::endl;
    // std::string fname = "doctest_raygrid_coordinates.svg";
    // Dimensions dimensions(100, 100);
    // Document doc(fname, Layout(dimensions, Layout::BottomLeft));
    

}
