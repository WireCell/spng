#include "WireCellSpng/Util.h"

#include <torch/torch.h>
// Name collission for "CHECK" between torch and doctest.
#undef CHECK
#include "WireCellUtil/doctest.h"

#include <iostream>

using namespace WireCell;

// torch's moral equivalent to size_t is:
using INT = int64_t;

static void do_simple_convo(const std::vector<INT>& signal_shape,
                            const std::vector<INT>& response_shape,
                            double sigma = 1.0)
{


    // we do not care about testing performance here, so hard-wire the use of CPU.
    auto topt = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU).requires_grad(false);

    auto signal = torch::zeros(signal_shape, topt);
    const INT npoints = signal_shape[0]/response_shape[0];
    {
        auto x = torch::randint(signal_shape[0], {npoints});
        auto y = torch::randint(signal_shape[1], {npoints});
        signal.index_put_({x,y}, 1.0);
    }

    torch::Tensor resp;
    {
        const double n0 = response_shape[0];
        auto g0 = Torch::gaussian1d(n0/2, sigma, {n0}, 0, n0-1, topt);
        const double n1 = response_shape[1];
        auto g1 = Torch::gaussian1d(n1/2, sigma, {n1}, 0, n1-1, topt);
        resp = torch::outer(g0,g1);
    }

    auto full_shape = Torch::linear_shape({resp}, signal_shape);

    auto kernel = Torch::convo_spec({resp}, full_shape);
    auto full_signal = Torch::pad(signal, 0.0, full_shape);

    auto meas_spec = torch::fft::fft2(full_signal)*kernel;
    auto meas = torch::real(torch::fft::ifft2(meas_spec));
    
    

}

TEST_CASE("spng torch simple convo") {
    // This test:
    // - Make a few single-sample impulses,
    // - convolves with a Gaussian,
    // - produces npz to make sure result visually looks okay.

    do_simple_convo({100,100}, {10,10}, 1);

}
