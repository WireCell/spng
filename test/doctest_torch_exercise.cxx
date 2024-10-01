#include "WireCellUtil/Units.h"
#include "WireCellUtil/TimeKeeper.h"

#include <torch/torch.h>
// Name collission for "CHECK" between torch and doctest.
#undef CHECK
#include "WireCellUtil/doctest.h"

#include <iostream>

using namespace WireCell;

TEST_CASE("spng torch basics") {
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    
}

using namespace torch::indexing;

static bool quiet = false;

template<typename T>
void dump(const T& ten, const std::string& msg="")
{
    if (quiet) return;
    if (msg.size()) {
        std::cerr << msg << "\n";
    }
    if (ten.dtype() == torch::kComplexDouble) {
        std::cerr << "real\n";
        std::cerr << torch::real(ten) << "\n";
        std::cerr << "imag\n";
        std::cerr << torch::imag(ten) << "\n";
    }
    else{
        std::cerr << ten << "\n";
    }    
}

static
void test_spng_torch_convo(torch::Device& device, const std::vector<long int>& shape = {2,5})
{
    // Cyclically convolve a 2D uniform random array with two 1D uniform random
    // arrays independently spanning the two dimensions in two ways: via
    // one-shot 2D DFT and via 2 1D, per-dimension DFTs.

    // In Python/Numpy:
    /*
      import numpy
      shape = (5, 10)
      s = numpy.random.uniform(0, 1, size=shape)
      f0 = numpy.random.uniform(0, 1, size=(shape[0],))
      f1 = numpy.random.uniform(0, 1, size=(shape[1],))
      f = numpy.outer(f0, f1)
      S = numpy.fft.fft2(s)
      F = numpy.fft.fft2(f)
      M = S*F
      m = numpy.fft.ifft2(M)
      S1 = numpy.fft.fft2(s, axes=(1,))
      F1 = numpy.fft.fft(f1)
      SF1 = S1*F1
      SF01 = numpy.fft.fft2(SF1, axes=(0,))
      F0 = numpy.fft.fft(f0).reshape(shape[0], 1)
      M01 = SF01*F0
      m01 = numpy.fft.ifft2(M01)
      dr = numpy.abs(numpy.real(m) - numpy.real(m01))
      assert numpy.all(dr < 1e-14)
     */
    auto topt = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(device)
        .requires_grad(false);


    // The "signal" in interval domain.
    // auto s = torch::rand(at::IntArrayRef(shape));
    auto s = torch::rand(shape, topt);
    dump(s, "s");

    // The "responses" along each dimension.
    auto f0 = torch::rand(shape[0], topt);
    auto f1 = torch::rand(shape[1], topt);

    // The total 2D response
    torch::Tensor f = torch::outer(f0, f1);

    dump(f0, "f0");
    dump(f1, "f1");
    dump(f, "f");

    {                           // peek at array shape
        auto sizes = f.sizes();
        // std::cerr << sizes << "\n";
        size_t size = sizes.size();
        REQUIRE(size == 2);
        REQUIRE(sizes[0] == shape[0]);
        REQUIRE(sizes[1] == shape[1]);
    }

    // One shot 2D convo
    auto S = torch::fft::fft2(s);
    auto F = torch::fft::fft2(f);
    auto M = S*F;
    auto m = torch::fft::ifft2(M); // complex

    // Per dimension convo
    auto S1 = torch::fft::fft2(s, torch::nullopt, {1});
    auto F1 = torch::fft::fft(f1);
    dump(S1, "S1");
    dump(F1, "F1");

    // auto SF1 = torch::zeros(shape);
    auto SF1 = S1*F1;
    dump(SF1, "SF1");

    auto SF01 = torch::fft::fft2(SF1, torch::nullopt, {0});
    dump(SF01, "SF01");

    auto F0 = torch::fft::fft(f0).reshape({-1,1});
    auto M01 = SF01 * F0;
    auto m01 = torch::fft::ifft2(M01);

    auto m_r = torch::real(m);
    auto m_i = torch::imag(m);
    dump(m, "m");
    auto m01_r = torch::real(m01);
    auto m01_i = torch::imag(m01);
    dump(m01, "m01");

    auto dr = torch::abs(m_r - m01_r);
    dump(dr, "dr");
    auto small = torch::all( dr < 1e-14 );
    dump(small, "small");

    REQUIRE(small.item<bool>());

}


TEST_CASE("spng torch convo") {
    std::vector<long int> small = {2,5};

    torch::Device cpu(torch::kCPU);
    test_spng_torch_convo(cpu);
    torch::Device gpu(torch::kCUDA);
    test_spng_torch_convo(gpu);
    
    quiet = true;
    TimeKeeper tk("spng torch convo speed test");
    const size_t tries = 100000;

    std::vector<long int> big = {1000,10000};
    for (size_t ind=0; ind<tries; ++ind) {
        test_spng_torch_convo(cpu);
    }
    tk("cpu 10^3 x 10^4");
    for (size_t ind=0; ind<tries; ++ind) {
        test_spng_torch_convo(gpu);
    }
    tk("gpu 10^3 x 10^4");

    std::vector<long int> big2 = {1024,8192};
    for (size_t ind=0; ind<tries; ++ind) {
        test_spng_torch_convo(cpu);
    }
    tk("cpu 2^10 x 2^13");
    for (size_t ind=0; ind<tries; ++ind) {
        test_spng_torch_convo(gpu);
    }
    tk("gpu 2^10 x 2^13");


    std::cerr << tk.summary() << "\n";

}
