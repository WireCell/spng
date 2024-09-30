#include "WireCellUtil/Units.h"
#include "WireCellUtil/doctest.h"

// #pragma GCC diagnostic push
// #pragma GCC diagnostic warning "-Wvariadic-macros"
// #pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <torch/torch.h>
// #pragma GCC diagnostic pop

using namespace WireCell;

TEST_CASE("spng torch basics") {
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    
}
TEST_CASE("spng torch convo") {
    auto S = at::rand(100);
    auto S2 = at::rand({10,100});

    // auto rc = torch::rand(10);
    // auto rt = torch::rand(100);
    // auto R = torch::outer(rc, rt);
        

    // // 2D one-shot
    // auto cS = torch::fft::fft2(S);
    // auto cR = torch::fft::fft2(R);
    // auto cM = cS*cR;
    // auto M = torch::fft::ifft2(cM);


    // // dimension-wise
    // auto cSt = torch::fft::fft2(S, torch::nullopt, {1});
    // auto crt = torch::fft::fft(rt);
    // // row-wise multiply
    // auto cMt = /* cSt*crt */;
    // auto ccMtc = torch::fft::fft2(cMt, torch::nullopt, {0});
    // auto crc = torch::fft::fft(rc);
    // // col-wise multiply
    // auto cMtc = ;
    // Mtc = torch::fft::ifft2(cMtc)
        
    


}
