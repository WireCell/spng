#include "WireCellUtil/Units.h"
#include "WireCellUtil/doctest.h"
#include <ATen/ATen.h>

using namespace WireCell;

TEST_CASE("spng torch basics") {
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    
}
