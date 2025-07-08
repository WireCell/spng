#include "WireCellSpng/SPNGEnums.h"
#include <stdexcept>

std::string WireCell::SPNG::KindAsString(const TensorKind & kind) {
    if (kind == kBatch) {
        return "batch";
    }
    else if (kind == kChannel) {
        return "channel";
    }
    else if (kind == kTick) {
        return "tick";
    }

    throw std::invalid_argument("Unknown Tensor Kind");
    return "";
};

std::string WireCell::SPNG::DomainAsString(const TensorDomain & domain) {
    if (domain == kInterval) {
        return "interval";
    }
    else if (domain == kFourier) {
        return "fourier";
    }
    else if (domain == kNull) {
        return "null";
    }

    throw std::invalid_argument("Unknown Tensor Domain");
    return "";
};