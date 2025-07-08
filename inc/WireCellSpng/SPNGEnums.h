#ifndef SPNGENUMS_H
#define SPNGENUMS_H
#include <string>
namespace WireCell::SPNG {
    enum TensorKind {
        kBatch,
        kChannel,
        kTick
    };
    std::string KindAsString(const TensorKind & kind);
    
    enum TensorDomain {
        kNull, //Exclusively for batch
        kInterval,
        kFourier
    };
    std::string DomainAsString(const TensorDomain & domain);
}
#endif