#include "WireCellSpng/FrameToTensor.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellAux/FrameTools.h"
#include "WireCellAux/SimpleFrame.h"

WIRECELL_FACTORY(FrameToTensor, WireCell::SPNG::FrameToTensor,
                 WireCell::INamed,
                 WireCell::SPNG::IFrameToTensor)


using namespace WireCell;

WireCell::Configuration SPNG::FrameToTensor::default_configuration() const
{
    Configuration cfg;
    cfg["Test"] = true;
    return cfg;
}

bool SPNG::FrameToTensor::operator()(const input_pointer& in, output_pointer& out) {
    out = nullptr;
    if (!in) return true;
    // const size_t ntraces = in->traces()->size();
    
    torch::Tensor output_tensor = torch::zeros({100, 100, 3});
    out = SimpleTorchTensor::pointer(
        new SimpleTorchTensor(output_tensor.clone()));
    
    return true;
}
