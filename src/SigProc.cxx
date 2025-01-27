#include "WireCellSpng/SigProc.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensor.h"

WIRECELL_FACTORY(SPNGSigProc, WireCell::SPNG::SigProc,
                 WireCell::INamed,
                 WireCell::ITorchTensorFilter, WireCell::IConfigurable)

WireCell::SPNG::SigProc::SigProc()
  : Aux::Logger("SPNGSigProc", "spng") {
    // get wires for each plane

    // std::cout << m_anode->channels().size() << " " << nwire_u << " " << nwire_v << " " << nwire_w << std::endl;
}

WireCell::SPNG::SigProc::~SigProc() {};

bool WireCell::SPNG::SigProc::operator()(const input_pointer& in, output_pointer& out) {
    
    //Get the cloned tensor from the input
    auto tensor_clone = in->tensor();

    //Do some transformations

    std::cout << tensor_clone << std::endl;

    //Placeholder
    out = WireCell::SimpleTorchTensor::pointer(in);

    return true;
}