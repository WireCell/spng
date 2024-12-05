#include "WireCellSpng/SPNGSigProc.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensor.h"

WIRECELL_FACTORY(SPNGSigProc, WireCell::SPNG::SPNGSigProc,
                 WireCell::INamed,
                 WireCell::ITorchTensorFilter, WireCell::IConfigurable)

WireCell::SPNG::SPNGSigProc::SPNGSigProc()
  : Aux::Logger("SPNGSigProc", "spng") {
    // get wires for each plane

    // std::cout << m_anode->channels().size() << " " << nwire_u << " " << nwire_v << " " << nwire_w << std::endl;
}

WireCell::SPNG::SPNGSigProc::~SPNGSigProc() {};

bool WireCell::SPNG::SPNGSigProc::operator()(const input_pointer& in, output_pointer& out) {
    
    //Get the cloned tensor from the input
    auto tensor_clone = in->tensor();

    //Do some transformations

    std::cout << tensor_clone << std::endl;

    //Placeholder
    out = WireCell::SimpleTorchTensor::pointer(in);

    return true;
}