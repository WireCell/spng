#include "WireCellSpng/Decon.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/ITorchFieldResponse.h"

WIRECELL_FACTORY(SPNGDecon, WireCell::SPNG::Decon,
                 WireCell::INamed,
                 WireCell::ITorchTensorSetFilter, WireCell::IConfigurable)

WireCell::SPNG::Decon::Decon()
  : Aux::Logger("SPNGDecon", "spng") {
    // get wires for each plane

    // std::cout << m_anode->channels().size() << " " << nwire_u << " " << nwire_v << " " << nwire_w << std::endl;
}

WireCell::SPNG::Decon::~Decon() {};


void WireCell::SPNG::Decon::configure(const WireCell::Configuration& config) {
    m_field_response = get(config, "field_response", m_field_response);
    auto base_response = Factory::find_tn<ITorchFieldResponse>(m_field_response);
    std::cout << base_response << std::endl;
}

bool WireCell::SPNG::Decon::operator()(const input_pointer& in, output_pointer& out) {
    
    //Get the cloned tensor from the input
    // auto tensor_clone = in->tensor();

    // //Do some transformations

    // std::cout << tensor_clone << std::endl;

    //Placeholder
    out = WireCell::SPNG::SimpleTorchTensorSet::pointer(in);

    return true;
}