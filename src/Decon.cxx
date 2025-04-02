#include "WireCellSpng/Decon.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
// #include "WireCellSpng/ITorchFieldResponse.h"
#include "WireCellSpng/ITorchSpectrum.h"
// #include "WireCellSpng/ITorchColdElecResponse.h"

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
    // m_field_response = get(config, "field_response", m_field_response);
    // auto base_field_response = Factory::find_tn<ITorchSpectrum>(m_field_response);

    // m_coldelec_response = get(config, "coldelec_response", m_coldelec_response);
    // auto base_coldelec_response = Factory::find_tn<ITorchSpectrum>(m_coldelec_response);

    m_frer_spectrum = get(config, "frer_spectrum", m_frer_spectrum);
    base_frer_spectrum = Factory::find_tn<ITorchSpectrum>(m_frer_spectrum);
    // std::cout << base_frer_spectrum->spectrum() << std::endl;
}

bool WireCell::SPNG::Decon::operator()(const input_pointer& in, output_pointer& out) {
    
    log->debug("Running Decon");
    //Get the cloned tensor from the input
    auto tensor_clone = in->tensors()->at(0)->tensor().clone();
    for (const auto & s : tensor_clone.sizes()) std::cout << s << std::endl;
    // std::cout << tensor_clone << std::endl;
    //FFT on time dim
    tensor_clone = torch::fft::rfft(tensor_clone, std::nullopt, 1);

    //FFT on chan dim
    for (const auto & s : tensor_clone.sizes()) std::cout << s << std::endl;
    std::cout << "Running wire-fft on time-fft'd input" << std::endl;

    tensor_clone = torch::fft::fft(tensor_clone, std::nullopt, 0);
    std::cout << "Done" << std::endl;

    auto frer_spectrum_tensor = base_frer_spectrum->spectrum().clone();
    std::cout << "frer sizes" << std::endl;
    for (const auto & s : frer_spectrum_tensor.sizes()) std::cout << s << std::endl;
    frer_spectrum_tensor = torch::fft::rfft2(frer_spectrum_tensor);

    tensor_clone /= frer_spectrum_tensor;

    tensor_clone = torch::fft::irfft2(tensor_clone);

    //Placeholder
    out = WireCell::SPNG::SimpleTorchTensorSet::pointer(in);

    return true;
}