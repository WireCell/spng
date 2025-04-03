#include "WireCellSpng/Decon.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
// #include "WireCellSpng/ITorchFieldResponse.h"
#include "WireCellSpng/ITorchSpectrum.h"
// #include "WireCellSpng/ITorchColdElecResponse.h"

WIRECELL_FACTORY(SPNGDecon, WireCell::SPNG::Decon,
                 WireCell::INamed,
                 WireCell::ITorchTensorSetFilter, WireCell::IConfigurable)

WireCell::SPNG::Decon::Decon()
  : Aux::Logger("SPNGDecon", "spng") {

}

WireCell::SPNG::Decon::~Decon() {};


void WireCell::SPNG::Decon::configure(const WireCell::Configuration& config) {

    m_frer_spectrum = get(config, "frer_spectrum", m_frer_spectrum);
    base_frer_spectrum = Factory::find_tn<ITorchSpectrum>(m_frer_spectrum);

    m_wire_filter = get(config, "wire_filter", m_wire_filter);
    base_wire_filter = Factory::find_tn<ITorchSpectrum>(m_wire_filter);
}

bool WireCell::SPNG::Decon::operator()(const input_pointer& in, output_pointer& out) {
    out = nullptr;
    if (!in) {
        //Why is this needed?
        log->debug("EOS ");
        return true;
    }
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
    for (const auto & s : tensor_clone.sizes()) std::cout << s << std::endl;
    std::cout << "Done" << std::endl;
    // std::cout << tensor_clone << std::endl;

    auto frer_spectrum_tensor = base_frer_spectrum->spectrum().clone();
    std::cout << "frer sizes" << std::endl;
    for (const auto & s : frer_spectrum_tensor.sizes()) std::cout << s << std::endl;
    frer_spectrum_tensor = torch::fft::rfft2(frer_spectrum_tensor);
    for (const auto & s : frer_spectrum_tensor.sizes()) std::cout << s << std::endl;

    // std::cout << frer_spectrum_tensor << std::endl;

    tensor_clone /= frer_spectrum_tensor;
    // std::cout << tensor_clone << std::endl;

    tensor_clone = torch::fft::irfft2(tensor_clone);

    // std::cout << tensor_clone << std::endl;

    // TODO: set md
    Configuration set_md;

    //Clone the tensor to take ownership of the memory and put into 
    //output 
    std::vector<ITorchTensor::pointer> itv{
        std::make_shared<SimpleTorchTensor>(tensor_clone.clone())
    };
    out = std::make_shared<SimpleTorchTensorSet>(
        in->ident(), set_md,
        std::make_shared<std::vector<ITorchTensor::pointer>>(itv)
    );

    return true;
}