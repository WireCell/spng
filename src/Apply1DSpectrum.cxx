#include "WireCellSpng/Apply1DSpectrum.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/ITorchSpectrum.h"

WIRECELL_FACTORY(SPNGApply1DSpectrum, WireCell::SPNG::Apply1DSpectrum,
                 WireCell::INamed,
                 WireCell::ITorchTensorSetFilter, WireCell::IConfigurable)

WireCell::SPNG::Apply1DSpectrum::Apply1DSpectrum()
  : Aux::Logger("SPNGApply1DSpectrum", "spng") {

}

WireCell::SPNG::Apply1DSpectrum::~Apply1DSpectrum() {};


void WireCell::SPNG::Apply1DSpectrum::configure(const WireCell::Configuration& config) {

    m_base_spectrum_name = get(config, "base_spectrum_name", m_base_spectrum_name);
    log->debug("Loading Spectrum {}", m_base_spectrum_name);
    m_base_spectrum = Factory::find_tn<ITorchSpectrum>(m_base_spectrum_name);

    m_dimension = get(config, "dimension", m_dimension);
    log->debug("Will apply to dimension {}", m_dimension);
}

bool WireCell::SPNG::Apply1DSpectrum::operator()(const input_pointer& in, output_pointer& out) {
    out = nullptr;
    if (!in) {
        log->debug("EOS ");
        return true;
    }
    log->debug("Running Apply1DSpectrum");

    //Get the cloned tensor from the input
    auto tensor_clone = in->tensors()->at(0)->tensor().clone();
    auto sizes = tensor_clone.sizes();
    std::vector<int64_t> shape;
    for (const auto & s : sizes) {
        shape.push_back(s);
        log->debug("Got shape {}", s);
    }

    // if (m_dimension >= shape.size()) {

    // }

    
    // TODO -- Padding in time domain then trim
    //      -- Later down the line overlap/add for extended readout

    //FFT on requested dim
    tensor_clone = torch::fft::rfft(tensor_clone, std::nullopt, m_dimension);

    //Get the Wire filter -- already FFT'd
    auto spectrum_tensor = m_base_spectrum->spectrum({
        tensor_clone.sizes()[m_dimension]
    });

    log->debug("Spectrum size: {}", spectrum_tensor.sizes()[0]);
    log->debug("RFFt'd size: {}", tensor_clone.sizes()[m_dimension]);

    //Reshape input tensor for multiplication, do the filter multiplication
    tensor_clone = (tensor_clone.swapaxes(-1, m_dimension) * spectrum_tensor);
    log->debug("Swapped & mult'd");
    
    //Then reshape it back
    tensor_clone = tensor_clone.swapaxes(-1, m_dimension);

    //Inverse FFT in both dimensions
    log->debug("IFFTing");
    tensor_clone = torch::fft::irfft(tensor_clone, std::nullopt, m_dimension);
    log->debug("Done");

    // TODO: set md
    Configuration set_md;

    //Clone the tensor to take ownership of the memory and put into 
    //output 
    std::vector<ITorchTensor::pointer> itv{
        std::make_shared<SimpleTorchTensor>(tensor_clone)
    };
    out = std::make_shared<SimpleTorchTensorSet>(
        in->ident(), set_md,
        std::make_shared<std::vector<ITorchTensor::pointer>>(itv)
    );

    return true;
}