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

    m_coarse_time_offset = get(config, "coarse_time_offset", m_coarse_time_offset);
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
    auto sizes = tensor_clone.sizes();
    std::vector<int64_t> shape;
    for (const auto & s : sizes) {
        shape.push_back(s);
    }
    //FFT on time dim
    tensor_clone = torch::fft::rfft(tensor_clone, std::nullopt, 1);

    //FFT on chan dim
    tensor_clone = torch::fft::fft(tensor_clone, std::nullopt, 0);

    //Get the Field x Elec. Response and do FFT in both dimensons
    auto frer_spectrum_tensor = base_frer_spectrum->spectrum(shape).clone();
    frer_spectrum_tensor = torch::fft::rfft2(frer_spectrum_tensor);

    //Get the wire shift
    int wire_shift = base_frer_spectrum->shifts()[0];

    //Apply to input data
    tensor_clone = tensor_clone / frer_spectrum_tensor;

    //Get the Wire filter -- already FFT'd
    auto wire_filter_tensor = base_wire_filter->spectrum({shape[0]});

    //Multiply along the wire dimension
    tensor_clone = tensor_clone * wire_filter_tensor.view({-1,1});

    //Inverse FFT in both dimensions
    tensor_clone = torch::fft::irfft2(tensor_clone);

    //Shift along wire dimension
    tensor_clone = tensor_clone.roll(wire_shift, 0);

    //Shift along time dimension
    int time_shift = (int) (
        (m_coarse_time_offset + base_frer_spectrum->shifts()[1]) /
        in->metadata()["period"].asDouble()
    );
    tensor_clone = tensor_clone.roll(time_shift, 1);


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