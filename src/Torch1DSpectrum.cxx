#include "WireCellSpng/Torch1DSpectrum.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
// #include <torch/torch.h>

WIRECELL_FACTORY(Torch1DSpectrum, WireCell::SPNG::Torch1DSpectrum, WireCell::ITorchSpectrum, WireCell::IConfigurable)

using namespace WireCell;

SPNG::Torch1DSpectrum::Torch1DSpectrum() : Aux::Logger("Torch1DSpectrum", "spng") {}

SPNG::Torch1DSpectrum::~Torch1DSpectrum() {}

WireCell::Configuration SPNG::Torch1DSpectrum::default_configuration() const
{
    Configuration cfg;
    cfg["default_length"] = m_default_length;
    return cfg;
}

void SPNG::Torch1DSpectrum::configure(const WireCell::Configuration& cfg)
{

    m_default_length = get(cfg, "default_length", m_default_length);
    m_total_spectrum = torch::ones(m_default_length);
    auto accessor = m_total_spectrum.accessor<float,1>();

    if (cfg.isMember("spectra")) {
        m_spectra.clear();
        m_spectra_tns.clear();
        for (auto tn: cfg["spectra"]) {
            m_spectra_tns.push_back(tn.asString());
            m_spectra.push_back(Factory::find_tn<IFilterWaveform>(tn.asString()));
            log->debug("Adding {} to list of spectra", m_spectra_tns.back());

            auto vals = m_spectra.back()->filter_waveform(m_default_length);
            for (size_t i = 0; i != vals.size(); i++) {
                accessor[i] *= vals.at(i);
            }
        }
    }

}

torch::Tensor SPNG::Torch1DSpectrum::spectrum() const { return m_total_spectrum; }

torch::Tensor SPNG::Torch1DSpectrum::spectrum(const std::vector<int64_t> & shape) {

    //TODO -- add in throw if not 1D shape
    m_total_spectrum = torch::ones(shape);
    auto accessor = m_total_spectrum.accessor<float,1>();

    for (const auto & spectrum : m_spectra) {
        auto vals = spectrum->filter_waveform(shape[0]);
        for (size_t i = 0; i != vals.size(); i++) {
            accessor[i] *= vals.at(i);
        }
    }

    return m_total_spectrum;
}