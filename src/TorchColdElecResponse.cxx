#include "WireCellSpng/TorchColdElecResponse.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
// #include "WireCellUtil/Waveform.h"
#include "WireCellIface/IWaveform.h"
// #include <torch/torch.h>

WIRECELL_FACTORY(TorchColdElecResponse, WireCell::SPNG::TorchColdElecResponse, WireCell::ITorchColdElecResponse, WireCell::IConfigurable)

using namespace WireCell;

SPNG::TorchColdElecResponse::TorchColdElecResponse() : Aux::Logger("TorchColdElecResponse", "spng") {}

SPNG::TorchColdElecResponse::~TorchColdElecResponse() {}

WireCell::Configuration SPNG::TorchColdElecResponse::default_configuration() const
{
    Configuration cfg;
    cfg["coldelec_response"] = m_coldelec_response;
    cfg["extra_scale"] = m_extra_scale;
    cfg["do_fft"] = m_do_fft;
    cfg["nticks"] = m_nticks;
    cfg["tick_period"] = m_tick_period;
    return cfg;
}

void SPNG::TorchColdElecResponse::configure(const WireCell::Configuration& cfg)
{
    m_coldelec_response = get(cfg, "coldelec_response", m_coldelec_response);
    m_extra_scale = get(cfg, "extra_scale", m_extra_scale);
    m_do_fft = get(cfg, "do_fft", m_do_fft);
    m_nticks = get(cfg, "nticks", m_nticks);
    m_tick_period = get(cfg, "tick_period", m_tick_period);

    auto ier = Factory::find_tn<IWaveform>(m_coldelec_response);

    //Get the waveform from the electronics response
    WireCell::Binning tbins(m_nticks, 0, m_nticks * m_tick_period);
    auto ewave = ier->waveform_samples(tbins);
    m_elec_response = torch::zeros({m_nticks});

    auto accessor = m_elec_response.accessor<float,1>();

    for (int i = 0; i < m_nticks; ++i) {
        accessor[i] = ewave[i];
    }

    m_elec_response = m_elec_response*m_extra_scale;

    if (m_do_fft) {
        log->debug("Performing FFT");
        m_elec_response = torch::fft::fft(m_elec_response);
    }

}

torch::Tensor SPNG::TorchColdElecResponse::coldelec_response() const { return m_elec_response; }