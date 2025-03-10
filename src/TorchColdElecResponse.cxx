#include "WireCellSpng/TorchColdElecResponse.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/Response.h"
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
    cfg["shaping"] = m_shaping;
    cfg["gain"] = m_gain;

    return cfg;
}

void SPNG::TorchColdElecResponse::configure(const WireCell::Configuration& cfg)
{
    m_coldelec_response = get(cfg, "coldelec_response", m_coldelec_response);
    m_extra_scale = get(cfg, "extra_scale", m_extra_scale);
    m_do_fft = get(cfg, "do_fft", m_do_fft);
    m_nticks = get(cfg, "nticks", m_nticks);
    m_tick_period = get(cfg, "tick_period", m_tick_period);
    m_gain = get(cfg, "gain", m_gain);
    m_shaping = get(cfg, "shaping", m_shaping);
    //TODO -- replace with 
    // std::cout << "Trying to get iwaveform " << m_coldelec_response << std::endl;
    // auto ier = Factory::find_tn<IWaveform>(m_coldelec_response);

    auto elec_resp_generator
        = std::make_unique<Response::ColdElec>(m_gain, m_shaping);

    //Get the waveform from the electronics response
    WireCell::Binning tbins(m_nticks, 0, m_nticks * m_tick_period);
    auto ewave = elec_resp_generator->generate(tbins);

    m_elec_response = torch::zeros({m_nticks});

    auto accessor = m_elec_response.accessor<float,1>();

    for (int i = 0; i < m_nticks; ++i) {
        accessor[i] = ewave[i];
        // accessor[i] = 1.;
    }

    m_elec_response = m_elec_response*m_extra_scale;

    if (m_do_fft) {
        log->debug("Performing FFT");
        m_elec_response = torch::fft::fft(m_elec_response);
    }
}

torch::Tensor SPNG::TorchColdElecResponse::coldelec_response() const { return m_elec_response; }