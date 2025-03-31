#include "WireCellSpng/TorchFRERSpectrum.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellIface/IFieldResponse.h"
#include "WireCellIface/IWaveform.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/FFTBestLength.h"
// #include <torch/torch.h>

WIRECELL_FACTORY(TorchFRERSpectrum, WireCell::SPNG::TorchFRERSpectrum, WireCell::ITorchSpectrum, WireCell::IConfigurable)

using namespace WireCell;

SPNG::TorchFRERSpectrum::TorchFRERSpectrum() : Aux::Logger("TorchFRERSpectrum", "spng"), m_cache(5) {}

SPNG::TorchFRERSpectrum::~TorchFRERSpectrum() {}

WireCell::Configuration SPNG::TorchFRERSpectrum::default_configuration() const
{
    Configuration cfg;
    cfg["elec_response"] = m_elec_response_name;
    cfg["extra_scale"] = m_extra_scale;
    // cfg["do_fft"] = m_do_fft;
    cfg["default_nticks"] = m_default_nticks;
    cfg["default_period"] = m_default_period;
    cfg["default_nchans"] = m_default_nchans;
    // cfg["tick_period"] = m_tick_period;
    // cfg["shaping"] = m_shaping;
    // cfg["gain"] = m_gain;
    cfg["inter_gain"] = m_inter_gain;
    cfg["ADC_mV"] = m_ADC_mV;

    cfg["field_response"] = m_field_response_name;
    cfg["fr_plane_id"] = m_plane_id;
    // cfg["do_fft"] = m_do_fft;
    // cfg["do_average"] = m_do_average;

    return cfg;
}

void SPNG::TorchFRERSpectrum::configure(const WireCell::Configuration& cfg)
{
    m_field_response_name = get(cfg, "field_response", m_field_response_name);
    m_elec_response_name = get(cfg, "elec_response", m_elec_response_name);
    m_plane_id = get(cfg, "fr_plane_id", m_plane_id);
    m_extra_scale = get(cfg, "extra_scale", m_extra_scale);

    m_default_nticks = get(cfg, "default_nticks", m_default_nticks);
    m_default_nchans = get(cfg, "default_nchans", m_default_nchans);
    
    m_inter_gain = get(cfg, "inter_gain", m_inter_gain);
    m_ADC_mV = get(cfg, "ADC_mV", m_ADC_mV);
    m_shape = {m_default_nchans, m_default_nticks};


    m_field_response = Factory::find_tn<IFieldResponse>(m_field_response_name)->field_response();
    m_field_response_avg = Response::wire_region_average(
        m_field_response
    );



    m_elec_response = Factory::find_tn<IWaveform>(m_elec_response_name);
    torch::Tensor elec_response_tensor = torch::zeros(m_fravg_nticks);
    WireCell::Binning tbins(m_fravg_nticks, 0, m_fravg_nticks*m_fravg_period);
    auto ewave = m_elec_response->waveform_samples(tbins);
    auto accessor = elec_response_tensor.accessor<float,1>();
    for (int i = 0; i < m_fravg_nticks; ++i) {
        accessor[i] = ewave[i];
    }
    elec_response_tensor *= m_inter_gain * m_ADC_mV * (-1);
    elec_response_tensor = torch::fft::fft(elec_response_tensor);

    bool found_plane = false;
    for (auto & plane : m_field_response_avg.planes) {
        if (plane.planeid != m_plane_id) continue;
        
        found_plane = true;
        //Metadata of avg FR
        m_fravg_nticks = fft_best_length(
            plane.paths[0].current.size()
        );
        m_fravg_period = m_field_response_avg.period;
        m_fravg_nchans = plane.paths.size();
        if (m_fravg_nchans == 0) {
            THROW(ValueError() <<
                errmsg{String::format("TorchFRERSpectrum::%s: ", m_field_response_name)} <<
                errmsg{"Got 0 nrows (electron paths)"});
        }

        if (m_fravg_nticks == 0) {
            THROW(ValueError() <<
                errmsg{String::format("TorchFRERSpectrum::%s: ", m_field_response_name)} <<
                errmsg{"Got 0 ncols"});
        }

        m_shape = {m_fravg_nchans, m_fravg_nticks};

        m_total_response = torch::zeros(m_shape);
        log->debug("Got {} {}", m_fravg_nchans, m_fravg_nticks);
        auto accessor = m_total_response.accessor<float,2>();

        for (int irow = 0; irow < m_fravg_nchans; ++irow) {
            auto& path = plane.paths[irow];
            for (int icol = 0; icol < m_fravg_nticks; ++icol) {
                accessor[irow][icol] = path.current[icol];
            }
        }

        m_total_response = torch::fft::fft(m_total_response);
    }
    if (!found_plane) {
        THROW(ValueError() <<
            errmsg{String::format("TorchFRERSpectrum::%s: ", m_field_response_name)} <<
            errmsg{String::format("Could not find plane %d", m_plane_id)});
    }

    m_total_response *= elec_response_tensor * m_fravg_period;
    auto total_response_accessor = m_total_response.accessor<float,2>();
    //Redigitize according to default nchans, nticks.
    m_applied_response = torch::zeros({m_default_nchans, m_default_nticks});
    auto applied_response_accessor = m_applied_response.accessor<float,2>();
    for (int irow = 0; irow < m_fravg_nchans; ++irow) {
        int fcount = 1;
        for (int i = 0; i < m_default_nticks; i++) {
            float ctime = m_default_period*i;

            if (fcount < m_fravg_nticks)
                while (ctime > fcount*m_fravg_period) {
                    fcount++;
                    if (fcount >= m_fravg_nticks) break;
                }

            if (fcount < m_fravg_nticks) {
                applied_response_accessor[irow][i] = ((ctime - m_fravg_period*(fcount - 1)) / m_fravg_period * total_response_accessor[irow][fcount - 1] +
                             (m_fravg_period*fcount - ctime) / m_fravg_period * total_response_accessor[irow][fcount]);  // / (-1);
            }
            else {
                applied_response_accessor[irow][i] = 0;
            }
        }
    }
}

torch::Tensor SPNG::TorchFRERSpectrum::spectrum() const { return m_applied_response; }