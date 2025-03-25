#include "WireCellSpng/TorchFRERSpectrum.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellIface/IFieldResponse.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
// #include <torch/torch.h>

WIRECELL_FACTORY(TorchFRERSpectrum, WireCell::SPNG::TorchFRERSpectrum, WireCell::ITorchSpectrum, WireCell::IConfigurable)

using namespace WireCell;

SPNG::TorchFRERSpectrum::TorchFRERSpectrum() : Aux::Logger("TorchFRERSpectrum", "spng") {}

SPNG::TorchFRERSpectrum::~TorchFRERSpectrum() {}

WireCell::Configuration SPNG::TorchFRERSpectrum::default_configuration() const
{
    Configuration cfg;
    cfg["coldelec_response"] = m_coldelec_response;
    cfg["extra_scale"] = m_extra_scale;
    cfg["do_fft"] = m_do_fft;
    cfg["default_nticks"] = m_default_nticks;
    cfg["default_nchans"] = m_default_nchans;
    cfg["tick_period"] = m_tick_period;
    cfg["shaping"] = m_shaping;
    cfg["gain"] = m_gain;

    cfg["field_response"] = m_field_response;
    cfg["fr_plane_id"] = m_plane_id;
    cfg["do_fft"] = m_do_fft;
    cfg["do_average"] = m_do_average;

    return cfg;
}

void SPNG::TorchFRERSpectrum::configure(const WireCell::Configuration& cfg)
{
    m_field_response = get(cfg, "field_response", m_field_response);
    m_plane_id = get(cfg, "fr_plane_id", m_plane_id);
    m_do_fft = get(cfg, "do_fft", m_do_fft);
    m_do_average = get(cfg, "do_average", m_do_average);
    m_coldelec_response = get(cfg, "coldelec_response", m_coldelec_response);
    m_extra_scale = get(cfg, "extra_scale", m_extra_scale);
    m_do_fft = get(cfg, "do_fft", m_do_fft);
    m_default_nticks = get(cfg, "default_nticks", m_default_nticks);
    m_default_nchans = get(cfg, "default_nchans", m_default_nchans);
    m_tick_period = get(cfg, "tick_period", m_tick_period);
    m_gain = get(cfg, "gain", m_gain);
    m_shaping = get(cfg, "shaping", m_shaping);
    m_shape = {m_default_nchans, m_default_nticks};


    auto ifr = Factory::find_tn<IFieldResponse>(m_field_response);
    auto the_response = (
        m_do_average ?
        Response::wire_region_average(ifr->field_response()) :
        ifr->field_response()
    );

/* TODO add in the corresponding Cold Elec */

    bool found_plane = false;
    for (auto & plane : the_response.planes) {
        if (plane.planeid != m_plane_id) continue;
        
        found_plane = true;
        
        int64_t nrows = plane.paths.size();
        if (nrows == 0) {
            THROW(ValueError() <<
                errmsg{String::format("TorchFRERSpectrum::%s: ", m_field_response)} <<
                errmsg{"Got 0 nrows (electron paths)"});
        }

        int64_t ncols = plane.paths[0].current.size();
        if (ncols == 0) {
            THROW(ValueError() <<
                errmsg{String::format("TorchFRERSpectrum::%s: ", m_field_response)} <<
                errmsg{"Got 0 ncols"});
        }

        m_shape = {nrows, ncols};

        m_fr = torch::zeros({nrows, ncols});
        log->debug("Got {} {}", nrows, ncols);
        auto accessor = m_fr.accessor<float,2>();

        for (int irow = 0; irow < nrows; ++irow) {
            auto& path = plane.paths[irow];
            for (int icol = 0; icol < ncols; ++icol) {
                accessor[irow][icol] = path.current[icol];
            }
        }

        if (m_do_fft) {
            log->debug("Performing FFT");
            m_fr = torch::fft::fft(m_fr);
        }
        break;
    }
    if (!found_plane) {
        THROW(ValueError() <<
            errmsg{String::format("TorchFRERSpectrum::%s: ", m_field_response)} <<
            errmsg{String::format("Could not find plane %d", m_plane_id)});
    }

}

torch::Tensor SPNG::TorchFRERSpectrum::spectrum() const { return m_fr; }