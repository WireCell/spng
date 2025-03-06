#include "WireCellSpng/TorchFieldResponse.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellIface/IFieldResponse.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
// #include <torch/torch.h>

WIRECELL_FACTORY(TorchFieldResponse, WireCell::SPNG::TorchFieldResponse, WireCell::ITorchFieldResponse, WireCell::IConfigurable)

using namespace WireCell;

SPNG::TorchFieldResponse::TorchFieldResponse() : Aux::Logger("TorchFieldResponse", "spng") {}

SPNG::TorchFieldResponse::~TorchFieldResponse() {}

WireCell::Configuration SPNG::TorchFieldResponse::default_configuration() const
{
    Configuration cfg;
    cfg["field_response"] = m_field_response;
    cfg["fr_plane_id"] = m_plane_id;
    cfg["do_fft"] = m_do_fft;
    cfg["do_average"] = m_do_average;
    return cfg;
}

void SPNG::TorchFieldResponse::configure(const WireCell::Configuration& cfg)
{
    m_field_response = get(cfg, "field_response", m_field_response);
    m_plane_id = get(cfg, "fr_plane_id", m_plane_id);
    m_do_fft = get(cfg, "do_fft", m_do_fft);
    m_do_average = get(cfg, "do_average", m_do_average);


    auto ifr = Factory::find_tn<IFieldResponse>(m_field_response);
    auto the_response = (
        m_do_average ?
        Response::wire_region_average(ifr->field_response()) :
        ifr->field_response()
    );

    bool found_plane = false;
    for (auto & plane : the_response.planes) {
        if (plane.planeid != m_plane_id) continue;
        
        found_plane = true;
        
        int nrows = plane.paths.size();
        if (nrows == 0) {
            THROW(ValueError() <<
                errmsg{String::format("TorchFieldResponse::%s: ", m_field_response)} <<
                errmsg{"Got 0 nrows (electron paths)"});
        }

        int ncols = plane.paths[0].current.size();
        if (ncols == 0) {
            THROW(ValueError() <<
                errmsg{String::format("TorchFieldResponse::%s: ", m_field_response)} <<
                errmsg{"Got 0 ncols"});
        }

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
            errmsg{String::format("TorchFieldResponse::%s: ", m_field_response)} <<
            errmsg{String::format("Could not find plane %d", m_plane_id)});
    }

}

torch::Tensor SPNG::TorchFieldResponse::field_response() const { return m_fr; }