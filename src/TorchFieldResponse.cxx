#include "WireCellSpng/TorchFieldResponse.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellIface/IFieldResponse.h"
#include "WireCellUtil/NamedFactory.h"

WIRECELL_FACTORY(TorchFieldResponse, WireCell::SPNG::TorchFieldResponse, WireCell::ITorchFieldResponse, WireCell::IConfigurable)

using namespace WireCell;

SPNG::TorchFieldResponse::TorchFieldResponse() : Aux::Logger("TorchFieldResponse", "spng") {}

SPNG::TorchFieldResponse::~TorchFieldResponse() {}

WireCell::Configuration SPNG::TorchFieldResponse::default_configuration() const
{
    Configuration cfg;
    cfg["field_response"] = m_field_response;
    cfg["fr_plane_id"] = m_plane_id;
    return cfg;
}

void SPNG::TorchFieldResponse::configure(const WireCell::Configuration& cfg)
{
    m_field_response = get(cfg, "field_response", m_field_response);
    auto base_response = Factory::find_tn<IFieldResponse>(m_field_response);

    // bool found_plane = false;

    for (auto & plane : base_response->field_response().planes) {
        if (plane.planeid != m_plane_id) continue;
        
        // found_plane = true;
        
        int nrows = plane.paths.size();
        int ncols = plane.paths[0].current.size();
        // m_frs.push_back(torch::zeros({nrows, ncols}));
        m_fr = torch::zeros({nrows, ncols});
        log->debug("Got {} {}", nrows, ncols);
        auto accessor = m_fr.accessor<float,2>();

        for (int irow = 0; irow < nrows; ++irow) {
            auto& path = plane.paths[irow];
            for (int icol = 0; icol < ncols; ++icol) {
                accessor[irow][icol] = path.current[icol];
            }
        }
        // if (m_do_average) {
        // }
        // if (m_do_fft) {
        // }
    }

    // log->debug(m_fr);

    // if (!found_plane) {
    //     //Throw if not found
    // }
}

torch::Tensor SPNG::TorchFieldResponse::field_response() const { return m_fr; }