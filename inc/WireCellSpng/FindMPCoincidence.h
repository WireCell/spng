#ifndef WIRECELL_SPNGFINDMPCOINCIDENCE
#define WIRECELL_SPNGFINDMPCOINCIDENCE

#include "WireCellAux/Logger.h"

#include "WireCellSpng/ITorchTensorSetFilter.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellSpng/ITorchSpectrum.h"



namespace WireCell {
namespace SPNG {
    class FindMPCoincidence : public Aux::Logger,
                    public WireCell::ITorchTensorSetFilter, public WireCell::IConfigurable {
    public:
        FindMPCoincidence( );
        virtual ~FindMPCoincidence();

        virtual bool operator()(const input_pointer& in, output_pointer& out);
        virtual void configure(const WireCell::Configuration& config);
        virtual WireCell::Configuration default_configuration() const {
            Configuration cfg;
            return cfg;
        };
    private:
        int m_rebin_val{-1};
        int m_target_plane_index{0};
        int m_aux_plane_l_index{1};
        int m_aux_plane_m_index{2};
    };
}
}

#endif