#ifndef WIRECELL_SPNGSIGPROC
#define WIRECELL_SPNGSIGPROC

#include "WireCellAux/Logger.h"

#include "WireCellSpng/ITorchTensorFilter.h"
#include "WireCellIface/IConfigurable.h"



namespace WireCell {
namespace SPNG {
    class SigProc : public Aux::Logger,
                        public WireCell::ITorchTensorFilter, public WireCell::IConfigurable {
    public:
        SigProc( );
        virtual ~SigProc();

        virtual bool operator()(const input_pointer& in, output_pointer& out);
        virtual void configure(const WireCell::Configuration& config) {};
        virtual WireCell::Configuration default_configuration() const {
            Configuration cfg;
            return cfg;
        };
    };
}
}

#endif