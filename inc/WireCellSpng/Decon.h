#ifndef WIRECELL_SPNGDECON
#define WIRECELL_SPNGDECON

#include "WireCellAux/Logger.h"

#include "WireCellSpng/ITorchTensorSetFilter.h"
#include "WireCellIface/IConfigurable.h"



namespace WireCell {
namespace SPNG {
    class Decon : public Aux::Logger,
                    public WireCell::ITorchTensorSetFilter, public WireCell::IConfigurable {
    public:
        Decon( );
        virtual ~Decon();

        virtual bool operator()(const input_pointer& in, output_pointer& out);
        virtual void configure(const WireCell::Configuration& config);
        virtual WireCell::Configuration default_configuration() const {
            Configuration cfg;
            return cfg;
        };
    private:
        std::string m_field_response{"FieldResponse"};
    };
}
}

#endif