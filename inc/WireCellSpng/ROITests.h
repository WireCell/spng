#ifndef WIRECELL_SPNGROITests
#define WIRECELL_SPNGROITests

#include "WireCellAux/Logger.h"

#include "WireCellSpng/ITorchTensorSetFilter.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellSpng/ITorchSpectrum.h"

namespace WireCell {
    namespace SPNG {
        class ROITests : public Aux::Logger,
                      public WireCell::ITorchTensorSetFilter, public WireCell::IConfigurable {
        public:
            ROITests( );
            virtual ~ROITests();

            virtual bool operator()(const input_pointer& in, output_pointer& out);
            virtual void configure(const WireCell::Configuration& config);
            virtual WireCell::Configuration default_configuration() const {
                Configuration cfg;
                return cfg;
            };
            virtual void finalize();
        private:
            std::string m_frer_spectrum{"FRERSpectrum"};
            std::string m_wire_filter{"Torch1DSpectrum"};
            std::shared_ptr<ITorchSpectrum> base_frer_spectrum, base_wire_filter;
            int m_coarse_time_offset = 0;
        };
    }
}
#endif