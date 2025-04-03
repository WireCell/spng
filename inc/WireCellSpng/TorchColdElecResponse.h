/** This component provides coldelec response data as read in from a "WCT
 * coldelec response" JSON file */

 #ifndef WIRECELLSPNG_TORCHCOLDELECRESPONSE
 #define WIRECELLSPNG_TORCHCOLDELECRESPONSE
 #include "WireCellAux/Logger.h"
 #include "WireCellSpng/ITorchSpectrum.h"
 #include "WireCellIface/IConfigurable.h"
 #include "WireCellUtil/Units.h"
 
 namespace WireCell {
     namespace SPNG {
         class TorchColdElecResponse : public Aux::Logger, 
                                       public ITorchSpectrum,
                                       public IConfigurable {
            public:
                // Create directly with the JSON data file or delay that
                // for configuration.
                TorchColdElecResponse();

                virtual ~TorchColdElecResponse();

                // ITorchSpectrum
                virtual torch::Tensor spectrum() const;
                virtual torch::Tensor spectrum(const std::vector<int64_t> & shape) {
                    return torch::zeros(shape);
                };

                // virtual std::vector<int64_t> shape() constl

                // IConfigurable
                virtual void configure(const WireCell::Configuration& config);
                virtual WireCell::Configuration default_configuration() const;
 
            private:
                torch::Tensor m_elec_response;
                std::string m_coldelec_response{"ColdElecResponse"};
                bool m_do_fft = false;
                float m_extra_scale = 1.;
                float m_tick_period = 0.;
                double m_gain = 0.;
                double m_shaping = 0.;

                int m_nticks = 0;
         };
 
     }  // namespace spng
 
 }  // namespace WireCell
 #endif