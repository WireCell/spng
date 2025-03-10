/** This component provides coldelec response data as read in from a "WCT
 * coldelec response" JSON file */

 #ifndef WIRECELLSPNG_TORCHCOLDELECRESPONSE
 #define WIRECELLSPNG_TORCHCOLDELECRESPONSE
 #include "WireCellAux/Logger.h"
 #include "WireCellSpng/ITorchColdElecResponse.h"
 #include "WireCellIface/IConfigurable.h"
 #include "WireCellUtil/Units.h"
 
 namespace WireCell {
     namespace SPNG {
         class TorchColdElecResponse : public Aux::Logger, 
                                    public ITorchColdElecResponse,
                                    public IConfigurable {
            public:
                // Create directly with the JSON data file or delay that
                // for configuration.
                TorchColdElecResponse();

                virtual ~TorchColdElecResponse();

                // ITorchColdElecResponse
                virtual torch::Tensor coldelec_response() const;

                // IConfigurable
                virtual void configure(const WireCell::Configuration& config);
                virtual WireCell::Configuration default_configuration() const;
 
            private:
                torch::Tensor m_elec_response;
                std::string m_coldelec_response{"ColdElecResponse"};
                bool m_do_fft = false;
                float m_extra_scale = 1.;
                float m_tick_period = 0.;
                int m_nticks = 0;
         };
 
     }  // namespace spng
 
 }  // namespace WireCell
 #endif