/** This component provides field response data as read in from a "WCT
 * field response" JSON file */

 #ifndef WIRECELLSPNG_TORCHFIELDRESPONSE
 #define WIRECELLSPNG_TORCHFIELDRESPONSE
 #include "WireCellAux/Logger.h"
 #include "WireCellSpng/ITorchFieldResponse.h"
 #include "WireCellIface/IConfigurable.h"
 #include "WireCellUtil/Units.h"
 
 namespace WireCell {
     namespace SPNG {
         class TorchFieldResponse : public Aux::Logger, 
                                    public ITorchFieldResponse,
                                    public IConfigurable {
            public:
             // Create directly with the JSON data file or delay that
             // for configuration.
             TorchFieldResponse();
 
             virtual ~TorchFieldResponse();
 
             // ITorchFieldResponse
             virtual torch::Tensor field_response() const;
 
             // IConfigurable
             virtual void configure(const WireCell::Configuration& config);
             virtual WireCell::Configuration default_configuration() const;
 
            private:
             WireCell::Configuration m_base_fr;
             torch::Tensor m_fr;
             std::string m_field_response{"FieldResponse"};
             int m_plane_id = 0;
         };
 
     }  // namespace spng
 
 }  // namespace WireCell
 #endif