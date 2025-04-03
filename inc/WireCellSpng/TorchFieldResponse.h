/** This component provides field response data as read in from a "WCT
 * field response" JSON file */

 #ifndef WIRECELLSPNG_TORCHFIELDRESPONSE
 #define WIRECELLSPNG_TORCHFIELDRESPONSE
 #include "WireCellAux/Logger.h"
 #include "WireCellSpng/ITorchSpectrum.h"
 #include "WireCellIface/IConfigurable.h"
 #include "WireCellUtil/Units.h"
 
 namespace WireCell {
     namespace SPNG {
         class TorchFieldResponse : public Aux::Logger, 
                                    public ITorchSpectrum,
                                    public IConfigurable {
            public:
             // Create directly with the JSON data file or delay that
             // for configuration.
             TorchFieldResponse();
 
             virtual ~TorchFieldResponse();
 
             // ITorchSpectrum
             virtual torch::Tensor spectrum() const;
             virtual torch::Tensor spectrum(const std::vector<int64_t> & shape) {
                return torch::zeros(shape);
            };

            //  virtual std::vector<int64_t> shape() const;

             // IConfigurable
             virtual void configure(const WireCell::Configuration& config);
             virtual WireCell::Configuration default_configuration() const;
 
            private:
             torch::Tensor m_fr;
             std::string m_field_response{"FieldResponse"};
             int m_plane_id = 0;
             bool m_do_fft = false;
             bool m_do_average = false;
         };
 
     }  // namespace spng
 
 }  // namespace WireCell
 #endif