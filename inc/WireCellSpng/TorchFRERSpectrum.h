/** This component provides field response data as read in from a "WCT
 * field response" JSON file */

 #ifndef WIRECELLSPNG_TORCHFRERSPECTRUM
 #define WIRECELLSPNG_TORCHFRERSPECTRUM
 #include "WireCellAux/Logger.h"
 #include "WireCellSpng/ITorchSpectrum.h"
 #include "WireCellIface/IConfigurable.h"
 #include "WireCellUtil/Units.h"
 
 namespace WireCell {
     namespace SPNG {
         class TorchFRERSpectrum : public Aux::Logger, 
                                    public ITorchSpectrum,
                                    public IConfigurable {
            public:
             // Create directly with the JSON data file or delay that
             // for configuration.
             TorchFRERSpectrum();
 
             virtual ~TorchFRERSpectrum();
 
             // ITorchSpectrum
             virtual torch::Tensor spectrum() const;
 
            //  virtual std::vector<int64_t> shape() const;

             // IConfigurable
             virtual void configure(const WireCell::Configuration& config);
             virtual WireCell::Configuration default_configuration() const;
 
            private:
             torch::Tensor m_fr;
             std::string m_field_response{"FieldResponse"};
             std::string m_coldelec_response{"ColdElecResponse"};
             
             //Relevant for Field Response
             int m_plane_id = 0;
             bool m_do_average = false;


             //Relevant for Cold Elec Response
             float m_extra_scale = 1.;
             float m_tick_period = 0.;
             double m_gain = 0.;
             double m_shaping = 0.;
             int m_default_nticks = 0;
             int m_default_nchans = 0;


             bool m_do_fft = false;             
         };
 
     }  // namespace spng
 
 }  // namespace WireCell
 #endif