#ifndef WIRECELL_SPNG_FRAMETOTORCHFANOUT
#define WIRECELL_SPNG_FRAMETOTORCHFANOUT

#include "WireCellSpng/IFrameToTorchFanout.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellUtil/Logging.h"
#include "WireCellAux/Logger.h"

namespace WireCell {
    namespace SPNG {

        // Fan out 1 frame to N set at construction or configuration time.
        class FrameToTorchFanout
            : public Aux::Logger,
              public IFrameToTorchFanout, public IConfigurable {
           public:
            FrameToTorchFanout();
            virtual ~FrameToTorchFanout() {};

            // INode, override because we get multiplicity at run time.
            virtual std::vector<std::string> output_types();

            // IFanout
            virtual bool operator()(const input_pointer& in, output_vector& outv);

            // IConfigurable
            virtual void configure(const WireCell::Configuration& cfg);
            virtual WireCell::Configuration default_configuration() const;

           private:
            int m_multiplicity;

            //Wires per Plane per APA
            std::vector<int> m_planes;

            //Expected number of ticks in each readout frame
            int m_expected_nticks{-1};

            //TODO -- possibly add configuration
            //allowing for different behavior if receive unexpected ticks

            WireCell::Configuration m_cfg;
            // Log::logptr_t log;
        };
    }  // namespace Aux
}  // namespace WireCell

#endif