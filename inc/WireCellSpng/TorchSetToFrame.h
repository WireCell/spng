#ifndef WIRECELL_SPNG_TORCHSETTOFRAME
#define WIRECELL_SPNG_TORCHSETTOFRAME

#include "WireCellSpng/ITorchSetToFrame.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellUtil/Logging.h"
#include "WireCellAux/Logger.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/WirePlaneId.h"

namespace WireCell {
    namespace SPNG {

        // Fan out 1 frame to N set at construction or configuration time.
        class TorchSetToFrame
            : public Aux::Logger,
              public ITorchSetToFrame, public IConfigurable {
        public:
            TorchSetToFrame();
            virtual ~TorchSetToFrame() {};

            // INode, override because we get multiplicity at run time.

            // IFanout
            virtual bool operator()(const input_pointer& in, output_pointer& out);

            // IConfigurable
            virtual void configure(const WireCell::Configuration& cfg);
            virtual WireCell::Configuration default_configuration() const;

        private:
            //Wire Planes to pack together into an output TorchTensorSet
            std::map<const WirePlaneId, int> m_input_groups;
            std::unordered_map<int, int> m_channel_map;

            //How many wires in the TorchTensor in the ith output
            std::map<int, int> m_input_nchannels;
            std::string m_anode_tn{"AnodePlane"};
            IAnodePlane::pointer m_anode;
        };
    }  // namespace Aux
}  // namespace WireCell

#endif