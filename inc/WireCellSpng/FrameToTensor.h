#ifndef WIRECELL_SPNG_FRAMETOTENSOR
#define WIRECELL_SPNG_FRAMETOTENSOR

#include "WireCellSpng/IFrameToTensor.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellUtil/TagRules.h"
#include "WireCellAux/Logger.h"

namespace WireCell {
    namespace SPNG {

        /// Convert 1 frame to 1 TorchTensor
        /// TODO -- Flesh out description
        class FrameToTensor : public Aux::Logger,
                              public IFrameToTensor, public IConfigurable {
           public:
            FrameToTensor()
             : Aux::Logger("FrameToTensor", "spng") {};
            virtual ~FrameToTensor() {};

            // INode, override because we get multiplicity at run time.
            // virtual std::vector<std::string> output_types() {return };

            // IFanout
            virtual bool operator()(const input_pointer& in, output_pointer& out);

            // IConfigurable
            virtual void configure(const WireCell::Configuration& cfg) {};
            virtual WireCell::Configuration default_configuration() const;

           private:

        };
    }  // namespace SPNG
}  // namespace WireCell

#endif