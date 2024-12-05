#ifndef WIRECELL_SPNG_IFRAMETOTENSOR
#define WIRECELL_SPNG_IFRAMETOTENSOR

#include "WireCellIface/IFunctionNode.h"
#include "WireCellIface/IFrame.h"
#include "WireCellSpng/ITorchTensor.h"

namespace WireCell {
namespace SPNG {
    /** A frame fan-out component takes 1 input frame and produces one
     * TorchTensor.

     */
    class IFrameToTensor : public IFunctionNode<IFrame, ITorchTensor> {
       public:
        virtual ~IFrameToTensor() {};

        virtual std::string signature() { return typeid(IFrameToTensor).name(); }

        // Subclass must implement:
        // virtual std::vector<std::string> output_types() = 0;
        // and the already abstract:
        // virtual bool operator()(const input_pointer& in, output_vector& outv);
    };
}  // namespace SPNG
}  // namespace WireCell

#endif