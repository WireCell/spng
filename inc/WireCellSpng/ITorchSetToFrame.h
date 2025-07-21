#ifndef WIRECELL_SPNG_ITORCHSETTOFRAME
#define WIRECELL_SPNG_ITORCHSETTOFRAME

#include "WireCellIface/IFunctionNode.h"
#include "WireCellIface/IFrame.h"
#include "WireCellSpng/ITorchTensorSet.h"

namespace WireCell {
namespace SPNG {
    /** A frame fan-out component takes 1 input frame and produces one
     * TorchTensorSet.

     */
    class ITorchSetToFrame : public IFunctionNode<ITorchTensorSet, IFrame> {
       public:
        virtual ~ITorchSetToFrame() {};

        virtual std::string signature() { return typeid(ITorchSetToFrame).name(); }

        // and the already abstract:
        // virtual bool operator()(const input_pointer& in, output_vector& outv);
    };
}  // namespace SPNG
}  // namespace WireCell

#endif