/** An interface to a "forward" operator on a Torch set */

#ifndef WIRECELL_ITorchFORWARD
#define WIRECELL_ITorchFORWARD

#include "WireCellUtil/IComponent.h"
#include "WireCellSpng/ITorchSet.h"

namespace WireCell::SPNG{
    class ITorchForward : public IFunctionNode<ITorchTensorSet,ITorchTensorSet> {
      public:
        virtual ~ITorchForward() {};

        virtual std::string signature() { return typeid(ITorchForward).name(); }
      };
}  // namespace WireCell

#endif  // WIRECELL_ITorchFORWARD
