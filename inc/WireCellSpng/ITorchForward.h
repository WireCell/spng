/** An interface to a "forward" operator on a Torch set */

#ifndef WIRECELL_ITorchFORWARD
#define WIRECELL_ITorchFORWARD

#include "WireCellUtil/IComponent.h"
#include "WireCellIface/ITorchSet.h"

namespace WireCell::SPNG{
    class ITorchForward : public IComponent<ITorchForward> {
      public:
        virtual ~ITorchForward();

        virtual ITorchSet::pointer forward(const ITorchSet::pointer& input) const = 0;
    };
}  // namespace WireCell

#endif  // WIRECELL_ITorchFORWARD
