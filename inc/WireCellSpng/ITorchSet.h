#ifndef WIRECELL_ITorchSET
#define WIRECELL_ITorchSET

#include "WireCellSpng/ITorch.h"
#include "WireCellUtil/Configuration.h"


namespace WireCell {

    class ITorchSet : public IData<ITorchSet> {
       public:
        virtual ~ITorchSet() {}

        /// Return some identifier number that is unique to this set.
        virtual int ident() const = 0;

        /// Optional metadata associated with the set of Torchs
        virtual Configuration metadata() const { return Configuration(); }

        /// Return the Torchs in this set.
        virtual ITorch::shared_vector Torchs() const = 0;
    };
}  // namespace WireCell

#endif
