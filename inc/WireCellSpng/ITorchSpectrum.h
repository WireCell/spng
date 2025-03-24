/** FILL ME OUT
 */

 #ifndef WIRECELLSPNG_ITORCHSPECTRUM
 #define WIRECELLSPNG_ITORCHSPECTRUM
 
 #include "WireCellUtil/IComponent.h"
 #include <torch/torch.h>

 namespace WireCell {
 
    class ITorchSpectrum : public IComponent<ITorchSpectrum> {
    public:
        virtual ~ITorchSpectrum();
 
        /// Return the coldelec response data
        virtual torch::Tensor spectrum() const = 0;

        /// Get the base shape of the response
        virtual const std::vector<int64_t> & shape() const {return m_shape;};
    protected:
        std::vector<int64_t> m_shape;
    };
 
 }  // namespace WireCell
 
 #endif