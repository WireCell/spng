/** FILL ME OUT
 */

 #ifndef WIRECELLSPNG_ITORCHCOLDELECRESPONSE
 #define WIRECELLSPNG_ITORCHCOLDELECRESPONSE
 
 #include "WireCellUtil/IComponent.h"
 #include <torch/torch.h>

 namespace WireCell {
 
     class ITorchColdElecResponse : public IComponent<ITorchColdElecResponse> {
        public:
         virtual ~ITorchColdElecResponse();
 
        /// Return the coldelec response data
        virtual torch::Tensor coldelec_response() const = 0;
     };
 
 }  // namespace WireCell
 
 #endif