/** FILL ME OUT
 */

 #ifndef WIRECELLSPNG_ITORCHFIELDRESPONSE
 #define WIRECELLSPNG_ITORCHFIELDRESPONSE
 
 #include "WireCellUtil/IComponent.h"
 #include <torch/torch.h>

 namespace WireCell {
 
     class ITorchFieldResponse : public IComponent<ITorchFieldResponse> {
        public:
         virtual ~ITorchFieldResponse();
 
        /// Return the field response data
        virtual torch::Tensor field_response() const = 0;
     };
 
 }  // namespace WireCell
 
 #endif