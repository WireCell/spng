#ifndef WIRECELL_SPNG_TORCHTENSORHANDLE
#define WIRECELL_SPNG_TORCHTENSORHANDLE

#include <torch/torch.h>

namespace WireCell {
  class TorchTensorHandle {
   public:

     TorchTensorHandle(torch::Tensor & input_tensor);
     
     torch::Tensor clone_tensor() const;

   private:

     torch::Tensor m_tensor;
  };
}

#endif