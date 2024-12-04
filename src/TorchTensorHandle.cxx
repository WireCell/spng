#include "WireCellSpng/TorchTensorHandle.h"


namespace WireCell {

TorchTensorHandle::TorchTensorHandle(torch::Tensor & input_tensor)
  : m_tensor(input_tensor.detach()) {
  //Turn off comp. graph 
  m_tensor.requires_grad_(false);
}

TorchTensorHandle::~TorchTensorHandle() {
    //We don't have the pointer filled if created from a tensor,
    //but if we do have one free up the memory in the correct location
    if (m_data != nullptr) {
        if (m_tensor.device() == torch::kCPU) free(m_data);
        else if (m_tensor.device() == torch::kCUDA) cudaFree(m_data);
    }
}

//Return a clone of the tensor to keep the handle data const correct
torch::Tensor TorchTensorHandle::clone_tensor() const {
    return m_tensor.clone();
}


//Return dtypes -- TODO flesh this out with all other dtypes (12 in total)
///see https://pytorch.org/docs/stable/tensor_attributes.html
torch::Dtype TorchTensorHandle::get_dtype(const std::vector<float> & input) {
  return at::kFloat;
}
torch::Dtype TorchTensorHandle::get_dtype(const std::vector<int> & input) {
  return at::kInt;
}

}