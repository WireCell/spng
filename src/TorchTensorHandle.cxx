#include "WireCellSpng/TorchTensorHandle.h"

namespace WireCell {

TorchTensorHandle::TorchTensorHandle(torch::Tensor & input_tensor)
  : m_tensor(input_tensor.detach()) {
  m_tensor.requires_grad_(false); //Turn off comp. graph.
}

torch::Tensor TorchTensorHandle::clone_tensor() const {
    return m_tensor.clone();
}

}