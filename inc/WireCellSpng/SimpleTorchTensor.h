#ifndef WIRECELL_SPNG_SIMPLETORCHTENSOR
#define WIRECELL_SPNG_SIMPLETORCHTENSOR 
#include "WireCellSpng/ITorchTensor.h"
#include "WireCellUtil/Exceptions.h"
#include <stdexcept>
// #include <format>

namespace WireCell {
class SimpleTorchTensor: public ITorchTensor {
  public:
    SimpleTorchTensor(
      torch::Tensor tensor,
      const std::vector<SPNG::TensorKind> & kind,
      const std::vector<SPNG::TensorDomain> & domain,
      const std::vector<std::string> & batch_label = {}, //Allow empty in case batch not supplied within kind
      const Configuration& md = Json::nullValue
    )
    : m_tensor(tensor), m_md(md) {
      m_tensor.requires_grad_(false); //Turn off the computational graph
      
      if (kind.size() == 0) {
        THROW(ValueError() << errmsg{"Need to supply nonzero-size TensorKind vector"});
      }
      if (domain.size() == 0) {
        THROW(ValueError() << errmsg{"Need to supply nonzero-size TensorDomain vector"});
      }
      
      if (kind[0] == SPNG::kBatch && batch_label.size() == 0) {
        THROW(ValueError() << errmsg{"First dimension of TorchTensor is Batch -- need to supply batch label"});
      }
      if (kind[0] == SPNG::kBatch && domain[0] != SPNG::kNull) {
        THROW(ValueError() << errmsg{"Domain of batch dimension must be Null"});
      }

      if (kind.size() > 1) {
        for (size_t i = 1; i < kind.size(); ++i) {
          if (kind[i] == SPNG::kBatch) {
            THROW(ValueError() << errmsg{String::format("Kind 'Batch' found at unallowed position %d", i)});
          }
        }
      }
      
      if (kind.size() != domain.size()) {
        THROW(ValueError() << errmsg{"TorchTensor Kind and Domain sizes need to match"});
      }

      // if (kind.size() != tensor.sizes()[0])
        
      m_kind = kind;
      m_domain = domain;
      m_batch_label = batch_label;//Need to check that the batch label matches the shape of tensor in the first dimension
    }

    virtual torch::Tensor tensor() const { return m_tensor.detach().clone(); }
    virtual Configuration metadata() const { return m_md; }
    virtual std::string dtype() const { return torch::toString(m_tensor.dtype()); }
    virtual std::vector<int64_t> shape() const { return m_tensor.sizes().vec(); }
    virtual torch::Device device() const { return m_tensor.device(); }

    virtual const std::vector<SPNG::TensorKind> & kind() const { return m_kind; }
    virtual const std::vector<SPNG::TensorDomain> & domain() const { return m_domain; }
    virtual const std::vector<std::string> & batch_label() const { return m_batch_label; }


    // //Returns true if upgraded, false if not (which means batch was already first index)
    // bool make_batch() {
    //   // if (m_kind.size() > 0) {
    //   //   //If something has gone wrong (which it shouldn't, given the constrcutor) THROW exception
    //   // }

    //   if (m_kind[0] != kBatch) {
    //     m_kind.insert(m_kind.begin(), kBatch);
    //     return true;
    //   }

    //   return false;
    // }

  private:
    torch::Tensor m_tensor;
    std::vector<SPNG::TensorKind> m_kind;
    std::vector<SPNG::TensorDomain> m_domain;
    std::vector<std::string> m_batch_label;
    Configuration m_md;

};

}
#endif