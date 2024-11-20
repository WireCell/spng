#ifndef WIRECELL_SPNG_TORCHTENSOR
#define WIRECELL_SPNG_TORCHTENSOR

#include "WireCellSpng/IArray.h"
#include "WireCellIface/IFrame.h"

#include <torch/torch.h>

namespace WireCell {

  class TorchTensor: public IArray {
   public:
     TorchTensor(const std::vector<float> & input,
                 size_t m, size_t n,
                 torch::Device device=torch::kCPU);

     TorchTensor(torch::Tensor & input_tensor);
     TorchTensor(torch::Tensor input_tensor);

     ~TorchTensor(){};

     TorchTensor operator[](int a) const;

     //size_t GetSize();
     virtual std::string device() const;
     //virtual float * data();

     //Arithmetic operators
     TorchTensor operator+(const TorchTensor & rh) const;
     TorchTensor operator-(const TorchTensor & rh) const;
     TorchTensor operator*(const TorchTensor & rh) const;
     TorchTensor operator/(const TorchTensor & rh) const;

     template<class T>
     TorchTensor operator+(const T & rh) const;

     template<class T>
     TorchTensor operator-(const T & rh) const;

     template<class T>
     TorchTensor operator*(const T & rh) const;

     template<class T>
     TorchTensor operator/(const T & rh) const;

     //Maybe get rid of these
     torch::Tensor get_tensor() const;
     torch::Tensor & get_tensor_ref();

     TorchTensor reshape(size_t m, size_t n) const;

     void to(torch::Device device) {
        the_tensor.to(device);
     };

     friend std::ostream& operator<<(std::ostream& stream, const TorchTensor & tensor);

   private:
     std::vector<float> data;
     float * device_ptr;
     torch::Tensor the_tensor;

  };

  class TorchTensorFactory {
   public:
    static TorchTensor zeros(size_t n, size_t m);
    static TorchTensor ones(size_t n, size_t m);
    static TorchTensor vals(float v, size_t n, size_t m);
  };

}


#endif
