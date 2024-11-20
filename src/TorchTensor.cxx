#include "WireCellSpng/TorchTensor.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace WireCell {



TorchTensor TorchTensorFactory::vals(float v, size_t m, size_t n) {
  std::vector<float> input(n*m, v);
  return TorchTensor(input, m, n);
}

TorchTensor TorchTensorFactory::zeros(size_t m, size_t n) {
  return vals(0., m, n);
}

TorchTensor TorchTensorFactory::ones(size_t m, size_t n) {
  return vals(1., m, n);
}

TorchTensor::TorchTensor(const std::vector<float> & input,
                         size_t m,
                         size_t n,
                         torch::Device device)
    : data(input) {

  // Create a CUDA device pointer, populate it with data from the host
  //TODO -- MAKE SURE THE MEMORY HANDLING IS CLEAN
  if (device == torch::kCUDA) {
    //device_ptr;
    cudaMalloc((void **)&device_ptr, input.size() * sizeof(float));
    cudaMemcpy(device_ptr, &data[0], input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  //Make the dtype an arg
  auto options = torch::TensorOptions().dtype(at::kFloat)
                                       .device(device);

  the_tensor = torch::from_blob(
      (device == torch::kCPU ? &data[0] : device_ptr),
      //&data[0],
      //device_ptr,
      {m, n},
      options
  );
}

TorchTensor::TorchTensor(torch::Tensor & input_tensor)
  : the_tensor(input_tensor) {}

TorchTensor::TorchTensor(torch::Tensor input_tensor)
  : the_tensor(input_tensor) {}


TorchTensor TorchTensor::operator[](int a) const {
  return TorchTensor(the_tensor[{a}].detach()); 
}

std::string TorchTensor::device() const {
  return the_tensor.device().str();
}

TorchTensor TorchTensor::operator*(const TorchTensor & rh) const {
  return TorchTensor(
    this->the_tensor.detach() * rh.the_tensor.detach()
  );
}
TorchTensor TorchTensor::operator/(const TorchTensor & rh) const {
  return TorchTensor(
    this->the_tensor.detach() / rh.the_tensor.detach()
  );
}

TorchTensor TorchTensor::operator-(const TorchTensor & rh) const {
  return TorchTensor(
    this->the_tensor.detach() - rh.the_tensor.detach()
  );
}

TorchTensor TorchTensor::operator+(const TorchTensor & rh) const {
  return TorchTensor(
    this->the_tensor.detach() + rh.the_tensor.detach()
  );
}

//Is this good practice?...
template<class T>
TorchTensor TorchTensor::operator+(const T & rh) const {
  return TorchTensor(
    this->the_tensor.detach() + rh 
  );
}
template TorchTensor TorchTensor::operator+<float>(const float & rh) const;
template TorchTensor TorchTensor::operator+<int>(const int & rh) const;

template<class T>
TorchTensor TorchTensor::operator*(const T & rh) const {
  return TorchTensor(
    this->the_tensor.detach() * rh 
  );
}
template TorchTensor TorchTensor::operator*<float>(const float & rh) const;
template TorchTensor TorchTensor::operator*<int>(const int & rh) const;


template<class T>
TorchTensor TorchTensor::operator-(const T & rh) const {
  return TorchTensor(
    this->the_tensor.detach() - rh 
  );
}
template TorchTensor TorchTensor::operator-<float>(const float & rh) const;
template TorchTensor TorchTensor::operator-<int>(const int & rh) const;

template<class T>
TorchTensor TorchTensor::operator/(const T & rh) const {
  return TorchTensor(
    this->the_tensor.detach() / rh 
  );
}
template TorchTensor TorchTensor::operator/<float>(const float & rh) const;
template TorchTensor TorchTensor::operator/<int>(const int & rh) const;


torch::Tensor TorchTensor::get_tensor() const {
  return the_tensor.detach().clone();
}

torch::Tensor & TorchTensor::get_tensor_ref() {
  return the_tensor;
}

std::ostream& operator<<(std::ostream& stream, const TorchTensor & tensor) {
  stream << tensor.the_tensor;

  return stream;
}

TorchTensor TorchTensor::reshape(size_t m, size_t n) const {
  return TorchTensor(this->data, m, n); 
}

/*TorchTensor::TorchTensor(IFrame & frame) {

  auto traces = frame.traces();
  size_t m = (*traces.get()).size();
  //Check if the size is 0
  size_t n = 0; //traces[0].ChargeSequence.size();
  //Check if the size is 0

  //Flatten it
  size_t prev_size = 0;
  for (auto & trace : *traces.get()) {
    auto & seq = trace->charge();
    n = seq.size();

    //Make sure they're all the same size 
    if (prev_size > 0 && n != prev_size) {

      std::string message = "Error! Attempted to make jagged TorchTensor" ;
      throw std::runtime_error(message);
    }

    //Put it in the flat vec
    for (auto & val : seq) {
      data.push_back(val);
    }
    prev_size = n;
  }

  auto options = torch::TensorOptions().dtype(at::kFloat);
  the_tensor = torch::from_blob(
      &data[0],
      {m, n},
      options
  );
}*/

/*size_t TorchTensor::GetSize() {
  return data.size();
}*/
}
