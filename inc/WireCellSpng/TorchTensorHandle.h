#ifndef WIRECELL_SPNG_TORCHTENSORHANDLE
#define WIRECELL_SPNG_TORCHTENSORHANDLE

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace WireCell {
  class TorchTensorHandle {
   public:

    template <class T>
    TorchTensorHandle(const std::vector<T> & input,
                      at::IntArrayRef sizes,
                      torch::Device device=torch::kCPU){

      if (device == torch::kCUDA) {
        //If we're using cuda, allocate memory on the device (TODO -- check which device)
        //and copy data into the pointer (on device)
        cudaMalloc((void **)&m_data, input.size() * sizeof(T));
        cudaMemcpy(m_data, &input[0], input.size() * sizeof(T),
                  cudaMemcpyHostToDevice);
      }
      else {
        //Do the same thing on host (CPU)
        m_data = malloc(input.size()*sizeof(T));
        memcpy(m_data, &input[0], input.size()*sizeof(T));
      }

      //Get the type of data, turn off comp. graph, and send to device.
      auto options = torch::TensorOptions().dtype(get_dtype(input))
                                           .requires_grad(false)
                                           .device(device);

      //Create the tensor from the data we imported
      m_tensor = torch::from_blob(
          m_data,
          sizes,
          options
      );
    }
    
    TorchTensorHandle(torch::Tensor & input_tensor);
    ~TorchTensorHandle();
    torch::Tensor clone_tensor() const;

   private:

    //TODO -- add in more dtypes
    //see https://pytorch.org/docs/stable/tensor_attributes.html
    torch::Dtype get_dtype(const std::vector<float> & input);
    torch::Dtype get_dtype(const std::vector<int> & input);

    torch::Tensor m_tensor;
    void * m_data = nullptr;
  };
}

#endif