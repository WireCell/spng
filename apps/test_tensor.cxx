#include <iostream>
#include "WireCellSpng/TorchTensor.h"
//#include "torch/torch.h"

void print_vec(const std::vector<float> & data) {
  for (const auto & d : data) std::cout << d << " ";
  std::cout << std::endl;
}

int main(int argc, char * argv[]) {

  /*torch::Device device = torch::kCPU;
  if (argc > 1) {
    device = torch::kCUDA;
  }*/

  std::vector<float> data = {
      0,  1,  2,  3,
      4,  5,  6,  7,
      8,  9, 10, 11,
      12, 13, 14, 15
  };

  WireCell::TorchTensor my_tensor(data, 4, 4/*, device*/);
  auto tensor_copy = my_tensor.get_tensor();
  std::cout << tensor_copy << std::endl;

  std::cout << std::endl;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << tensor_copy[i][j] << " " << my_tensor[i][j] << std::endl;
    }
  }

  std::cout << my_tensor.get_tensor_ref() << std::endl;
  print_vec(data);

  std::vector<float> data2(16, 1.);
  WireCell::TorchTensor my_tensor2(data2, 4, 4);

  auto added = my_tensor + my_tensor2;

  //std::ostringstream stream;
  //stream << my_tensor.get_tensor();
  //std::string tensor_string = stream.str();
  std::cout << my_tensor << std::endl;
  std::cout << my_tensor2 << std::endl;
  std::cout << added << std::endl;

  data[0] += 1;
  std::cout << "Changed data? ";
  print_vec(data);

  //std::cout << (my_tensor[0][0] * 2) << std::endl;

  std::cout << my_tensor.get_tensor() << std::endl;
  std::cout << "Ostream: " << my_tensor << std::endl;

  std::cout << WireCell::TorchTensorFactory::zeros(3,2) << std::endl;
  std::cout << WireCell::TorchTensorFactory::ones(3,2) << std::endl;
  std::cout << WireCell::TorchTensorFactory::vals(7., 3,2) << std::endl;

  std::cout << "Subtraction:" << std::endl;
  auto subbed = my_tensor - my_tensor2;

  std::cout << my_tensor << std::endl;
  std::cout << my_tensor2 << std::endl;
  std::cout << subbed << std::endl;

  std::cout << (WireCell::TorchTensorFactory::ones(3,2) -
                WireCell::TorchTensorFactory::ones(3,2)) << std::endl;

  std::cout << "x2" << std::endl;
  std::cout << my_tensor * 2 << std::endl;
  std::cout << std::endl;

  std::cout << "+2" << std::endl;
  std::cout << my_tensor + 2 << std::endl;
  std::cout << std::endl;


  std::cout << "Testing reshape" << std::endl;
  std::cout << my_tensor.reshape(2,8) << std::endl; 
  std::cout << std::endl;
  std::cout << my_tensor << std::endl; 
  std::cout << std::endl;

  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Sending to GPU" << std::endl;
    auto cuda_tensor = my_tensor.get_tensor_ref().to(torch::kCUDA);
    //my_tensor.get_tensor_ref().cuda();
    std::cout << cuda_tensor << std::endl;
  }
  std::cout << my_tensor.get_tensor_ref().device() << std::endl;

  std::cout << torch::ones({1,1}).cuda() << std::endl;


  //auto options = torch::TensorOptions().dtype(at::kFloat);
  //torch::Tensor the_tensor = torch::from_blob(
  //    &data[0],
  //    {4, 4},
  //    options
  //);

  //std::cout << "Accessing tensor:" << std::endl;
  //for (int i = 0; i < 4; ++i) {
  //  std::cout << "Row " << i << ":";
  //  for (int j = 0; j < 4; ++j) {
  //    std::cout << " " << the_tensor[i][j].item<float>();
  //  }
  //  std::cout << std::endl;
  //}

  //torch::Device device = torch::kCPU;
  //std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  //if (torch::cuda::is_available()) {
  //  std::cout << "CUDA is available! Training on GPU." << std::endl;
  //  device = torch::kCUDA;
  //}
  //the_tensor.to(torch::kCUDA);

  return 0;
}
