#include "torch/torch.h"
#include <initializer_list>

torch::Tensor GetTensor(at::IntArrayRef sizes) {
  return torch::zeros(sizes);
}

int main(int argc, char * argv[]) {

  std::cout << "1x1" << std::endl;
  std::cout << GetTensor({1,1}) << std::endl;


  std::cout << "2x5" << std::endl;
  std::cout << GetTensor({2,5}) << std::endl;

  return 0;

}
