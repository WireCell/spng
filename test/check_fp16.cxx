#include "WireCellSpng/Torch.h"  // One-stop header.
#include <chrono>
#include <torch/torch.h>

/**
 * need to configure with torch_cpu on gpvm
 * e.g.--with-libtorch="$LIBTORCH_FQ_DIR/" --with-libtorch-libs torch,torch_cpu,c10
 * model for testing can be founc here:
 * https://www.phy.bnl.gov/~hyu/dunefd/dnn-roi-pdvd/Pytorch-UNet/ts-model/
*/

int main(int argc, const char* argv[])
{

    std::cout<<"Test with the GPU information on the machine "<<std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        int device_count = torch::cuda::device_count();
        std::cout << "Number of GPUs: " << device_count << std::endl;
        for (int i = 0; i < device_count; ++i) {
            torch::Device device(torch::kCUDA, i);
            std::cout << "GPU " << i << ": " << device.str() << std::endl;
        }
    } else {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }

    std::cout << "WireCell::pytorch : test loading TorchScript Model\n";

    const std::string mname = "/nfs/data/1/abashyal/spng/spng_dev_050525/toolkit/spng/test/ts-model/unet-l23-cosmic500-e50-new.ts";
    auto dtype = torch::kFloat16;

    torch::jit::script::Module module;
    // Deserialize the ScriptModule from a file using torch::jit::load().
    auto start = std::chrono::high_resolution_clock::now();
    torch::Device device = torch::Device(torch::kCUDA,0);
    module = torch::jit::load(mname, device);
    module.to(at::kCPU, dtype);
    torch::TensorOptions options = torch::TensorOptions().dtype(dtype);
    torch::Tensor iten = torch::rand({1, 3, 800, 600}, options);
    // torch::Tensor iten = torch::zeros({1, 3, 800, 600}, options);
    std::vector<torch::IValue> itens {iten};
    auto otens = module.forward(itens).toTensor();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "timing: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    return 0;
}