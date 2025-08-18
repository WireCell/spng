#include "WireCellSpng/Util.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>

namespace WireCell::SPNG {
    
    void save_torchtensor_data(const  torch::Tensor& tensor, const std::string& filename) {
        // Create a map<string, torch::Tensor> to hold the tensor with a unique key
        torch::serialize::OutputArchive archive;
    
        // Store the tensor with a unique key
        archive.write("tensor", tensor);
    
        try {
            // Save the tensor in one file
            archive.save_to(filename);
        } catch (const c10::Error& e) {
            std::cerr << "Util::save_torchtensor_data  Error saving tensor: " << e.what() << std::endl;
        }
    }  
    
    void save_simpletensor_data(const ITorchTensorSet::pointer& in, const std::string& filename) {
        // Create a map<string, torch::Tensor> to hold all tensors with unique keys
        torch::serialize::OutputArchive archive;
    
        int idx = 0;
        for (const auto& tensor : *in->tensors()) {
            auto ten = tensor->tensor();
    

    
            // Move tensor to CPU if needed for portability
            torch::Tensor cpu_tensor = ten.device().is_cpu() ? ten : ten.to(torch::kCPU);
    
            // Store with a unique key
            std::string key = "tensor_" + std::to_string(idx++);
            archive.write(key, cpu_tensor);
        }
    
        try {
            // Save the entire map of tensors in one file
            archive.save_to(filename);
        } catch (const c10::Error& e) {
            std::cerr << "Util::save_simpletensor_data  Error saving tensors: " << e.what() << std::endl;
        }
    }
    

    
    

    ITorchTensorSet::pointer to_itensor(const std::vector<torch::IValue>& inputs) {
        auto itv = std::make_shared<ITorchTensor::vector>();
    
        for (size_t i = 0; i < inputs.size(); ++i) {
            try {
                const auto& ivalue = inputs[i];
                if (!ivalue.isTensor()) {
                    std::cerr << "Error: Expected torch::IValue at index " << i << " to be a Tensor\n";
                    continue;
                }
                torch::Tensor ten = ivalue.toTensor();
    
    
                if (ten.dim() != 4) {
                    std::cerr << "Error: Tensor at index " << i << " must be 4D, got " << ten.dim() << std::endl;
                    continue;
                }
    
                if(ten.scalar_type() != torch::kFloat32) {
                    ten = ten.to(torch::kFloat32);
                    std::cout << "Converted tensor " << i << " to float32." << std::endl;
                }
    
                std::cout << "Tensor " << i << ": shape=" << ten.sizes() 
                << ", dtype=" << ten.dtype() 
                << ", device=" << ten.device() << std::endl;
    
                // No forced dtype or device conversion here to preserve input tensors as-is
                auto stp = std::make_shared<SimpleTorchTensor>(ten);
                itv->emplace_back(stp);
    
            } catch (const std::exception& e) {
                std::cerr << "Exception caught while processing tensor " << i << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception caught while processing tensor " << i << std::endl;
            }
        }
        return std::make_shared<SimpleTorchTensorSet>(0, Json::nullValue, itv);
    }

    //ITorchTensor --> torch::IValue
    std::vector<torch::IValue> from_itensor(const ITorchTensorSet::pointer& in, bool is_gpu)
    {
        // Create a new SimpleTorchTensorSet to hold the converted tensors
        std::vector<torch::IValue> ret;
        //Populate this function as needed...

        for(auto iten: *in->tensors()) {
            // Convert each tensor to IValue
            torch::Tensor ten = iten->tensor();
            if(ten.dim() != 4) {
                THROW(ValueError() << errmsg{"Tensor must be 4D"});
            }
            //why casting to float?
            if(ten.scalar_type() != torch::kFloat32) {
                ten = ten.to(torch::kFloat32);
            }
            if (is_gpu) {
                ten = ten.to(torch::Device(torch::kCUDA, 0));
                assert(ten.device().type() == torch::kCUDA);
            } 
            ret.emplace_back(ten);  
        }
        return ret;
    }
}