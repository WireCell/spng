#include "WireCellSpng/Util.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"

#include <cmath>

namespace WireCell::SPNG {
    /*
    torch::Tensor gaussian1d(double mean, double sigma,
                                    int64_t npoints, double xmin, double xmax,
                                    torch::TensorOptions options)
    {
        auto x = torch::linspace(xmin, xmax, npoints, options);
        auto rel = (x - mean)/sigma;
        const double norm = sqrt(2*M_PI*sigma*sigma);
        return norm * torch::exp(-0.5*rel*rel);
    }


    std::vector<int64_t> linear_shape(const std::vector<torch::Tensor>& tens, 
                                            torch::IntArrayRef extra_shape)
    {
        // Find shape that assures linear convolution.
        std::vector<int64_t> shape = {extra_shape[0], extra_shape[1]};
        for (const auto& ten : tens) {
            auto sizes = ten.sizes();
            shape[0] += sizes[0] - 1;
            shape[1] += sizes[1] - 1;
        }
        return shape;
    }


    torch::Tensor pad(torch::Tensor ten, double value, torch::IntArrayRef shape)
    {
        using torch::indexing::Slice;

        torch::Tensor padded = torch::zeros(shape, ten.options()) + value;
        auto s = ten.sizes();
        padded.index_put_({
                Slice(0,std::min(s[0], shape[0])),
                Slice(0,std::min(s[1], shape[1]))
            }, ten);
        return padded;    
    }

    torch::Tensor convo_spec(const std::vector<torch::Tensor>& tens, 
                                    torch::IntArrayRef shape)
    {
        using torch::indexing::Slice;

        // Return value will be complex spectrum in Fourier domain.
        torch::Tensor fourier;
        // Allocate working array in interval domain.
        torch::Tensor interval = torch::zeros(shape, tens[0].options());

        const size_t ntens = tens.size();

        // First, accumulate denominator.
        for (size_t ind=0; ind<ntens; ++ind) {
            // Caveat: zero-padding is not always appropriate for every ten in tens.
            interval.zero_();

            const auto& ten = tens[ind];
            auto s = ten.sizes();
            interval.index_put_({
                    Slice(0,std::min(s[0], shape[0])),
                    Slice(0,std::min(s[1], shape[1]))
                }, ten);

            if (ind == 0) {
                fourier = torch::fft::fft2(interval);
            }
            else {
                fourier *= torch::fft::fft2(interval);
            }
        }

        return fourier;
    }


    torch::Tensor filtered_decon_2d(const std::vector<torch::Tensor>& numerator,
                                        const std::vector<torch::Tensor>& denominator,
                                        torch::IntArrayRef shape)
    {
        if (denominator.empty()) {  // graceful degradation 
            return convo_spec(numerator, shape);
        }

        // Note: this suffers holding an extra array which could be avoided by
        // accumulating denominator convolutions, inverting that result and
        // continuing accumulating numerator convolutions.
        auto num = convo_spec(numerator, shape);
        auto den = convo_spec(denominator, shape);
        return torch::divide(num, den); // fixme, divide-by-zero?
    }


    torch::Tensor filtered_decon_2d_auto(const std::vector<torch::Tensor>& numerator,
                                            const std::vector<torch::Tensor>& denominator,
                                            torch::IntArrayRef extra_shape)
    {
        std::vector<torch::Tensor> all_in(numerator.begin(), numerator.end());
        all_in.insert(all_in.end(), denominator.begin(), denominator.end());

        if (all_in.empty()) {
            return torch::Tensor();
        }

        auto shape = linear_shape(all_in, extra_shape);

        return filtered_decon_2d(numerator, denominator, shape);
    }
    */
    ITorchTensorSet::pointer to_itensor( const std::vector<torch::IValue>& inputs){
        auto itv = std::make_shared<ITorchTensor::vector>();
        // Populate this function as needed...

        for (const auto& ivalue : inputs) {
            if (!ivalue.isTensor()) {
                THROW(ValueError() << errmsg{"Expected torch::IValue to be a Tensor"});
            }
            torch::Tensor ten = ivalue.toTensor().cpu();
            if (ten.dim() != 4) {
                THROW(ValueError() << errmsg{"Tensor must be 4D"});
            }

            //why casting to float?
            if (ten.scalar_type() != torch::kFloat32) {
                ten = ten.to(torch::kFloat32);
            }
            //From torch tensor to ITorchTensor
            auto stp = std::make_shared<SimpleTorchTensor>(ten);
            itv->emplace_back(stp);
        }
        return std::make_shared<SimpleTorchTensorSet>(0,Json::nullValue, itv);
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