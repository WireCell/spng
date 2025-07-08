#include "WireCellSpng/TorchTensorSetStacker.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellAux/FrameTools.h"
#include "WireCellAux/SimpleFrame.h"
#include "WireCellIface/INamed.h"
#include <cuda.h>

WIRECELL_FACTORY(TorchTensorSetStacker, WireCell::SPNG::TorchTensorSetStacker,
                 WireCell::INamed,
                 WireCell::SPNG::ITorchTensorSetFanin)


using namespace WireCell;

WireCell::Configuration SPNG::TorchTensorSetStacker::default_configuration() const
{
    Configuration cfg;
    cfg["multiplicity"] = m_multiplicity;
    cfg["output_set_tag"] = m_output_set_tag;


    return cfg;
}

void SPNG::TorchTensorSetStacker::configure(const WireCell::Configuration& config)
{
    //Get the anode to make a channel map for output
    m_multiplicity = get(config, "multiplicity", m_multiplicity);
    m_output_set_tag = get(config, "output_set_tag", m_output_set_tag.asString());
    log->debug("Will tag output set with {}", m_output_set_tag);
}

std::vector<std::string> SPNG::TorchTensorSetStacker::input_types()
{
    if (!m_multiplicity) {
        return std::vector<std::string>();
    }
    const std::string tname = std::string(typeid(input_type).name());
    std::vector<std::string> ret(m_multiplicity, tname);
    return ret;
}


SPNG::TorchTensorSetStacker::TorchTensorSetStacker()
    : Aux::Logger("TorchTensorSetStacker", "spng") {}

bool SPNG::TorchTensorSetStacker::operator()(const input_vector& inv, output_pointer& out) {
    out = nullptr;


    size_t neos = 0;
    for (const auto& in : inv) {
        if (!in) {
            ++neos;
        }
    }
    if (neos) {
        log->debug("EOS with {}", neos);
        return true;
    }


    if (m_multiplicity > 0 && (inv.size() != m_multiplicity)) {
        THROW(ValueError()
            << errmsg{"Expected input multiplicity ({" + std::to_string(inv.size()) +
            "}) to match configuration ({" + std::to_string(m_multiplicity) + "})"});
    }

    std::vector<torch::Tensor> inputs;

    std::vector<TensorKind> output_kind;
    std::vector<TensorDomain> output_domain;

    for (auto in : inv) {
        auto this_set_tag = in->metadata()["tag"];
        
        for (auto tensor : (*in->tensors())) {
            const auto & this_kind = tensor->kind();
            const auto & this_domain = tensor->domain();

            //Check that kind + domain matches
            if (output_kind.size() == 0) {//First time
                output_kind = this_kind;
                output_domain = this_domain;
            }
            else {
                if (this_kind != output_kind) {
                    THROW(ValueError() << errmsg{"Trying to stack tensors with mismatching TensorKinds"});
                }
                if (this_domain != output_domain) {
                    THROW(ValueError() << errmsg{"Trying to stack tensors with mismatching TensorDomains"});
                }
            }
            
            inputs.push_back(tensor->tensor().clone());

            //We might need to unsqueeze this so we can concat later
            if (this_kind[0] != kBatch)
                inputs.back() = torch::unsqueeze(inputs.back(), 0);
            

            // auto this_tensor_tag = tensor->metadata()["tag"];
            // std::string combined_tag = this_set_tag.asString() + ":" + this_tensor_tag.asString();
            // log->debug("Adding tensor with tag {}", combined_tag);
            // Configuration tensor_md;
            // tensor_md["tag"] = combined_tag;
        //     itv.push_back(
        //         std::make_shared<SimpleTorchTensor>(tensor->tensor(), tensor_md)
        //     );
        }
    }
    Configuration tensor_md;
    // std::vector<SPNG::TensorKind> tensor_kind = {};
    // std::vector<SPNG::TensorDomain> tensor_domain = {};
    if (output_kind[0] != kBatch) {
        output_kind.insert(output_kind.begin(), kBatch);
        output_domain.insert(output_domain.begin(), kNull);
    }
    std::vector<std::string> batch_label(inputs.size(), "temp");

    auto output_tensor = std::make_shared<SimpleTorchTensor>(
        torch::cat(inputs), output_kind, output_domain, batch_label, tensor_md
    );
    std::vector<ITorchTensor::pointer> itv{
        output_tensor
    };

    Configuration output_md;
    output_md["tag"] = m_output_set_tag;
    log->debug("Writing out Tensor Set with output set tag {} {}", output_md["tag"], m_output_set_tag);
    out = std::make_shared<SimpleTorchTensorSet>(
        inv.at(0)->ident(), output_md,
        std::make_shared<std::vector<ITorchTensor::pointer>>(itv)
    );


    return true;
}
