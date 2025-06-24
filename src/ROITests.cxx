#include "WireCellSpng/ROITests.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellUtil/NamedFactory.h"

//register this (ROITests) as a factory
WIRECELL_FACTORY(SPNGROITests,// name of the factory
    WireCell::SPNG::ROITests, // name of the class
    WireCell::INamed, // name of the interface 1 (allows object to have unique name)
    //WireCell::ITorchTensorSetFilter, // interface 2 (process ITorchTensorSet)
    WireCell::SPNG::ITorchForward, //interface 2 (process ITorchForward)
    WireCell::IConfigurable // interface 3 (allows configuration)
    )

using namespace WireCell;
using namespace WireCell::SPNG;

//First step. Get the output of the Collator here as input. 
ROITests::ROITests()
    : Aux::Logger("SPNGROITests", "spng")
{
}
ROITests::~ROITests()
{
}

void ROITests::configure(const WireCell::Configuration& cfg)
{
   m_cfg.anode = get(cfg, "anode",m_cfg.anode);
}

void ROITests::finalize()
{

}

bool ROITests::operator()(const input_pointer& in, output_pointer& out)
{
    out = nullptr;
    std::cout<<"Calling the ROITests operator()"<<std::endl;
    log->debug("Calling ROITests operator()");
    if (!in) {
        log->debug("ROITests: EOS ");
        return true;
    }
    log->debug("Running ROITests");
    
     //TODO -- Loop over input tensors
    auto tensors = in->tensors();
    if (tensors->empty()) {
        log->debug("ROITests: No tensors in input set");
        return false;
    }
    //Process each tensor in the input set
    // Process each tensor in the input set
    for (size_t i = 0; i < tensors->size(); ++i) {
        auto tensor = tensors->at(i)->tensor();
        auto tensor_clone = tensor.clone();

        // Check the tensor tags (metadata)
        auto metadata = tensors->at(i)->metadata();
        if (metadata.isMember("tag")) {
            log->debug("ROITests: Found tag in metadata: {}", metadata["tag"].asString());
        } else {
            log->warn("ROITests: No tag found in metadata for tensor {}", i);
        }
    }
    out = std::make_shared<SimpleTorchTensorSet>(
        in->ident(), in->metadata(), tensors
    );
    return true;
}