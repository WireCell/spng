#include "WireCellSpng/ROITests.h"
#include "WireCellSpng/ITorchTensorSetFilter.h"

#include "WireCellUtil/NamedFactory.h"

//register this (ROITests) as a factory
WIRECELL_FACTORY(SPNGROITests,// name of the factory
    WireCell::SPNG::ROITests, // name of the class
    WireCell::INamed, // name of the interface 1 (allows object to have unique name)
    WireCell::ITorchTensorSetFilter, // interface 2 (process ITorchTensorSet)
    WireCell::IConfigurable // interface 3 (allows configuration)
    )

using namespace WireCell;
using namespace WireCell::SPNG;

//First step. Get the output of Decon.cxx here as input. 
ROITests::ROITests()
    : Aux::Logger("ROITests", "spng")
{
}
ROITests::~ROITests()
{
}

void ROITests::configure(const WireCell::Configuration& cfg)
{
    //configuration parameters required to run this code.
}

void ROITests::finalize()
{

}

bool ROITests::operator()(const input_pointer& in, output_pointer& out)
{
    out = nullptr;
    if (!in) {
        log->debug("EOS ");
        return true;
    }
    log->debug("Running ROITests");
    
    auto tensor_clone = in->tensors()->at(0)->tensor().clone();
    auto sizes = tensor_clone.sizes();
    std::cout<<"ROITests: tensor_clone sizes: "<<sizes<<std::endl;
    return true;
}