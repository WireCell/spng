#include "WireCellUtil/NamedFactory.h"
#include "WireCellIface/IConfigurable.h"
//#include "WireCellAux/Logger.h"
#include "WireCellSpng/Torch.h"  // One-stop header.
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/ITorchTensorSet.h"
#include "WireCellSpng/ITorchForward.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/Util.h"
#include <chrono>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>


// Just some test interface...
class TestInterface:public WireCell::SPNG::ITorchForward,
                    public WireCell::IConfigurable
{
public:
    TestInterface();
    virtual ~TestInterface();
    virtual WireCell::ITorchTensorSet::pointer forward(const WireCell::ITorchTensorSet::pointer& input) const override;
    virtual void configure(const WireCell::Configuration& cfg);

private:
    mutable torch::jit::script::Module m_module;
    torch::Device m_device{torch::kCPU};

};

WIRECELL_FACTORY(TestInterface,
    TestInterface,
    WireCell::SPNG::ITorchForward,
    WireCell::IConfigurable)

TestInterface::TestInterface(){}


TestInterface::~TestInterface()
{
}

WireCell::ITorchTensorSet::pointer TestInterface::forward(const WireCell::ITorchTensorSet::pointer& input) const
{
    // Perform the forward pass using the TorchScript model
    auto ret = std::make_shared<WireCell::SPNG::SimpleTorchTensorSet>(0);
    return ret;
}
void TestInterface::configure(const WireCell::Configuration& cfg)
{
    // Configure the TorchScript model
    std::cout<<"Call the TestInterface::configure "<<std::endl;
    std::string model_path = cfg["model"].asString();
    m_module = torch::jit::load(model_path,m_device);
}

int main(int argc, const char* argv[])
{
    // Create an instance of TestInterface using WireCell::Factory
    const std::string test_iface = "TestInterface";
    const std::string test_inst = "test";
    {
        auto test_interface = std::make_shared<TestInterface>();
        WireCell::Configuration cfg;
        cfg["model"] = "/nfs/data/1/abashyal/spng/spng_dev_050525/toolkit/spng/test/ts-model/test-2.3.ts";
        //check if the path exists
        if (!std::filesystem::exists(cfg["model"].asString())) {
            std::cerr << "Model path does not exist: " << cfg["model"].asString() << std::endl;
            return 1;
        }
        std::cout<<"TestInterface: Configuring with model path: "<<cfg["model"].asString()<<std::endl;
        std::cout.flush(); //output before the crash..
        test_interface->configure(cfg);
    }
    return 0;

}