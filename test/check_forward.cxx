#include "WireCellUtil/NamedFactory.h"
//#include "WireCellUtil/IComponent.h"
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
#include <signal.h>
#include <fstream>
#include <memory>

class Interface{
    public:
        typedef std::shared_ptr<Interface> pointer;
        virtual ~Interface();
};

Interface::~Interface(){};

template<class Type>
class IComponent : virtual public WireCell::Interface{
    public:
        virtual ~IComponent(){};
        typedef std::shared_ptr<Type> pointer;
        typedef std::vector<pointer> vector;
};

class TestForward: public IComponent<TestForward> {
    public:
      virtual ~TestForward(){};

     virtual WireCell::ITorchTensorSet::pointer forward(const WireCell::ITorchTensorSet::pointer& input) const = 0;
  };


/** An interface to a "forward" operator on a Torch set */

class TestBase {
public:
    virtual ~TestBase() = default;
    virtual void dummyMethod() = 0;
};

// Just some test interface...
//If the TestInterface does not inherit anything from ITorchForward or IConfigurable, no error during torch::jit::load...
class TestInterface:public TestForward,
                    //public WireCell::IConfigurable,
                    public TestBase
{
public:
    TestInterface();
    virtual ~TestInterface();
    //~TestInterface();
    virtual WireCell::ITorchTensorSet::pointer forward(const WireCell::ITorchTensorSet::pointer& input) const;
    virtual void configure(const WireCell::Configuration& cfg);
    virtual void dummyMethod() override;

private:
    mutable std::unique_ptr<torch::jit::script::Module> m_module;
    //mutable torch::jit::script::Module m_module;
    torch::Device m_device;
    bool m_configured = false;

};


TestInterface::TestInterface():
    m_device(torch::kCPU)
{}


TestInterface::~TestInterface()
{
}

void TestInterface::dummyMethod() {
    std::cout << "Dummy method called" << std::endl;
}

WireCell::ITorchTensorSet::pointer TestInterface::forward(const WireCell::ITorchTensorSet::pointer& input) const
{
    // Perform the forward pass using the TorchScript model
    auto ret = std::make_shared<WireCell::SPNG::SimpleTorchTensorSet>(0);
    return ret;
}


void TestInterface::configure(const WireCell::Configuration& cfg)
{
    std::cout << "TestInterface::configure called" << std::endl;
    
    try {
        if (!cfg.isMember("model")) {
            throw std::runtime_error("No 'model' key in configuration");
        }
        
        std::string model_path = cfg["model"].asString();
        std::cout << "Model path: " << model_path << std::endl;
        
        // Check file exists and is readable
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file does not exist: " + model_path);
        }
        
        std::cout << "File exists, checking size..." << std::endl;
        auto file_size = std::filesystem::file_size(model_path);
        std::cout << "File size: " << file_size << " bytes" << std::endl;
        
        if (file_size == 0) {
            throw std::runtime_error("Model file is empty: " + model_path);
        }
        
        std::cout << "About to call torch::jit::load..." << std::endl;
        std::cout.flush();  // Force output before potential crash
        
        // Load model with explicit error handling
            // Create module on heap
    m_module = std::make_unique<torch::jit::script::Module>(
        torch::jit::load(model_path, m_device)
    );

    std::cout << "torch::jit::load completed successfully" << std::endl;

    // Set to evaluation mode
    //m_module.eval();
        std::cout << "Model set to evaluation mode" << std::endl;
        
        m_configured = true;
        std::cout << "Configuration completed successfully" << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch C10 error in configure: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in configure: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown exception in configure" << std::endl;
        throw;
    }
}

int main(int argc, const char* argv[])
{
    // Create an instance of TestInterface using WireCell::Factory
    try {
        torch::manual_seed(42);
        torch::set_num_threads(1);  // Use single thread to avoid conflicts
        std::cout << "PyTorch initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize PyTorch: " << e.what() << std::endl;
        return 1;
    }
    auto test_interface = std::make_shared<TestInterface>();
    WireCell::Configuration cfg;
    const std::string model_path = "/nfs/data/1/abashyal/spng/spng_dev_050525/toolkit/spng/test/ts-model/test-2.3.ts";
    cfg["model"] = model_path;
    //check if the path exists
    
    if (!std::filesystem::exists(cfg["model"].asString())) {
        std::cerr << "Model path does not exist: " << cfg["model"].asString() << std::endl;
        return 1;
    }

    std::cout << "TestInterface: Configuring with model path: " << model_path << std::endl;
    test_interface->configure(cfg);

return 0;

}