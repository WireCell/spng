#include "WireCellSpng/TorchContext.h"
#include "WireCellUtil/NamedFactory.h"

using namespace WireCell;
using namespace WireCell::SPNG;

TorchContext::TorchContext() {}
TorchContext::~TorchContext() { }
TorchContext::TorchContext(const std::string& devname)
{
    connect(devname);
}
void TorchContext::connect(const std::string& devname)
{
    std::cout<<"Connecting to Torch device: " << devname << std::endl;
    // Use almost 1/2 the memory and 3/4 the time.
    torch::NoGradGuard no_grad;

    if (devname == "cpu") {
        m_dev = torch::Device(torch::kCPU);
    }
    else {
        int devnum = 0;
        if (devname.size() > 3) {
            devnum = atoi(devname.substr(3).c_str());
        }
        m_dev = torch::Device(torch::kCUDA, devnum);
    }
    m_devname = devname;
    std::cout << "Connected to Torch device: " << m_devname << std::endl;
}
