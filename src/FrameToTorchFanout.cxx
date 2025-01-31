#include "WireCellSpng/FrameToTorchFanout.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellAux/FrameTools.h"
#include "WireCellAux/SimpleFrame.h"
#include "WireCellIface/INamed.h"


WIRECELL_FACTORY(FrameToTorchFanout, WireCell::SPNG::FrameToTorchFanout,
                 WireCell::INamed,
                 WireCell::SPNG::IFrameToTorchFanout)


using namespace WireCell;

WireCell::Configuration SPNG::FrameToTorchFanout::default_configuration() const
{
    Configuration cfg;
    cfg["Test"] = true;
    //What to do with planes?
    return cfg;
}

void SPNG::FrameToTorchFanout::configure(const WireCell::Configuration& config)
{
    //Fill the wires per-plane
    std::cout << "Checking planes" << std::endl;
    if (config.isMember("planes")) {
        std::cout << "Got {} planes" << config["planes"] << std::endl;
        for (auto nwires : config["planes"]) { 
            std::cout << "\tGot {} nwires" << nwires.asInt() << std::endl;
            m_planes.push_back(nwires.asInt());
        }
    }

    std::cout << "Getting ticks & mult" << std::endl;
    m_expected_nticks = get(config, "expected_nticks", m_expected_nticks);
    std::cout << "Got {}" << m_expected_nticks << std::endl;
    m_multiplicity = get(config, "multiplicity", m_multiplicity);
    std::cout << "Got {}" << m_multiplicity << std::endl;
}

std::vector<std::string> SPNG::FrameToTorchFanout::output_types()
{
    const std::string tname = std::string(typeid(ITorchTensor).name());
    std::vector<std::string> ret(m_multiplicity, tname);
    return ret;
}


SPNG::FrameToTorchFanout::FrameToTorchFanout()
    : Aux::Logger("FrameToTorchFanout", "spng") {}

bool SPNG::FrameToTorchFanout::operator()(const input_pointer& in, output_vector& outv) {
    outv.resize(m_multiplicity);
    //Default null ptrs
    for (size_t ind = 0; ind < m_multiplicity; ++ind) {
        outv[ind] = nullptr;
    }

    //Nothing in, nothing out
    if (!in) {  //  pass on EOS
        log->debug("Exiting");
        return true;
    }

    //Exit if no traces
    const size_t ntraces = in->traces()->size();
    log->debug("Ntraces: {}", ntraces);
    if (ntraces == 0) {
        log->debug("No traces, exiting");
        return true;
    }


    //Build up Tensors according to the planes
    for (size_t i = 0; i < m_planes.size(); ++i) {
        const auto & nwires = m_planes[i];
        log->debug("Making tensor of shape: {} {}", nwires, m_expected_nticks);
        torch::Tensor plane_tensor = torch::zeros({nwires, m_expected_nticks});
        outv[i] = SimpleTorchTensor::pointer(
            new SimpleTorchTensor(plane_tensor.clone())
        );
        log->debug("Made");

    }


    // torch::Tensor plane_tensor = torch::zeros({ntraces, nticks});
    for (size_t i = 0; i < ntraces; ++i) {
        auto chan = (*in->traces())[0]->channel();
        log->debug("Trace {}, Channel {}", i, chan);
        // auto nticks = (*in->traces())[0]->charge().size();
        // for (size_t j = 0; j < nticks; ++j) {
        //     plane_tensor.index_put_({(int64_t)i, (int64_t)j}, (*in->traces())[i]->charge()[j]);
        // }    
    }

    return true;
}
