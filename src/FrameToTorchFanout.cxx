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
    log->debug("Checking planes");
    if (config.isMember("planes")) {
        log->debug("Got {} planes", config["planes"]);
        for (auto nwires : config["planes"]) { 
            log->debug("\tGot {} nwires", nwires.asInt());
            m_planes.push_back(nwires.asInt());
        }
    }

    
    //Fill the wires per-plane
    log->debug("Getting the channel map");
    if (config.isMember("channel_ranges")) {
        log->debug("Got {} channels", config["channel_ranges"]);
        for (auto ranges : config["channel_ranges"]) {
            // if (ranges.size() != 2) {
            // }
            auto plane = ranges[0].asUInt64();
            m_channel_ranges[plane] = std::vector<std::pair<size_t, size_t>>();
            for (auto range : ranges[1]) {
                log->debug("Range: {}", range);
                m_channel_ranges[plane].emplace_back(
                    range[0].asUInt64(), range[1].asUInt64()
                );
            }
            log->debug("Plane {}", ranges.size());
        }
    }

    log->debug("Getting ticks & mult");
    m_expected_nticks = get(config, "expected_nticks", m_expected_nticks);
    log->debug("Got {}", m_expected_nticks);
    m_multiplicity = get(config, "multiplicity", m_multiplicity);
    log->debug("Got {}", m_multiplicity);
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
        auto chan = (*in->traces())[i]->channel();
        int in_tensor_chan = 0;
        size_t this_plane = 0;

        bool found = false;

        //Loop over the ranges and see where this channel falls
        for (const auto & [plane, ranges] : m_channel_ranges) {
            size_t in_tensor_start = 0;
            for (const auto & range : ranges) {
                if (chan >= range.first && chan < range.second) {
                    this_plane = plane;
                    found = true;
                    in_tensor_chan = in_tensor_start + (chan - range.first);
                    break;
                }
                in_tensor_start += (range.second - range.first);
            }
        }

        //TODO -- configure Allow continue or throw
        if (!found) {
            continue;
        }

        //This is really inefficient
        auto nticks = (*in->traces())[i]->charge().size();
        for (size_t j = 0; j < nticks; ++j) {
            outv[this_plane]->tensor().index_put_({(int)in_tensor_chan, (int)j}, (*in->traces())[i]->charge()[j]);
            // plane_tensor.index_put_({(int64_t)i, (int64_t)j}, (*in->traces())[i]->charge()[j]);
        }
    }

    return true;
}
