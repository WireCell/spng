#include "WireCellSpng/TorchSetToFrame.h"
#include "WireCellIface/INamed.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Units.h"
#include "WireCellAux/SimpleTrace.h"
#include "WireCellAux/SimpleFrame.h"
#include <algorithm>
WIRECELL_FACTORY(TorchSetToFrame, WireCell::SPNG::TorchSetToFrame,
                 WireCell::INamed,
                 WireCell::SPNG::ITorchSetToFrame)

using namespace WireCell;

WireCell::SPNG::TorchSetToFrame::TorchSetToFrame()
    : Aux::Logger("TorchSetToFrame", "spng") {}

WireCell::Configuration SPNG::TorchSetToFrame::default_configuration() const
{
    Configuration cfg;
    cfg["anode"] = m_anode_tn;
    return cfg;
}

void SPNG::TorchSetToFrame::configure(const WireCell::Configuration& config) {

    //Get the anode to make a channel map for output
    m_anode_tn = get(config, "anode", m_anode_tn);
    m_anode = Factory::find_tn<IAnodePlane>(m_anode_tn);

    //Get output groups (map WirePlaneId --> output index)
    if (config.isMember("input_groups")) {
        auto groups = config["input_groups"];
        int i = 0;
        for (auto group : groups) {
            for (auto wpid : group) {
                WirePlaneId the_wpid(wpid.asInt());
                m_input_groups[the_wpid] = i;
                log->debug("WPID: {}", WirePlaneId(wpid.asInt()));
            }
            ++i;
        }
    }

    //Make a map to go from the channel ID to the output group    
    //First Loop over the faces in the anode we're working with
    for (const auto & face : m_anode->faces()) {
        if (!face) {   // A null face means one sided AnodePlane.
            continue;  // Can be "back" or "front" face.
                       //Throw instead?
        }

        for (const auto & plane : face->planes()) {

            int in_group = m_input_groups[plane->planeid()];
            for (const auto & channel : plane->channels()) {

                //Within a given plane, the traces will be in order as they were
                //seen here. We have a map to determine what the output
                //size (nchannels) is when we make the tensors later.
                //When first accessed with [] it will put in zero.
                //Using obj++ returns the original obj value before incrementing.
                m_channel_map[m_input_nchannels[in_group]++] = channel->ident();
                // std::cout << "[hyu1]chmap: " << channel->ident() << " " << plane->ident() << " " << m_channel_map[channel->ident()] << std::endl;
            }
        }
    }
}

bool SPNG::TorchSetToFrame::operator()(const input_pointer& in, output_pointer& out) {
    out = nullptr;

    //Nothing in, nothing out
    if (!in) {  //  pass on EOS
        log->debug("Exiting");
        return true;
    }

   

    //Iterate over the set of TorchTensors
    int in_group = 0;
    ITrace::vector traces;
    for (auto torch_tensor : *(in->tensors())) {
        //Get the tensor and clone it. And send it to CPU
        auto tensor_clone = torch_tensor->tensor().clone().to(torch::kCPU);
        int nticks = tensor_clone.sizes()[1];
        at::TensorAccessor<double,2> accessor = tensor_clone.accessor<double,2>();
        
        for (int chan_index = 0; chan_index < m_input_nchannels[in_group]; ++chan_index) {
            ITrace::ChargeSequence charge(nticks);
            
            //Get the pointer to the underlying data (hope this works)
            for (int itick = 0; itick < nticks; ++itick) {
                charge[itick] = accessor[chan_index][itick];
            }
            auto chid = m_channel_map[chan_index];
            int tbin = 0;
            auto trace = std::make_shared<WireCell::Aux::SimpleTrace>(chid, tbin, charge);
            traces.push_back(trace);
        
        }
        ++in_group;
    }
    auto frame = std::make_shared<WireCell::Aux::SimpleFrame>(0, -250.*units::us, traces, 0.5*units::us);
}