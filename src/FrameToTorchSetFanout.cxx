#include "WireCellSpng/FrameToTorchSetFanout.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellAux/FrameTools.h"
#include "WireCellAux/SimpleFrame.h"
#include "WireCellIface/INamed.h"
#include <cuda.h>

WIRECELL_FACTORY(FrameToTorchSetFanout, WireCell::SPNG::FrameToTorchSetFanout,
                 WireCell::INamed,
                 WireCell::SPNG::IFrameToTorchSetFanout)


using namespace WireCell;

WireCell::Configuration SPNG::FrameToTorchSetFanout::default_configuration() const
{
    Configuration cfg;
    cfg["anode"] = m_anode_tn;
    return cfg;
}

void SPNG::FrameToTorchSetFanout::configure(const WireCell::Configuration& config)
{
    //Get the anode to make a channel map for output
    m_anode_tn = get(config, "anode", m_anode_tn);
    m_anode = Factory::find_tn<IAnodePlane>(m_anode_tn);


    m_expected_nticks = get(config, "expected_nticks", m_expected_nticks);
    log->debug("Got {}", m_expected_nticks);
    // m_multiplicity = get(config, "multiplicity", m_multiplicity);
    // log->debug("Got {}", m_multiplicity);

    //Get output groups (map WirePlaneId --> output index)
    if (config.isMember("output_groups")) {
        auto groups = config["output_groups"];
        int i = 0;
        m_multiplicity = groups.size();
        for (auto group : groups) {
            for (auto wpid : group) {
                WirePlaneId the_wpid(wpid.asInt());
                m_output_groups[the_wpid] = i;
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

        //Loop over each plane in the face
        for (const auto & plane : face->planes()) {

            //Each channel in the plane will be associated
            //with the corresponding WirePlaneId's output group
            for (const auto & channel : plane->channels()) {

                auto out_group = m_output_groups[plane->planeid()];
                m_channel_to_output_group[channel->ident()] = out_group;

                //Within a given plane, the traces will be in order as they were
                //seen here. We have a map to determine what the output
                //size (nchannels) is when we make the tensors later.
                //When first accessed with [] it will put in zero.
                //Using obj++ returns the original obj value before incrementing.
                m_channel_map[channel->ident()] = m_output_nchannels[out_group]++;
            }
        }
    }

}

std::vector<std::string> SPNG::FrameToTorchSetFanout::output_types()
{
    const std::string tname = std::string(typeid(ITorchTensorSet).name());
    log->debug("Got {}", m_multiplicity);
    std::vector<std::string> ret(m_multiplicity, tname);
    return ret;
}


SPNG::FrameToTorchSetFanout::FrameToTorchSetFanout()
    : Aux::Logger("FrameToTorchSetFanout", "spng") {}

bool SPNG::FrameToTorchSetFanout::operator()(const input_pointer& in, output_vector& outv) {
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
    log->debug("Tick (Period): {}", in->tick());
    if (ntraces == 0) {
        log->debug("No traces, exiting");
        return true;
    }


    for (auto face : m_anode->faces()) {
        if (!face) {   // A null face means one sided AnodePlane.
                    //Throw?
            continue;  // Can be "back" or "front" face.
        }
        for (auto plane : face->planes()) {
            std::cout << plane->planeid() << std::endl;
        }
    }

    std::vector<at::TensorAccessor<float,2>> accessors;
    std::vector<torch::Tensor> tensors;
    //Build up tenors + accessors to store input trace values

    for (const auto & [out_group, nchannels] : m_output_nchannels) {
        log->debug("Making tensor of shape: {} {}", nchannels, m_expected_nticks);
        torch::Tensor plane_tensor = torch::zeros({nchannels, m_expected_nticks});
        tensors.push_back(plane_tensor);
        accessors.push_back(tensors.back().accessor<float,2>());
    }

    

    //Now loop over the traces from the input frame, get where the output should go,
    //and put into the temp vector
    for (size_t i = 0; i < ntraces; ++i) {
        auto trace = (*in->traces())[i];
        auto chan = trace->channel();

        //Will throw if not found
        auto output_group = m_channel_to_output_group.at(chan);
        auto output_index = m_channel_map.at(chan);

        const auto & charge_seq = trace->charge();
        //Number of ticks.
        //TODO Maybe check against expectations from config and throw if different
        //or consider allowing this 
        // const int ntbins = std::min((int) charge_seq.size(), m_nticks);
        auto ntbins = charge_seq.size();
        int tbin = trace->tbin();

        for (size_t j = 0; j < ntbins; ++j) {
            accessors[output_group][output_index][tbin + j] = charge_seq[j];
        }
    }
    bool has_cuda = torch::cuda::is_available();

    torch::Device device((has_cuda ? torch::kCUDA : torch::kCPU));
    //Build up Tensors according to the output groups
    for (const auto & [output_index, nchannels] : m_output_nchannels) {
        
        // TODO: set md
        Configuration set_md;
        set_md["period"] = in->tick();

        //Clone the tensor to take ownership of the memory and put into 
        //output 
        std::vector<ITorchTensor::pointer> itv{
            std::make_shared<SimpleTorchTensor>(tensors[output_index].to(device)) //.clone())
        };
        outv[output_index] = std::make_shared<SimpleTorchTensorSet>(
            in->ident(), set_md,
            std::make_shared<std::vector<ITorchTensor::pointer>>(itv)
        );
    }

    return true;
}
