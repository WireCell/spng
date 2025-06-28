#include "WireCellSpng/ROITests.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellSpng/Util.h"
#include "WireCellUtil/NamedFactory.h"

#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/ITrace.h"
#include "WireCellAux/PlaneTools.h"


//register this (ROITests) as a factory
WIRECELL_FACTORY(SPNGROITests,// name of the factory
    WireCell::SPNG::ROITests, // name of the class
    WireCell::INamed, // name of the interface 1 (allows object to have unique name)
    WireCell::ITorchTensorSetFilter, // interface 2 (process ITorchTensorSet)
    //WireCell::SPNG::ITorchForward, //interface 2 (process ITorchForward)
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
   m_cfg.apa = get(cfg, "apa",m_cfg.apa);
   m_cfg.plane = get(cfg, "plane", m_cfg.plane); 

   // Is it implemented already>
   /*
   auto apa = Factory::find_tn<IAnodePlane>(m_cfg.apa);
   auto ichans = Aux::plane_channels(apa,m_cfg.plane); //aux/src/PlaneTools.cxx

   //channel information

   for(const auto & ichan : ichans) {
    auto chid = ichan->ident();
    m_chset.insert(chid);
    m_chlist.push_back(chid);
   }

   //sort the channels
   m_cfg.sort_chanids = get(cfg, "sort_chanids", m_cfg.sort_chanids);
   if(m_cfg.sort_chanids) {
       std::sort(m_chlist.begin(), m_chlist.end());
   }
    */
   m_cfg.input_scale = get(cfg, "input_scale", m_cfg.input_scale);
   m_cfg.input_offset = get(cfg, "input_offset", m_cfg.input_offset);
   m_cfg.output_scale = get(cfg, "output_scale", m_cfg.output_scale);
   m_cfg.output_offset = get(cfg, "output_offset", m_cfg.output_offset);
   if (m_cfg.output_scale != 1.0) {
       log->debug("using output scale: {}", m_cfg.output_scale);
   }
   m_is_gpu = get(cfg, "is_gpu", m_is_gpu);

   //TODO AB: Other configuration parameters that needs to be forwarded to the TorchService
   

}

void ROITests::finalize()
{

}

bool ROITests::operator()(const input_pointer& in, output_pointer& out)
{
    out = nullptr;
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
    //TimeKeeper tk(fmt::format("call={}",m_save_count));
    //Process Each TorchTensors in input set to torch::Tensor
    std::vector<torch::Tensor> ch_tensors;
    log->debug("ROITests: Processing {} tensors", tensors->size());
    for (size_t i = 0; i < tensors->size(); ++i) {
        auto tensor = tensors->at(i)->tensor();
        // process tensor to torch_tensor to write to ch_tensors
        

        // Check the tensor tags (metadata)
        auto metadata = tensors->at(i)->metadata();
        if (metadata.isMember("tag")) {
            log->debug("ROITests: Found tag in metadata: {}", metadata["tag"].asString());
        } else {
            log->warn("ROITests: No tag found in metadata for tensor {}", i);
        }
        torch::Tensor scaled = tensor*m_cfg.input_scale + m_cfg.input_offset;
        //  ch_eigen.push_back(Array::downsample(arr, m_cfg.tick_per_slice, 1));
        auto tick_per_slice = m_cfg.tick_per_slice;
        auto nticks = tensor.size(1); //0 is channel, 1 is time
        int nticks_ds = nticks / tick_per_slice;
        //keep all the dimensions unchanged, select range from 0, nticks_ds* tick_per_slice
        //This will downsample the tensor by tick_per_slice
        auto trimmed = tensor.index({"...",torch::indexing::Slice(0, nticks_ds * tick_per_slice)}); 
        //now reshape the tensor
        auto reshaped = trimmed.view({tensor.size(0), nticks_ds, tick_per_slice});
        //reshaped tensor has the dimensions [channels, downsampled_time, tick_per_slice]
        //now take the mean along the last dimension (tick_per_slice)
        auto downsampled = reshaped.mean(2);
        //now cscale and offset the downsampled tensor
        auto dscaled = downsampled * m_cfg.input_scale + m_cfg.input_offset;
        ch_tensors.push_back(dscaled);

    }
    //stack vector of tensors along a new dimension (0)
    auto img = torch::stack(ch_tensors, 0); //stack all the tensors along a new dimension (0)
    //img is now a 3D stacked tensors 
    auto transposed = img.transpose(1,2); //transposition into ntags, nticks_ds, nchannels

    auto batch = torch::unsqueeze(transposed, 0); //add a batch dimension at the start
    

    auto chunks = batch.chunk(m_cfg.nchunks, 2); // chunk the batch into m_cfg.nchunks along the time dimension (2)
    //print the shape of a chunk
    log->debug("ROITests: Chunk shape: {}", chunks[0].sizes()[1]);

    std::vector<torch::Tensor> outputs;
    
    for (auto& chunk : chunks) {
        log->debug("ROITests: Chunk shape: {}", chunk.sizes()[1]);
        std::vector<torch::IValue> inputs = {chunk};
        auto iitens = to_itensor(inputs); // convert inputs to ITorchTensorSet
        auto oitens = m_forward->forward(iitens);
        //torch::Tensor out_chunk = oitens.toTensor().to(torch::kCUDA); // keep the data in gpu if needed for other stuff.
        torch::Tensor out_chunk = from_itensor(oitens, m_is_gpu)[0].toTensor().cpu(); // convert ITorchTensorSet to torch::Tensor
        outputs.push_back(out_chunk);
    }
    
    torch::Tensor out_tensor = torch::cat(outputs, 2); // concatenate the output chunks along the time dimension (2)

    auto mask = out_tensor.gt(m_cfg.mask_thresh);
    auto finalized_tensor = out_tensor*mask.to(out_tensor.dtype()); // apply the mask to the output tensor  
    finalized_tensor = finalized_tensor * m_cfg.output_scale + m_cfg.output_offset; // scale and offset the output tensor
    
    //std::vector<ITorchTensor::pointer>processed_tensors;
    auto shared_vec = std::make_shared<ITorchTensor::vector>();
    for(int i = 0; i < finalized_tensor.size(0); ++i) {
        auto single_tensor = finalized_tensor[i].detach().clone(); // detach and clone the tensor to avoid modifying the original tensor
        auto meta = tensors->at(i)->metadata(); // get the metadata from the original tensors
        shared_vec->push_back(
            std::make_shared<SimpleTorchTensor>(single_tensor,meta));
    }
      
    out = std::make_shared<SimpleTorchTensorSet>(
        in->ident(), in->metadata(), shared_vec
    );
    
    return true;
}