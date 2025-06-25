#include "WireCellSpng/ROITests.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/SimpleTorchTensor.h"
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
    //Process each tensor in the input set
    // Process each tensor in the input set
    log->debug("ROITests: Processing {} tensors", tensors->size());
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

        //scale arrays with the input scales



    }
    out = std::make_shared<SimpleTorchTensorSet>(
        in->ident(), in->metadata(), tensors
    );
    return true;
}