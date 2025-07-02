#include "WireCellSpng/TorchService.h"
#include "WireCellSpng/Util.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/String.h"
#include "WireCellUtil/Persist.h"

//do we rely on openMP...probably not..
//#include <omp.h>
#include <ATen/Parallel.h>

WIRECELL_FACTORY(TorchService, 
                 WireCell::SPNG::TorchService,
                 WireCell::SPNG::ITorchForward, // interface 1
                 WireCell::IConfigurable)

using namespace WireCell;



SPNG::TorchService::TorchService()
    : Aux::Logger("TorchService", "torch")
{

}

Configuration SPNG::TorchService::default_configuration() const
{
    Configuration cfg;

    // TorchScript model
    cfg["model"] = "model.ts";

    // one of: {cpu, gpu, gpuN} where "N" is a GPU number.  "gpu"
    // alone will use GPU 0.
    cfg["device"] = "gpu";
    
    return cfg;
}

void SPNG::TorchService::configure(const WireCell::Configuration& cfg)
{
    //Should be gpus
    auto dev = get<std::string>(cfg, "device", "gpu");
    m_ctx.connect(dev);

    auto model_path = Persist::resolve(cfg["model"].asString());
    if (model_path.empty()) {
        log->critical("no TorchScript model file provided");
        THROW(ValueError() << errmsg{"no TorchScript model file provided"});
    }

    // Use almost 1/2 the memory and 3/4 the time.
    // torch::NoGradGuard no_grad; //Why is this here?

    try {
        m_module = torch::jit::load(model_path, m_ctx.device());
    }
    catch (const c10::Error& e) {
        log->critical("error loading model: \"{}\" to device \"{}\": {}",
                      model_path, dev, e.what());
        throw;                  // rethrow
    }

    log->debug("loaded model \"{}\" to device \"{}\"",
               model_path, m_ctx.devname());
}

ITorchTensorSet::pointer SPNG::TorchService::forward(const ITorchTensorSet::pointer& in) const
{
    log->debug("TorchService::forward function entered");

    if (!in) {
        log->critical("TorchService::forward received a null input pointer");
        THROW(ValueError() << errmsg{"TorchService::forward received a null input pointer"});
    }

    try {
        log->debug("TorchService::forward called with input: {}", in->ident());
    }
    catch (const std::exception& e) {
        log->critical("Exception while accessing in->ident(): {}", e.what());
        THROW(ValueError() << errmsg{"Exception while accessing in->ident()"} <<
                            errmsg{" " + std::string(e.what())});
    }


    std::vector<torch::IValue> iival = SPNG::from_itensor(in, m_ctx.is_gpu());

    torch::IValue oival;

    try {
        oival = m_module.forward(iival);
    }
    catch (const std::runtime_error& err) {
        log->critical("Error running model on device: {}", err.what());
        THROW(ValueError() << errmsg{"error running model on device"} <<
                            errmsg{" " + std::string(err.what())});
    }

    ITorchTensorSet::pointer ret = SPNG::to_itensor({oival});
    return ret;
}
