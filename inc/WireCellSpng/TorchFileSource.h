#ifndef WIRECELL_SPNG_TORCHFILESOURCE
#define WIRECELL_SPNG_TORCHFILESOURCE

#include "WireCellSpng/ITorchTensorSetSource.h"
#include "WireCellSpng/TarStreamer.h"
#include "WireCellIface/ITerminal.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellAux/Logger.h"
#include "WireCellUtil/Stream.h"

namespace WireCell::SPNG {

class TorchFileSource : public Aux::Logger, public ITorchTensorSetSource,
                        public WireCell::IConfigurable, public WireCell::ITerminal
{
public:
    TorchFileSource();
    virtual ~TorchFileSource();

    // IConfigurable
    virtual WireCell::Configuration default_configuration() const override;
    virtual void configure(const WireCell::Configuration& config) override;

    // ITerminal
    virtual void finalize() override;

    // ITorchTensorSetSource
    virtual bool operator()(output_pointer &out) override;


private:
    using istream_t = boost::iostreams::filtering_istream;
    std::string m_inname{""};  // default to stdin
    std::string m_prefix{""};
    std::string m_tag{""}; // tag to filter entries by
    istream_t m_in; // 
    std::string m_model_path{""}; // path to the TorchScript model
    std::ifstream tarfile;
    TarStreamer m_tar_streamer;
    //torch::jit::script::Module m_module;



    bool read_head();
    void clear();



    //ITorchTensorSet:pointer read_tensor(const std::string& fname);
    nlohmann::json read_metadata(const std::string& fname); 
    torch::Tensor read_tensor_data(const std::string& fname);
};
}  // namespace WireCell::SPNG

#endif // WIRECELL_SPNG_TORCHFILESOURCE
