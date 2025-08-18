#include "WireCellSpng/TorchFileSource.h"
#include "WireCellUtil/Stream.h"
#include "WireCellUtil/String.h"
#include "WireCellUtil/Dtype.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"  
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Persist.h"

WIRECELL_FACTORY(TorchFileSource, WireCell::SPNG::TorchFileSource,
                 WireCell::INamed,
                 WireCell::ITorchTensorSetSource,
                 WireCell::IConfigurable)


using namespace WireCell;
using namespace WireCell::SPNG;
using namespace WireCell::Aux;
using namespace WireCell::Stream;
using namespace WireCell::String;


bool ends_with(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

/*
Check if a json metadata file has a matching npy data file 
*/
bool has_matching_npy(TarStreamer & m_tar_streamer, const std::string& json_name)
    {
        const std::string base = json_name;

        static const std::regex re(R"(^(tensor(?:set)?)_(\d+)(?:_([0-9]+))?_metadata\.json$)",
            std::regex::ECMAScript);
        
        std::smatch m;
        if(!std::regex_match(base, m, re)) {
            return false; // Not a matching json file
        }   
        const std::string prefix = m[1]; //tensor or tensorset
        const std::string set_id = m[2]; //number 
        const bool has_idx = m[3].matched; //if has index

        if(!has_idx)return false; // We only handle indexed tensors

        const std::string idx = m[3];

        // Construct the expected npy filename
        std::string npy_name = prefix + "_" + set_id + "_" + idx + "_array.npy";

        //bookmark save the current iterator position
        auto mark = m_tar_streamer.bm();
        //first check if the next file in the stream is npy
        if(m_tar_streamer.next_npy()) {
            auto entry = m_tar_streamer.current();
            if(ends_with(entry.name, npy_name)) {
                m_tar_streamer.restore(mark); //restore the iterator position
                return true; // Found a matching npy file
            }
            else {
                return false;
            }
        }
        else{
            return false; // No npy file found
        }
    m_tar_streamer.restore(mark); //restore the iterator position
    return false;
}


// Split into colleted_type, filter_name, filter_type
std::vector<std::string> return_taginfo(const std::string&str) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ':')) {
        tokens.push_back(item);
}
return tokens; 
}

TorchFileSource::TorchFileSource()
    : Aux::Logger("TorchFileSource", "spng")
{
}

TorchFileSource::~TorchFileSource()
{
}


WireCell::Configuration TorchFileSource::default_configuration() const
{
    Configuration cfg;
    cfg["inname"] = m_inname;
    cfg["prefix"] = m_prefix;
    cfg["tag"] = m_tag; // tag to filter entries by

    return cfg;
}

void TorchFileSource::configure(const WireCell::Configuration& cfg)
{
    m_inname = get(cfg, "inname", m_inname);
    m_prefix = get(cfg, "prefix", m_prefix);
    m_tag = get(cfg, "tag", m_tag);
    m_tar_streamer.open(m_inname);


}

void TorchFileSource::finalize(){}

void TorchFileSource::clear(){}

bool TorchFileSource::operator()(output_pointer &out)
{
    
    auto tot_files = m_tar_streamer.total_files();
    std::cout<<"Total entries: " << tot_files << "\n";
    std::vector<torch::Tensor> tensors;
    if (tot_files == 0) {
        log->debug("No entries in tar file: {}", m_inname);
        return false; // if empty, do not send anything
    }
    tensors.reserve(tot_files/2); 

    while(m_tar_streamer.next()) {
        auto entry = m_tar_streamer.current();
        log->debug("Processing entry: {} of size: {}", entry.name, entry.size);

        if(ends_with(entry.name, ".json")) {
            // Read metadata
            auto json_meta = m_tar_streamer.read_current_json();
            log->debug("Read JSON metadata: {}", json_meta.dump(2));
            std::string j_name = entry.name;
            if(has_matching_npy(m_tar_streamer, j_name)) {
                log->debug("Found matching npy file for {}", j_name);
                //if the current is json, read the next npy file
                if(!m_tar_streamer.next_npy()) {
                    log->debug("No matching npy file found for {}", j_name);
                    continue;
                }
                std::string npy_name = m_tar_streamer.current().name;
                log->debug("Reading npy file for {} {}", npy_name, j_name);
                auto tensor_data = m_tar_streamer.read_current_npy();
                // Convert to torch tensor
                std::vector<int64_t> shape64(tensor_data.shape.begin(), tensor_data.shape.end());
                auto tensor = torch::from_blob(tensor_data.data.data(), shape64, torch::kFloat32).clone();
                tensors.push_back(tensor);
                log->debug("Read tensor with shape: [{}]", tensor_data.shape.size());
            }
        }
        else if(ends_with(entry.name, ".npy")) {
            log->debug("Reading npy Entry {}", entry.name);
            m_tar_streamer.skip_current_data(); // skip unknown entries
        }
        else {
            log->debug("Skipping unknown entry type: {}", entry.name);
            m_tar_streamer.skip_current_data(); // skip unknown entries
        }
    }
    ITorchTensor::shared_vector tensor_ptrs;
    std::vector<ITorchTensor::pointer> tmp_vec;
    for (const auto& t : tensors) {
        tmp_vec.emplace_back(std::make_shared<SimpleTorchTensor>(t));
    }
    log->debug("Read {} tensors from tar file", tmp_vec.size());
    if (tmp_vec.empty()) {
        log->debug("No tensors found in tar file: {}", m_inname);
        return false; // if empty, do not send anything
    }
    tensor_ptrs = std::make_shared<std::vector<ITorchTensor::pointer>>(std::move(tmp_vec));
    out = std::make_shared<SimpleTorchTensorSet>(0, Json::nullValue, tensor_ptrs);
    
    return true; // if empty, pass EOS
}
