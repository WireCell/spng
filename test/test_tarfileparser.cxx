/*
Check if a json metadata file has a matching npy data file 
*/

#include "WireCellSpng/TarStreamer.h"

/*
Check if a json metadata file has a matching npy data file 
*/

bool ends_with(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

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

int main(int argc, char** argv){
    TarStreamer ts(argv[1]);
    std::cout << "Total entries: " << ts.total_files() << "\n";
    
    while(ts.next()) {
        const auto &entry = ts.current();
        std::cout << "Entry: " << entry.name << " (" << entry.size << " bytes)\n";
        if(ends_with(entry.name, ".json")) {
            auto json_meta = ts.read_current_json();
            std::cout << "JSON metadata: " << json_meta.dump(2) << "\n";
            std::string j_name = entry.name;
            if(has_matching_npy(ts, j_name)) {
                std::cout << "Found matching npy file for " << j_name << "\n";
                //if the current is json, read the next npy file
                if(!ts.next_npy()) {
                    std::cout << "No matching npy file found for " << j_name << "\n";
                    continue;
                }
                std::string npy_name = ts.current().name;
                std::cout << "Reading npy file for " << npy_name <<" "<<j_name<< "\n";
                auto tensor_data = ts.read_current_npy();
                //print first 5 entries 
                for(size_t i = 0; i < std::min<size_t>(5, tensor_data.data.size()); ++i) {
                    std::cout << tensor_data.data[i] << " ";
                }
                std::cout << "\n";
            }
            else {
                std::cout << "No matching npy file for " << entry.name << "\n";
            }
            
        }
        else if(ends_with(entry.name, ".npy")) {
            std::cout<<" Reading npy Entry "<< entry.name << "\n";
            ts.skip_current_data();
        }
        else{
            std::cout<<"Unknow Format "<<std::endl;
            ts.skip_current_data();
        }
        std::cout<<"End of entry: " << entry.name <<" total files read "<< ts.files_read() << " remaining " << ts.files_remaining() << "\n"; 
    }
    return 0;
}