// test custard.hpp
//
// Beware, the code here would NOT be ideal for a real app 

#include "WireCellUtil/custard/custard_file.hpp"
#include "WireCellUtil/custard/pigenc.hpp"
#include "WireCellUtil/cnpy.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <string>

#include <cstdint>
#include <regex>
#include <stdexcept>

struct NpyFloat32 {
    std::vector<size_t> shape;
    std::vector<float>  data;
};

// --- helpers for endian + swap ---
static inline bool is_little_endian_host() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}
static inline uint16_t le16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
static inline uint32_t le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static inline uint32_t bswap32(uint32_t u) {
    return (u >> 24) | ((u >> 8) & 0x0000FF00u) |
           ((u << 8) & 0x00FF0000u) | (u << 24);
}
static inline uint64_t bswap64(uint64_t u) {
    u = ((u & 0x00000000FFFFFFFFULL) << 32) | ((u & 0xFFFFFFFF00000000ULL) >> 32);
    u = ((u & 0x0000FFFF0000FFFFULL) << 16) | ((u & 0xFFFF0000FFFF0000ULL) >> 16);
    u = ((u & 0x00FF00FF00FF00FFULL) <<  8) | ((u & 0xFF00FF00FF00FF00ULL) >>  8);
    return u;
}

// Return itemsize (4 or 8) and endianness char ('<','>','|','=')
static inline std::pair<char,int> parse_descr(std::string descr) {
    if (descr.empty()) throw std::runtime_error("NPY: empty descr");
    char endian;
    std::string type;

    // Accept optional endian prefix. If missing, assume native ('=').
    if (descr[0] == '<' || descr[0] == '>' || descr[0] == '|' || descr[0] == '=') {
        endian = descr[0];
        type = descr.substr(1);
    } else {
        endian = '=';                // assume native endian if not provided
        type = descr;                // e.g., "f8"
    }

    if (type == "f4") return { endian, 4 };
    if (type == "f8") return { endian, 8 };

    throw std::runtime_error("NPY: only f4/f8 supported, got descr=" + descr);
}

NpyFloat32 load_npy_float32_from_buffer(const std::vector<char>& buf) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(buf.data());
    size_t nbytes = buf.size();

    // magic + version
    const char magic[] = "\x93NUMPY";
    if (nbytes < 10 || std::memcmp(bytes, magic, 6) != 0)
        throw std::runtime_error("Not a valid NPY file");

    uint8_t major = bytes[6];
    [[maybe_unused]] uint8_t minor = bytes[7];  // not used but kept for clarity
    size_t header_len = (major == 1) ? le16(bytes + 8) :
                        (major == 2 || major == 3) ? le32(bytes + 8) :
                        throw std::runtime_error("Unsupported NPY version");
    size_t hdr_off = (major == 1) ? 10 : 12;

    if (hdr_off + header_len > nbytes)
        throw std::runtime_error("Header overruns buffer");

    std::string header(reinterpret_cast<const char*>(bytes + hdr_off), header_len);

    // parse header (simple regex is fine for standard numpy headers)
    std::smatch m;
    std::regex re_descr(R"('descr'\s*:\s*'([^']+)')");
    std::regex re_fortran(R"('fortran_order'\s*:\s*(True|False))");
    std::regex re_shape(R"('shape'\s*:\s*\(([^)]*)\))");
    if (!std::regex_search(header, m, re_descr))   throw std::runtime_error("Missing descr in header");
    std::string descr = m[1];
    if (!std::regex_search(header, m, re_fortran)) throw std::runtime_error("Missing fortran_order in header");
    bool fortran_order = (m[1] == "True");
    if (!std::regex_search(header, m, re_shape))   throw std::runtime_error("Missing shape in header");
    std::string shape_txt = m[1];

    if (fortran_order)
        throw std::runtime_error("Fortran order not supported by this minimal loader");

    auto [endian, itemsize] = parse_descr(descr);  // supports f4/f8
    bool need_le = (endian == '<') || (endian == '='); // '=' = native-endian; assume little-endian host
    bool is_be   = (endian == '>');

    // shape
    std::vector<size_t> shape;
    {
        std::regex re_dim(R"(\s*(\d+)\s*,?)");
        for (auto it = std::sregex_iterator(shape_txt.begin(), shape_txt.end(), re_dim);
             it != std::sregex_iterator(); ++it) {
            shape.push_back(static_cast<size_t>(std::stoull((*it)[1])));
        }
        if (shape.empty()) shape.push_back(0);
    }
    size_t count = 1;
    for (auto d : shape) count *= d;

    size_t data_off = hdr_off + header_len;
    size_t bytes_needed = count * size_t(itemsize);
    if (data_off + bytes_needed > nbytes)
        throw std::runtime_error("Data overruns buffer");

    NpyFloat32 out;
    out.shape = std::move(shape);
    out.data.resize(count);

    const uint8_t* raw = bytes + data_off;

    if (itemsize == 4) {
        // float32 path
        if (is_be && is_little_endian_host()) {
            // swap each u32 then copy as float
            for (size_t i = 0; i < count; ++i) {
                uint32_t u;
                std::memcpy(&u, raw + 4*i, 4);
                u = bswap32(u);
                std::memcpy(&out.data[i], &u, 4);
            }
        } else {
            std::memcpy(out.data.data(), raw, bytes_needed);
        }
    } else { // itemsize == 8 (float64)
        // read doubles, swap if necessary, then convert to float
        for (size_t i = 0; i < count; ++i) {
            uint64_t u64;
            std::memcpy(&u64, raw + 8*i, 8);
            if (is_be && is_little_endian_host()) u64 = bswap64(u64);
            double d;
            std::memcpy(&d, &u64, 8);
            out.data[i] = static_cast<float>(d);
        }
    }
    return out;
}


bool ends_with(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


int unpack(std::string archive)
{
    std::cerr << "unpacking: " << archive << std::endl;
    std::ifstream fi(archive);
    while (fi) {
        custard::Header head;
        fi.read(head.as_bytes(), 512);
        if (fi.eof()) {
            return 0;
        }
        assert (fi);

        if (! head.size()) {
            std::cerr << "skipping empty\n";
            continue;
        }

        std::cerr << head.name() << "\n"
                  << "stored check sum: " << head.chksum() << "\n"
                  << "  calculated sum: " << head.checksum() << "\n"
                  << "       file size: " << head.size() 
                  << "\n";

        assert(head.chksum() == head.checksum());
        
        std::string path = head.name();
        std::cerr << archive << " -> " << path << std::endl;
        while (path[0] == '/') {
            // At leat pretend to be a little secure
            path.erase(path.begin());
        }
        std::ofstream fo(path);
        assert (fo);
        
        // This is NOT smart on large files!
        std::string buf(head.size(), 0);
        fi.read((char*)buf.data(), buf.size());
        assert (fi);
        fo.write(buf.data(), buf.size());
        assert (fo);

        // get past padding
        size_t npad = head.padding();
        std::cerr << head.name() << " skipping " << npad << " after " << head.size() << std::endl;
        fi.seekg(npad, fi.cur);
        assert (fi);

    }
    return 0;
}

int pack(std::string archive, int nmembers, char* member[])
{
    std::ofstream fo(archive);
    for (int ind = 0; ind<nmembers; ++ind) {
        std::string path(member[ind]);

        std::ifstream fi(path, std::ifstream::ate | std::ifstream::binary);        
        assert (fi);

        auto siz = fi.tellg();
        fi.seekg(0);
        assert (fi);

        // note, real tar preserves mtime, uid, etc.
        custard::Header head(path, siz);

        fo.write(head.as_bytes(), 512);
        assert (fo);

        std::string buf(head.size(), 0);
        fi.read((char*)buf.data(), buf.size());
        assert (fi);

        fo.write(buf.data(), buf.size());
        assert (fo);

        size_t npad = 512 - head.size() % 512;
        std::string pad(npad, 0);
        fo.write(pad.data(), pad.size());
        assert (fo);
    }
    return 0;
}

int read_file(std::string archive){
    std::ifstream tarfile(archive, std::ifstream::binary);
    custard::File custard_tar;
    while(size_t fsize = custard_tar.read_start(tarfile)){
        std::string fname = custard_tar.header().name();
        //std::cerr << "Reading file: " << fname << " of size: " << fsize << std::endl;
        //now place to store the buffer
        //std::vector<char>buff(fsize);
        //tarfile.read(buff.data(), fsize);
        if(ends_with(fname,".json")){
            std::cout<<"Reading JSON file: " << fname << std::endl;
            std::vector<char>buff(fsize);
            tarfile.read(buff.data(), fsize);
            //parse the json
            nlohmann::json j = nlohmann::json::parse(buff);
            std::cout << "Parsed JSON: " << j.dump(4) << std::endl;
        }
        
        else if(ends_with(fname,".npy")){
            //std::ifstream infile(fname, std::ios::binary);
            std::cout<<"Reading NPY file: " << fname << std::endl;
            std::vector<char> buff(fsize);
            tarfile.read(buff.data(), fsize);
            auto arr = load_npy_float32_from_buffer(buff);
            //cnpy::NpyArray arr = cnpy::npy_load(fname);
            std::cout << "Numpy array shape: ";
            for(auto s : arr.shape) std::cout << s << " "<< std::endl;
            size_t num_elements = 1;
            for(size_t s : arr.shape) num_elements *= s;
            std::cout << "Number of elements: " << num_elements << std::endl;
            std::cout<< "First 10 elements: ";
            for(size_t i = 0; i < std::min(num_elements, size_t(10)); ++i) {
                std::cout << arr.data[i] << " ";
            } std::cout << std::endl;
            
        }
            
        else if(ends_with(fname,".tar")){
            std::cerr << "Extracting tar file: " << fname << std::endl;
            unpack(fname);
        }
        else if(ends_with(fname,".custard")){
            std::cerr << "Extracting custard file: " << fname << std::endl;
            unpack(fname);
        }
        else {
            std::cerr << "Unknown file type: " << fname << std::endl;
        }
        size_t pad =(fsize % 512 == 0) ? 0 : (512 - fsize % 512);
        tarfile.seekg(pad, std::ios::cur);
    }
    return 0;
}
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " archive.tar [file ...]" << std::endl;
        std::cerr << " with no files, extract archive.tar, otherwise produce it\n";
        return -1;
    }

    if (argc == 2) {
        return read_file(argv[1]);
    }

    return pack(argv[1], argc-2, argv+2);
}
