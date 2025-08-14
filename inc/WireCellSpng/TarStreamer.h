// tar_streamer.cpp
#include "WireCellUtil/custard/custard_file.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/*
@brief A minimal NPY loader for float32/float64 data from a tar archive.
This code reads NPY files from a tar archive, extracts the data, and converts it to
a format suitable for use in WireCell applications. It supports both little-endian and
big-endian formats, and can handle both float32 and float64 data types.
It also includes a simple JSON parser to read metadata from NPY files.

Expected file format supported by this code:
tar file containing:
- NPY files with float32 or float64 data
- JSON metadata files
- Other files (ignored)
*/

using json = nlohmann::json;

// ---------- tiny NPY-from-memory loader (supports f4 and f8 -> float) ----------
struct NpyFloat32 {
    std::vector<size_t> shape;
    std::vector<float>  data;
};


static inline bool host_is_le() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}
static inline uint16_t le16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
static inline uint32_t le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint32_t bswap32(uint32_t u) {
    return (u >> 24) | ((u >> 8) & 0x0000FF00u) | ((u << 8) & 0x00FF0000u) | (u << 24);
}
static inline uint64_t bswap64(uint64_t u) {
    u = ((u & 0x00000000FFFFFFFFULL) << 32) | ((u & 0xFFFFFFFF00000000ULL) >> 32);
    u = ((u & 0x0000FFFF0000FFFFULL) << 16) | ((u & 0xFFFF0000FFFF0000ULL) >> 16);
    u = ((u & 0x00FF00FF00FF00FFULL) <<  8) | ((u & 0xFF00FF00FF00FF00ULL) >>  8);
    return u;
}
static inline std::pair<char,int> parse_descr(std::string descr) {
    if (descr.empty()) throw std::runtime_error("NPY: empty descr");
    char endian;
    std::string type;
    if (descr[0] == '<' || descr[0] == '>' || descr[0] == '|' || descr[0] == '=') {
        endian = descr[0];
        type = descr.substr(1);
    } else {
        endian = '='; // no prefix -> native
        type = descr;
    }
    if (type == "f4") return { endian, 4 };
    if (type == "f8") return { endian, 8 };
    throw std::runtime_error("NPY: only f4/f8 supported, got descr=" + descr);
}

static NpyFloat32 load_npy_float32_from_buffer(const std::vector<char>& buf) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(buf.data());
    size_t nbytes = buf.size();
    const char magic[] = "\x93NUMPY";
    if (nbytes < 10 || std::memcmp(bytes, magic, 6) != 0) throw std::runtime_error("Not a valid NPY file");

    uint8_t major = bytes[6];
    (void)bytes[7]; // minor (unused)
    size_t hdr_len = (major == 1) ? le16(bytes + 8) :
                     (major == 2 || major == 3) ? le32(bytes + 8) :
                     throw std::runtime_error("Unsupported NPY version");
    size_t hdr_off = (major == 1) ? 10 : 12;
    if (hdr_off + hdr_len > nbytes) throw std::runtime_error("NPY: header overruns buffer");

    std::string header(reinterpret_cast<const char*>(bytes + hdr_off), hdr_len);

    std::smatch m;
    std::regex re_descr(R"('descr'\s*:\s*'([^']+)')");
    std::regex re_fortran(R"('fortran_order'\s*:\s*(True|False))");
    std::regex re_shape(R"('shape'\s*:\s*\(([^)]*)\))");
    if (!std::regex_search(header, m, re_descr))   throw std::runtime_error("NPY: missing descr");
    std::string descr = m[1];
    if (!std::regex_search(header, m, re_fortran)) throw std::runtime_error("NPY: missing fortran_order");
    bool fortran_order = (m[1] == "True");
    if (!std::regex_search(header, m, re_shape))   throw std::runtime_error("NPY: missing shape");
    std::string shape_txt = m[1];
    if (fortran_order) throw std::runtime_error("NPY: fortran_order=True not supported");

    auto [endian, itemsize] = parse_descr(descr);
    bool be_data = (endian == '>');

    std::vector<size_t> shape;
    {
        std::regex re_dim(R"(\s*(\d+)\s*,?)");
        for (auto it = std::sregex_iterator(shape_txt.begin(), shape_txt.end(), re_dim);
             it != std::sregex_iterator(); ++it) {
            shape.push_back(static_cast<size_t>(std::stoull((*it)[1])));
        }
        if (shape.empty()) shape.push_back(0);
    }
    size_t count = 1; for (auto d : shape) count *= d;

    size_t data_off = hdr_off + hdr_len;
    size_t bytes_needed = count * size_t(itemsize);
    if (data_off + bytes_needed > nbytes) throw std::runtime_error("NPY: data overruns buffer");

    NpyFloat32 out;
    out.shape = std::move(shape);
    out.data.resize(count);

    const uint8_t* raw = bytes + data_off;
    if (itemsize == 4) {
        if (be_data && host_is_le()) {
            for (size_t i = 0; i < count; ++i) {
                uint32_t u; std::memcpy(&u, raw + 4*i, 4);
                u = bswap32(u);
                std::memcpy(&out.data[i], &u, 4);
            }
        } else {
            std::memcpy(out.data.data(), raw, bytes_needed);
        }
    } else { // f8 -> convert to float
        for (size_t i = 0; i < count; ++i) {
            uint64_t u64; std::memcpy(&u64, raw + 8*i, 8);
            if (be_data && host_is_le()) u64 = bswap64(u64);
            double d; std::memcpy(&d, &u64, 8);
            out.data[i] = static_cast<float>(d);
        }
    }
    return out;
}

// ---------- TAR streamer ----------
class TarStreamer {
public:
    struct Entry {
        std::string name;
        size_t      size = 0;
    };
    
    TarStreamer() = default;
    explicit TarStreamer(const std::string& tar_path){
        open(tar_path);
    }

    void open(const std::string& tar_path) {
        if (tar_path.empty()) throw std::runtime_error("TarStreamer: empty path");
        if (tar_path_ == tar_path) return; // already opened
        tar_path_ = tar_path;
        in_.close();
        in_.open(tar_path, std::ios::binary);
        if (!in_) throw std::runtime_error("Failed to open tar: " + tar_path_);
        total_files_ = count_entries_();
    }

    //Want something that iterates and restores the iterator state
    struct bookmark{
        std::streampos pos;
        size_t files_read;
        Entry current;
    };

    bookmark bm() const{
        return bookmark{in_.tellg(), files_read_, cur_};
    };

    void restore(const bookmark& b){
        in_.clear(); // clear any eof flags
        in_.seekg(b.pos);
        files_read_ = b.files_read;
        cur_ = b.current;
    }

    // Move to next entry; returns false at end
    bool next() {
        if (!in_) return false;
        size_t fsize = file_.read_start(in_);
        if (fsize == 0) return false;              // end of archive

        cur_.name = file_.header().name();
        cur_.size = fsize;
        ++files_read_;
        return true;
    }

    // Access current entry meta
    const Entry& current() const { return cur_; }

    // Read current entry's raw bytes into memory (and advance stream to padding boundary)
    std::vector<char> read_current_bytes() {
        if (cur_.size == 0) return {};
        std::vector<char> buf(cur_.size);
        in_.read(buf.data(), cur_.size);
        // skip padding to 512-byte block
        size_t pad = (cur_.size % 512 == 0) ? 0 : (512 - cur_.size % 512);
        in_.seekg(pad, std::ios::cur);
        return buf;
    }

    // stream current entry as JSON
    json read_current_json() {
        auto buf = read_current_bytes();
        return json::parse(buf);
    }

    // stream current entry as NPY (float32/float64->float)
    NpyFloat32 read_current_npy() {
        auto buf = read_current_bytes();
        return load_npy_float32_from_buffer(buf);
    }

    // Skip current entry’s data without materializing it
    void skip_current_data() {
        // move stream by size + padding
        size_t pad = (cur_.size % 512 == 0) ? 0 : (512 - cur_.size % 512);
        in_.seekg(cur_.size + pad, std::ios::cur);
    }

    //  fetch next JSON / next NPY entry
    bool next_json() {
        while (next()) {
            if (ends_with_(cur_.name, ".json")) return true;
            skip_current_data();
        }
        return false;
    }
    bool next_npy() {
        while (next()) {
            if (ends_with_(cur_.name, ".npy")) return true;
            skip_current_data();
        }
        return false;
    }

    // Counters
    size_t total_files()      const { return total_files_; }
    size_t files_read()       const { return files_read_; }          // how many headers consumed via next/next_*
    size_t files_remaining()  const { return (total_files_ >= files_read_) ? (total_files_ - files_read_) : 0; }

private:
    // Count entries by scanning with a separate ifstream (safe; doesn’t disturb main stream)
    size_t count_entries_() const {
        std::ifstream probe(tar_path_, std::ios::binary);
        if (!probe) return 0;
        custard::File tmp;
        size_t count = 0;
        while (size_t fsize = tmp.read_start(probe)) {
            // jump over data + padding
            size_t pad = (fsize % 512 == 0) ? 0 : (512 - fsize % 512);
            probe.seekg(fsize + pad, std::ios::cur);
            ++count;
        }
        return count;
    }

    static bool ends_with_(std::string_view s, std::string_view suf) {
        return s.size() >= suf.size() && 0 == s.compare(s.size() - suf.size(), suf.size(), suf);
    }

    std::string   tar_path_;
    mutable std::ifstream in_;
    custard::File file_;
    Entry         cur_;

    size_t total_files_ = 0;
    size_t files_read_  = 0;
};
