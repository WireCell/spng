/*

  Apples-to-apples performance comparison between ArrayFire and LibTorch for
  bottleneck operations identified from OmnibusSigProc.

  It is built in a stand-alone manner:

  

 */

// This main program is really a couple libraries all smashed together.  Each
// array technology is kept in an #ifdef/#endif and the testing scaffolding is
// type independent.
#define USE_ARRAYFIRE
#define USE_LIBTORCH


// A small subset of technology-independent array API which are specialize for a
// given array technology below.
template<typename T> T fft2(const T& arr);
template<typename T> T ifft2(const T& arr);
template<typename T> T median(const T& arr);
template<typename T> T quantile(const T& arr, const T& q);


#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using time_point = std::chrono::high_resolution_clock::time_point;

time_point now()
{
    return std::chrono::high_resolution_clock::now();
}
double tdiffms(time_point t)
{
    return std::chrono::duration<double, std::milli>(now() - t).count();
}

struct Perf {
    using size_type = long long;
    using shape_type = std::vector<size_type>;

    json data;

    std::default_random_engine rng;
    std::uniform_real_distribution<float> runiform;

    Perf()
        : runiform(0,1)
        {}

    virtual void add(const std::string& name, const float* data, const shape_type& shape) = 0;
    virtual void init(const json& cfg) {
        data = cfg;
        int seed = get_update("seed", 12345);
    };
    virtual void run() = 0;

    template<typename T>
    T get_update(const std::string& key, T def) {
        auto ret = data.value<T>(key, def);
        data[key] = ret;
        return ret;
    }

    std::unique_ptr<float[]> randu(size_t n) {
        auto dat = std::make_unique<float[]>(n);
        for (size_t ind=0; ind<n; ++ind) {
            dat[ind] = runiform(rng);
        }
        return dat;
    }

    void add_array(const std::string& name, const json& cfg) {
        std::cerr << "add_array " << name << " " << cfg << "\n";
        auto jshape = cfg["shape"];
        size_t ndim = jshape.size();
        std::cerr << "jshape=" << jshape << " ndim=" << ndim << "\n";
        shape_type shape(ndim);
        size_t nele = 1;
        for (size_t ind=0; ind<ndim; ++ind) {
            size_t dim = jshape[ind];
            shape[ind] = dim;
            std::cerr << ind << " " << dim << "\n";
            nele *= dim;
        }
        std::cerr << "add_array " << name << " " << cfg << " nele=" << nele << "\n";

        auto data = randu(nele);
        add(name, data.get(), shape);
    }


};


// A "bag of arrays" and test.
template<typename T>
struct PerfT : public Perf {

    std::unordered_map<std::string, T> arrays;

    json run_convo(const json& cfg) {
        std::cerr << "run_convo: " << cfg << "\n";
        size_t repeat = cfg["repeat"];
        std::string sname = cfg["signal"];
        std::string rname = cfg["response"];
        std::cerr << "repeat:" << repeat << " sname:"<<sname << " rname:" <<rname << "\n";
        const T& s = arrays[sname];
        const T& r = arrays[rname];

        auto R = fft2(r);

        auto t = now();
        while (repeat--) {
            std::cerr << "convolve: " << repeat << " " << tdiffms(t) << "\n";
            auto S = ifft2(fft2(s) * R);
        }

        json results;
        results["time"] = tdiffms(t);
        return results;
    }
    json run_median(const json& cfg) {
        json results;
        return results;
    }
    json run_quantile(const json& cfg) {
        json results;
        return results;
    }

    virtual void run() {
        auto arrays = data["arrays"];
        for (auto& [name, cfg] : arrays.items()) {
            add_array(name, cfg);
        }
        

        auto& tests = data["tests"];
        size_t ntests = tests.size();
        for (size_t itest=0; itest<ntests; ++itest) {
            auto& test = tests[itest];
            std::string kind = test["kind"];

            if (kind == "convo") {
                tests[itest]["results"] = run_convo(test);
                continue;
            }
            if (kind == "median") {
                tests[itest]["results"] = run_median(test);
                continue;
            }
            if (kind == "quantile") {
                tests[itest]["results"] = run_quantile(test);
                continue;
            }
            std::cerr << "unknown test: " << test << "\n";
        }
        
    }
    
};


#ifdef USE_ARRAYFIRE
#include <arrayfire.h>

struct PerfAF : public PerfT<af::array> {
    
    virtual void init(const json& cfg) {
        Perf::init(cfg);
        if (get_update<bool>("gpu", false)) {
            af::setBackend(AF_BACKEND_CUDA);
        }
        else {
            af::setBackend(AF_BACKEND_CPU);            
        }
    }

    virtual void add(const std::string& name, const float* data, const shape_type& shape) {
        af::dim4 dims(shape.size(), shape.data());
        arrays[name] = af::array(dims, data);
    }

};



template<>
af::array fft2<af::array>(const af::array& arr)
{
    return af::fft2(arr);
}
template<>
af::array ifft2<af::array>(const af::array& arr)
{
    return af::ifft2(arr);
}
template<>
af::array median<af::array>(const af::array& arr)
{
    return af::median(arr);
}
    
// ArrayFire apparently does not provide quantile().  This is a nominal
// implementation but assumes 1D array.
template<>
af::array quantile<af::array>(const af::array& arr, const af::array& q)
{
    af::array ret(q.dims(), q.type());
    auto sorted = af::sort(arr);
    
    auto N = q.dims(0);
    gfor(af::seq ind, N) {
        ret(ind) = sorted(q(ind)*N);
    }
    return ret;
}
#endif

#ifdef USE_LIBTORCH
#include "torch/torch.h"

struct PerfLT : public PerfT<torch::Tensor> {

    torch::TensorOptions to;

    virtual void init(const json& cfg) {
        Perf::init(cfg);
        if (get_update<bool>("gpu", false)) {
            to = to.device(torch::kCUDA);
        }
    }

    virtual void add(const std::string& name, const float* data, const shape_type& shape) {
        std::vector<int64_t> tshape(shape.begin(), shape.end());
        arrays[name] = torch::from_blob((void*)data, tshape, to);
    }
    
};

template<>
torch::Tensor fft2<torch::Tensor>(const torch::Tensor& ten)
{
    return torch::fft::fft2(ten);
}
template<>
torch::Tensor ifft2<torch::Tensor>(const torch::Tensor& ten)
{
    return torch::fft::ifft2(ten);
}
template<>
torch::Tensor median<torch::Tensor>(const torch::Tensor& ten)
{
    return ten.median();
}
template<>
torch::Tensor quantile<torch::Tensor>(const torch::Tensor& ten, const torch::Tensor& q)
{
    return ten.quantile(q);
}

#endif


void die(const std::string& msg) {
    std::cerr << msg << "\n";
    exit (1);
}


int test_af(int argc, char* argv[])
{
    af::array aa;
    return 0;
}
int test_lt(int argc, char* argv[])
{
    torch::Tensor tt;
    return 0;
}

json perf(const json& cfg)
{
    std::string tech = cfg.value("tech", "af");
    std::unique_ptr<Perf> p;

    if (tech == "af") {
        p = std::make_unique<PerfAF>();
    }
    if (tech == "lt") {
        p = std::make_unique<PerfLT>();
    }
    if (!p) {
        json empty;
        return empty;
    }
    p->init(cfg);
    p->run();
    return p->data;
}

int main(int argc, char* argv[])
{
    std::string iname = "/dev/stdin";
    std::string oname = "/dev/stdout";
    if (argc > 1) {
        iname = argv[1];
        if (iname == "-") iname = "/dev/stdin";
    }
    if (argc > 2) {
        oname = argv[2];
        if (oname == "-") oname = "/dev/stdout";
    }
        
    std::ifstream ifile(iname);
    auto cfg = json::parse(ifile);

    auto res = perf(cfg);

    std::ofstream ofile(oname);
    ofile << std::setw(4) << res << std::endl;
    return 0;    
}
