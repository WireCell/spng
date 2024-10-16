/*

  Apples-to-apples performance comparison between ArrayFire and LibTorch for
  bottleneck operations identified from OmnibusSigProc.

  It is built in a stand-alone manner with a script:

    ./spng/test/build-afvslt

  Run all tests like:

    jsonnet spng/test/afvslt.jsonnet | OMP_NUM_THREADS=1 ./afvslt

  Run narrowed tests:

    jsonnet spng/test/afvslt.jsonnet -A tests=convo -A devices=gpu   -A techs=af,lt | OMP_NUM_THREADS=1 ./afvslt

  To add a new benchmark:

  0) Add Perf::run_<name>() ABC.
  1) Add generic templated implementation PerfT::run_<name>().
  2) If needed, extend the generic API in namespace "vs" and AF+LT imp.
  3) Extend the factory in perf() to call run_<name>() given a "kind" of "name".
  4) Extend afvslt.jsonnet to include config for the new "name"

 */

// This main program is really a couple libraries all smashed together.  Each
// array technology is kept in an #ifdef/#endif and the testing scaffolding is
// type independent.
#define USE_ARRAYFIRE
#define USE_LIBTORCH


// A small subset of technology-independent array API which are specialize for a
// given array technology below.
namespace vs {
    template<typename T> float to_float(const T& arr);

    template<typename T> T max(const T& arr);
    template<typename T> T min(const T& arr);
    template<typename T> T real(const T& arr);
    template<typename T> T imag(const T& arr);
    template<typename T> T fft2(const T& arr);
    template<typename T> T ifft2(const T& arr);
    template<typename T> T median(const T& arr, int dim=0);
    template<typename T> T sort(const T& arr, int dim=0);

    // Call to finish calculation (for lazy libs like AF).
    template<typename T> T& eval(T& arr);
    // Finish all lazy calculations
    template<typename T> void sync();
}

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

// Get a value of a key in an object, setting it to default if missing.
template<typename T> T getdef(json& jobj, const std::string& key, const T& def) {
    auto ret = jobj.value<T>(key, def);
    jobj[key] = ret;
    return ret;
}
    
using size_type = long long;
using shape_type = std::vector<size_type>;

size_type shape_size(const shape_type& shape)
{
    size_t nele = 1;
    for (auto& dim : shape) {
        nele *= dim;
    }
    return nele;
}

shape_type to_shape(const json& jshape)
{
    shape_type shape;
    for (const auto& ele : jshape) {
        size_t size = ele;
        shape.push_back(size);
    }
    return shape;
}

// Partially abstract base class for use by templated mid class with
// implementations by a concrete base class.
struct Perf {

    std::default_random_engine rng;
    std::uniform_real_distribution<float> runiform;

    Perf(long unsigned int seed = 12345)
        : rng{seed}
        , runiform(0,1)
        {}

    // Template class can call for one-time init prior to a run_*().  
    // Must return cfg, possibly after modification.
    virtual json prerun(json tst) { return tst; };


    // Tech-independent adding of a named, random array.  
    virtual void add(const std::string& name, const shape_type& shape) = 0;
    virtual void add(const std::string& name, const shape_type& shape, const float* data) = 0;

    // The tests
    virtual json run_arith(json tst) = 0;
    virtual json run_convo(json tst) = 0;
    virtual json run_median(json tst) = 0;
    virtual json run_sort(json tst) = 0;

    std::unique_ptr<float[]> randu(size_t n) {
        auto dat = std::make_unique<float[]>(n);
        for (size_t ind=0; ind<n; ++ind) {
            dat[ind] = runiform(rng);
        }
        return dat;
    }

};


struct TestConfig {
    size_t repeat;
    shape_type shape;
    std::string device;
    time_point start{now()};

    TestConfig(const json& tst)
        : repeat(tst["repeat"])
        , shape(tst["shape"])
        , device(tst["device"])
        {
        }

    void go() {
        start = now();
    }

    json results() {
        json ret;
        double ttot = tdiffms(start);
        ret["time"] = ttot;
        ret["dt"] = ttot / repeat;
        return ret;
    }
        

};


// A "bag of arrays" and test.
template<typename T>
struct PerfT : public Perf {

    std::unordered_map<std::string, T> arrays;

    json run_arith(json tst) {
        TestConfig tc(tst);

        add("A", tc.shape);
        add("B", tc.shape);
        add("C", tc.shape);

        T& A = arrays["A"];
        T& B = arrays["B"];
        T& C = arrays["C"];

        tc.go();

        for (size_t count = 0; count < tc.repeat; ++count) {
            A += B + C;
            A += B * C;
            A /= vs::max(A);
        }
        vs::sync<T>();
        return tc.results();
    }

    json run_convo(json tst) {
        TestConfig tc(tst);

        add("signal", tc.shape);
        add("response", tc.shape);
        add("total", tc.shape);
        const T& s = arrays["signal"];
        const T& r = arrays["response"];

        // meaningless accumulate to assure complete eval.
        T tot = arrays["total"];

        auto R = vs::fft2(r);

        tc.go();

        for (size_t count = 0; count < tc.repeat; ++count) {
            // auto t0 = now();

            // auto t = now();
            auto S = vs::fft2(s);
            // double dt_S = tdiffms(t);

            // t = now();
            auto SR = S*R;
            // double dt_SR = tdiffms(t);

            // t = now();
            auto m = vs::ifft2(SR);
            // double dt_m = tdiffms(t);
            
            // t = now();
            // double dt_sync = tdiffms(t);

            // double dt = tdiffms(t0);
            // std::cerr << count << ": S:" << dt_S << " SR:" << dt_SR << " m:" << dt_m
            //           << " sync:" << dt_sync << " tot:" << dt << "\n";

            tot += vs::real(m);
        }
        vs::sync<T>();

        return tc.results();
    }

    json run_median(json tst) {
        TestConfig tc(tst);

        add("signal", tc.shape);
        const T& s = arrays["signal"];

        shape_type shape1d = {tc.shape[0]};
        add("total", shape1d);
        T tot = arrays["total"];

        tc.go();

        for (size_t count = 0; count < tc.repeat; ++count) {

            tot += vs::median(s, 1);
        }
        vs::sync<T>();

        return tc.results();
    }

    json run_sort(json tst) {

        TestConfig tc(tst);

        add("signal", tc.shape);
        const T& s = arrays["signal"];

        add("total", tc.shape);
        T tot = arrays["total"];

        tc.go();

        for (size_t count = 0; count < tc.repeat; ++count) {
            tot += vs::sort(s, 1);
        }
        vs::sync<T>();

        return tc.results();
    }
    
};


#ifdef USE_ARRAYFIRE
#include <arrayfire.h>

struct PerfAF : public PerfT<af::array> {
    
    virtual json prerun(json tst) {
        if (getdef<std::string>(tst, "device", "cpu") == "gpu") {
            af::setBackend(AF_BACKEND_CUDA);
        }
        else {
            af::setBackend(AF_BACKEND_CPU);            
        }
        tst["tech"] = "af";
        return tst;
    }

    virtual void add(const std::string& name, const shape_type& shape) {
        auto data = randu(shape_size(shape));
        af::dim4 dims(shape.size(), shape.data());
        arrays[name] = af::array(dims, data.get()); // type from float*
    }

    virtual void add(const std::string& name, const shape_type& shape, const float* data) {
        af::dim4 dims(shape.size(), shape.data());
        arrays[name] = af::array(dims, data); // type from float*
    }

};


namespace vs {


template<>
float to_float<af::array>(const af::array& arr)
{
    return arr.scalar<float>();
}

template<>
af::array max<af::array>(const af::array& arr)
{
    return af::max(arr);
}
template<>
af::array min<af::array>(const af::array& arr)
{
    return af::min(arr);
}


template<>
af::array real<af::array>(const af::array& arr)
{
    return af::real(arr);
}
template<>
af::array imag<af::array>(const af::array& arr)
{
    return af::imag(arr);
}

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
af::array median<af::array>(const af::array& arr, int dim)
{
    return af::median(arr, dim);
}
    

template<> af::array& eval<af::array>(af::array& arr)
{
    return af::eval(arr);
}
template<> void sync<af::array>()
{
    af::sync();
}

template<> af::array sort<af::array>(const af::array& arr, int dim)
{
    return af::sort(arr, dim);
}


} // namespace vs

#endif  // USE_ARRAYFIRE

#ifdef USE_LIBTORCH
#include "torch/torch.h"

struct PerfLT : public PerfT<torch::Tensor> {

    std::string device{"cpu"};

    virtual json prerun(json tst) {
        device = getdef<std::string>(tst, "device", "cpu");
        tst["tech"] = "lt";
        return tst;
    }

    virtual void add(const std::string& name, const shape_type& shape) {
        auto data = randu(shape_size(shape));
        std::vector<int64_t> tshape(shape.begin(), shape.end());
        auto borrowed = torch::from_blob((void*)(data.get()), tshape);
        borrowed = borrowed.clone();
        if (device == "gpu") {
            borrowed = borrowed.to(torch::kCUDA);
        }
        arrays[name] = borrowed.clone();
    }

    virtual void add(const std::string& name, const shape_type& shape, const float* data) {
        std::vector<int64_t> tshape(shape.begin(), shape.end());
        auto borrowed = torch::from_blob((void*)(data), tshape);
        borrowed = borrowed.clone();
        if (device == "gpu") {
            borrowed = borrowed.to(torch::kCUDA);
        }
        arrays[name] = borrowed.clone();
    }

};


namespace vs {

template<>
float to_float<torch::Tensor>(const torch::Tensor& arr)
{
    return arr.item<float>();
}

template<>
torch::Tensor max<torch::Tensor>(const torch::Tensor& arr)
{
    return arr.max();
}
template<>
torch::Tensor min<torch::Tensor>(const torch::Tensor& arr)
{
    return arr.min();
}


template<>
torch::Tensor real<torch::Tensor>(const torch::Tensor& arr)
{
    return torch::real(arr);
}
template<>
torch::Tensor imag<torch::Tensor>(const torch::Tensor& arr)
{
    return torch::imag(arr);
}

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
torch::Tensor median<torch::Tensor>(const torch::Tensor& ten, int dim)
{
    auto tup = ten.median(dim);
    return std::get<0>(tup);
}

template<> torch::Tensor sort<torch::Tensor>(const torch::Tensor& arr, int dim)
{
    auto tup = torch::sort(arr, dim);
    return std::get<0>(tup);
}

// Note, torch IS lazy also, at least for GPU.  But, more is needed to implement
// eval/sync than I want to do right now.
// https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
template<> torch::Tensor& eval<torch::Tensor>(torch::Tensor& arr)
{
    return arr;
}

template<> void sync<torch::Tensor>()
{
    return;
}

} // namespace vs

#endif  // USE_LIBTORCH


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

json perf(json tsts)
{
    json results;

    for (auto& tst : tsts) {
        auto tech = getdef<std::string>(tst, "tech", "af");
        std::unique_ptr<Perf> p;

        if (tech == "af") {
            p = std::make_unique<PerfAF>();
        }
        else if (tech == "lt") {
            p = std::make_unique<PerfLT>();
        }
        if (!p) {
            std::cerr << "skipping unknown tech: " << tech << "\n" << tst << "\n";
            continue;
        }

        tst = p->prerun(tst);

        std::cerr << tst << "\n";

        json result;
        std::string kind = tst["kind"];
        if (kind == "arith") {
            result = p->run_arith(tst);
        }
        else if (kind == "convo") {
            result = p->run_convo(tst);
        }
        else if (kind == "median") {
            result = p->run_median(tst);
        }
        else if (kind == "sort") {
            result = p->run_sort(tst);
        }
        else {
            std::cerr << "unknown test: " << kind << " config: " << tst << "\n";
        }
        json jtest;
        jtest["input"] = tst;
        jtest["output"] = result;
        results.push_back(jtest);
    }
    return results;
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
    auto tsts = json::parse(ifile);

    auto res = perf(tsts);

    std::ofstream ofile(oname);
    ofile << std::setw(4) << res << std::endl;
    return 0;    
}
