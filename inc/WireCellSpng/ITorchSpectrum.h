/** FILL ME OUT
 */

 #ifndef WIRECELLSPNG_ITORCHSPECTRUM
 #define WIRECELLSPNG_ITORCHSPECTRUM
 
 #include "WireCellUtil/IComponent.h"
 #include <torch/torch.h>

 namespace WireCell {
 
    class ITorchSpectrum : public IComponent<ITorchSpectrum> {
    public:
        virtual ~ITorchSpectrum();
 
        /// Return the coldelec response data
        virtual torch::Tensor spectrum() const = 0;

        /// Get the base shape of the response
        virtual const std::vector<int64_t> & shape() const {return m_shape;};
    protected:
        std::vector<int64_t> m_shape;

        //Idea make a cache by hand with this
        // https://www.reddit.com/r/cpp_questions/comments/16d6seh/hash_value_of_stdvector_gets_computed_but_unable/
        // std::map<vector<int64_t>, torch::Tensor, boost::hash<std::vector<int>>

        // unless this works out of the box
        // https://www.boost.org/doc/libs/1_67_0/boost/compute/detail/lru_cache.hpp
    };
 
 }  // namespace WireCell
 
 #endif