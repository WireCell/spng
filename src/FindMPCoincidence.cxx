#include "WireCellSpng/FindMPCoincidence.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/ITorchSpectrum.h"

WIRECELL_FACTORY(SPNGFindMPCoincidence, WireCell::SPNG::FindMPCoincidence,
                 WireCell::INamed,
                 WireCell::ITorchTensorSetFilter, WireCell::IConfigurable)

WireCell::SPNG::FindMPCoincidence::FindMPCoincidence()
  : Aux::Logger("SPNGFindMPCoincidence", "spng") {

}

WireCell::SPNG::FindMPCoincidence::~FindMPCoincidence() {};


void WireCell::SPNG::FindMPCoincidence::configure(const WireCell::Configuration& config) {
    m_rebin_val = get(config, "rebin_val", m_rebin_val);

    //Get the indices of the planes we're working with.
    //We apply MP2/MP3 finding to some target plane n
    //And we need planes l & m to determine those.
    m_target_plane_index = get(config, "target_plane_index", m_target_plane_index);
    m_aux_plane_l_index = get(config, "aux_plane_l_index", m_aux_plane_l_index);
    m_aux_plane_m_index = get(config, "aux_plane_m_index", m_aux_plane_m_index);

    //Check that we aren't requesting any of the same 2 planes
    if ((m_target_plane_index == m_aux_plane_l_index) ||
        (m_target_plane_index == m_aux_plane_m_index) ||
        (m_aux_plane_m_index == m_aux_plane_l_index)) {
        THROW(ValueError() <<
            errmsg{"Must request unqiue indices for the target and auxiliary planes. Provided:\n"} <<
            errmsg{String::format("\tTarget (n): %d\n", m_target_plane_index)} <<
            errmsg{String::format("\tAux (l): %d\n", m_aux_plane_l_index)} <<
            errmsg{String::format("\tAux (m): %d\n", m_aux_plane_m_index)}
        );
    }

}

bool WireCell::SPNG::FindMPCoincidence::operator()(const input_pointer& in, output_pointer& out) {
    out = nullptr;
    if (!in) {
        log->debug("EOS ");
        return true;
    }
    log->debug("Running FindMPCoincidence");

    //Clone the inputs
    auto target_tensor_n = (*in->tensors())[m_target_plane_index]->tensor().clone();
    auto aux_tensor_l = (*in->tensors())[m_aux_plane_l_index]->tensor().clone();
    auto aux_tensor_m = (*in->tensors())[m_aux_plane_m_index]->tensor().clone();

    //Transform into bool tensors (activities)
    auto tester = torch::zeros({1}).to(target_tensor_n.device());
    target_tensor_n = (target_tensor_n > tester);
    aux_tensor_l = (aux_tensor_l > tester);
    aux_tensor_m = (aux_tensor_m > tester);

    //

    
    // TODO: set md?
    // Configuration set_md, tensor_md;
    // set_md["tag"] = m_output_set_tag;
    // tensor_md["tag"] = m_output_tensor_tag;

    // std::vector<ITorchTensor::pointer> itv{
    //     std::make_shared<SimpleTorchTensor>(tensor_clone, tensor_md)
    // };
    // out = std::make_shared<SimpleTorchTensorSet>(
    //     in->ident(), set_md,
    //     std::make_shared<std::vector<ITorchTensor::pointer>>(itv)
    // );

    return true;
}