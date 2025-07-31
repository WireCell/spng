#include "WireCellSpng/FindMPCoincidence.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellSpng/SimpleTorchTensor.h"
#include "WireCellSpng/SimpleTorchTensorSet.h"
#include "WireCellSpng/ITorchSpectrum.h"
#include "WireCellSpng/RayTest.h"
#include "WireCellSpng/RayTiling.h"


WIRECELL_FACTORY(SPNGFindMPCoincidence, WireCell::SPNG::FindMPCoincidence,
                 WireCell::INamed,
                 WireCell::ITorchTensorSetFilter, WireCell::IConfigurable)

WireCell::SPNG::FindMPCoincidence::FindMPCoincidence()
  : Aux::Logger("SPNGFindMPCoincidence", "spng") {

}

WireCell::SPNG::FindMPCoincidence::~FindMPCoincidence() {};


void WireCell::SPNG::FindMPCoincidence::configure(const WireCell::Configuration& config) {
    m_rebin_val = get(config, "rebin_val", m_rebin_val);

    m_debug_force_cpu = get(config, "debug_force_cpu", m_debug_force_cpu);

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

    m_readout_plane_width = get(config, "readout_plane_width", m_readout_plane_width);
    m_readout_plane_height = get(config, "readout_plane_height", m_readout_plane_height);
    m_pitch = get(config, "pitch", m_pitch);
    m_angle_in_radians = get(config, "angle_in_radians", m_angle_in_radians);

    //Get trivial blobs
    m_trivial_blobs = WireCell::Spng::RayGrid::trivial_blobs();
    //Create the views & coordinates used in RayGrid


    m_anode_tn = get(config, "anode", m_anode_tn);
    m_anode = Factory::find_tn<IAnodePlane>(m_anode_tn);

    int iface = 0;
    // std::vector<torch::Tensor> m_pitch_tensors;
    for (const auto& face : m_anode->faces()) {

        m_raygrid_views.push_back(torch::zeros({5, 2, 2}/*, options*/));

        std::cout << "Face: " << iface << std::endl;
        const auto & coords = face->raygrid();
        const auto & centers = coords.centers();
        auto next_rays = centers;
        const auto & pitch_dirs = coords.pitch_dirs();
        const auto & pitch_mags = coords.pitch_mags();
        for (int ilayer = 0; ilayer < coords.nlayers(); ++ilayer) {
            std::cout << "\tCenter: " << centers[ilayer] <<
            " Pitch Dir (Mag): " <<
            pitch_dirs[ilayer] << 
            " (" << pitch_mags[ilayer] << ")" << std::endl;

            next_rays[ilayer] += pitch_dirs[ilayer]*pitch_mags[ilayer];

            std::cout << "\t\tNext ray: " << next_rays[ilayer] << std::endl;

            //Set the values in the tensor
            m_raygrid_views.back().index_put_({ilayer, 0, 0}, centers[ilayer][2]);
            m_raygrid_views.back().index_put_({ilayer, 0, 1}, centers[ilayer][1]);
            m_raygrid_views.back().index_put_({ilayer, 1, 0}, next_rays[ilayer][2]);
            m_raygrid_views.back().index_put_({ilayer, 1, 1}, next_rays[ilayer][1]);            
        }
        ++iface;
    }

    // //For testing/building up do one face at first
    // const auto & face = m_anode->faces()[0];
    // std::cout << "Face: " << face << std::endl;
    // const auto & raygrid = face->raygrid();
    // std::cout << "Got Raygrid" << std::endl;
    // const auto & pitch_mags = face->raygrid().pitch_mags();
    // std::cout << pitch_mags << std::endl;
    // const auto & pitch_dirs = face->raygrid().pitch_dirs();
    
    // const auto & centers = face->raygrid().centers();
    // auto next_rays = centers;
    // std::cout << "Checking" << std::endl;
    // for (int ilayer = 0; ilayer < face->raygrid().nlayers(); ++ilayer) {
    //     std::cout << "ilayer: " << ilayer << std::endl;
    //     next_rays[ilayer] += pitch_dirs[ilayer]*pitch_mags[ilayer];
    // }

    // // torch::Tensor pitches = torch::zeros({5, 2, 2}/*, options*/);
    // // pitches.index_put_({0}, torch::Tensor(
    // //     {{face->raygrid().centers[0]}}
    // // ));
}

bool WireCell::SPNG::FindMPCoincidence::operator()(const input_pointer& in, output_pointer& out) {
    out = nullptr;
    if (!in) {
        log->debug("EOS ");
        return true;
    }
    log->debug("Running FindMPCoincidence");

    torch::Device device((
        (torch::cuda::is_available() && !m_debug_force_cpu) ? torch::kCUDA : torch::kCPU
    ));
    m_trivial_blobs = m_trivial_blobs.to(device);

    for (auto & v : m_raygrid_views) v = v.to(device);
    // m_raygrid_views = m_raygrid_views.to(device);

    std::cout << m_raygrid_views[0] << std::endl;

    std::vector<WireCell::Spng::RayGrid::Coordinates> m_raygrid_coords;
    for (auto & v : m_raygrid_views) {
        m_raygrid_coords.push_back(
            WireCell::Spng::RayGrid::Coordinates(v)
        );
        m_raygrid_coords.back().to(device);
    }
    // m_raygrid_coords.to(device);

    //Clone the inputs
    auto target_tensor_n = (*in->tensors())[m_target_plane_index]->tensor().clone().to(device);
    auto aux_tensor_l = (*in->tensors())[m_aux_plane_l_index]->tensor().clone().to(device);
    auto aux_tensor_m = (*in->tensors())[m_aux_plane_m_index]->tensor().clone().to(device);

    //Transform into bool tensors (activities)
    auto tester = torch::zeros({1}).to(device);
    torch::nn::MaxPool1d pool(torch::nn::MaxPool1dOptions(4));
    aux_tensor_l = (pool(aux_tensor_l) > tester);
    aux_tensor_m = (pool(aux_tensor_m) > tester);

    torch::Tensor output_tensor_active = torch::zeros(target_tensor_n.sizes(), torch::TensorOptions(device).dtype(torch::kFloat64));
    torch::Tensor output_tensor_inactive = torch::zeros(target_tensor_n.sizes(), torch::TensorOptions(device).dtype(torch::kFloat64));

    target_tensor_n = pool(target_tensor_n);

    std::cout << aux_tensor_l.sizes() << std::endl;
    std::cout << aux_tensor_m.sizes() << std::endl;
    std::cout << target_tensor_n.sizes() << std::endl;

    //Apply the first two 'real' layers -- the order doesn't matter?
    auto coords = m_raygrid_coords[0]; // For testing -- just one side of the APA

    std::cout << "Trivial Blobs" << m_trivial_blobs << std::endl;
    for (long int irow = 0; irow < aux_tensor_l.sizes().back(); ++irow) {
        std::cout << "Row: " << irow << std::endl;
        auto l_row = aux_tensor_l.index({0, torch::indexing::Slice(), irow});
        auto m_row = aux_tensor_m.index({0, torch::indexing::Slice(), irow});
        auto target_row = target_tensor_n.index({0, torch::indexing::Slice(), irow});
        // std::cout << l_row.sizes() << std::endl;
        // std::cout << torch::any(l_row) << std::endl;
        // std::cout << l_row << std::endl;

        // std::cout << m_row.sizes() << std::endl;
        // std::cout << torch::any(m_row) << std::endl;
        // std::cout << m_row << std::endl;

        // std::cout << target_row.sizes() << std::endl;
        // std::cout << torch::any(target_row) << std::endl;
        // std::cout << target_row << std::endl;

        auto blobs = WireCell::Spng::RayGrid::apply_activity(coords, m_trivial_blobs, l_row);
        std::cout << "First layer done" << std::endl;
        std::cout << blobs.sizes() << std::endl;
        if (blobs.size(0) == 0) {
            std::cout << "Found no blobs. Moving on" << std::endl;
            continue;
        }
        blobs = WireCell::Spng::RayGrid::apply_activity(coords, blobs, m_row);
        std::cout << "Second layer done" << std::endl;
        if (blobs.size(0) == 0) {
            std::cout << "Found no blobs. Moving on" << std::endl;
            continue;
        }

        //For the last layer, get the bounds of the would-be created blobs
        //MP3 means our target plane has activity overlapping with blobs from the first 2 layers
        auto target_active = (target_row > tester);
        // std::cout << target_active << std::endl;
        auto one = torch::ones({1}).to(device);
        if (target_active.any().item<bool>()) {
            auto mp3_blobs = WireCell::Spng::RayGrid::apply_activity(
                coords, blobs, target_active
            );

            // auto mp3_accessor = mp3_blobs.accessor<long, 3>();
            std::cout << "MP3 Blobs: " << mp3_blobs.sizes() << std::endl;
            for (int iblob = 0; iblob < mp3_blobs.size(0); ++iblob) {
                // std::cout << mp3_accessor[iblob][4][0] << " " << mp3_accessor[iblob][4][1] << std::endl;
                output_tensor_active.index_put_(
                    {
                        0,
                        torch::indexing::Slice(mp3_blobs.index({iblob, -1, 0}).item<long>(), mp3_blobs.index({iblob, -1, 1}).item<long>()),
                        torch::indexing::Slice(4*irow, 4*(irow+1))
                    },
                    one
                );
            }
            // std::cout << 4*irow << " " << 4*(irow+1) << std::endl;
            // std::cout << lo_mp3 << std::endl;
            // std::cout << hi_mp3 << std::endl;
        }
        
        //MP2 means our target plane does not have activity overlapping with blobs from the first 2 layers
        auto target_inactive = (target_row == tester);
        // std::cout << target_inactive << std::endl;
        if (target_inactive.any().item<bool>()) {
            auto mp2_blobs = WireCell::Spng::RayGrid::apply_activity(
                coords, blobs, target_inactive
            );
            
            // auto mp2_accessor = mp2_blobs.accessor<long, 3>();
            std::cout << "MP2 Blobs: " << mp2_blobs.sizes() << std::endl;
            std::cout << mp2_blobs << std::endl;
            for (int iblob = 0; iblob < mp2_blobs.size(0); ++iblob) {
                std::cout << iblob << std::endl;
                // std::cout << mp2_accessor[iblob][4][0] << " " << mp2_accessor[iblob][4][1] << std::endl;
                output_tensor_inactive.index_put_(
                    {
                        0,
                        torch::indexing::Slice(
                            mp2_blobs.index({iblob, -1, 0}).item<long>(),
                            mp2_blobs.index({iblob, -1, 1}).item<long>()),
                        torch::indexing::Slice(4*irow, 4*(irow+1))
                    },
                    one
                );
            }
            // std::cout << 4*irow << " " << 4*(irow+1) << std::endl;
            // std::cout << lo_mp2 << std::endl;
            // std::cout << hi_mp2 << std::endl;
        }
    }
    
    // TODO: set md?
    Configuration set_md, mp2_md, mp3_md;
    set_md["tag"] = "";//m_output_set_tag;
    mp2_md["tag"] = "mp2";
    mp3_md["tag"] = "mp3";

    std::vector<ITorchTensor::pointer> itv{
        std::make_shared<SimpleTorchTensor>(output_tensor_active.clone(), mp3_md),
        std::make_shared<SimpleTorchTensor>(output_tensor_inactive.clone(), mp2_md),
    };
    out = std::make_shared<SimpleTorchTensorSet>(
        in->ident(), set_md,
        std::make_shared<std::vector<ITorchTensor::pointer>>(itv)
    );

    return true;
}