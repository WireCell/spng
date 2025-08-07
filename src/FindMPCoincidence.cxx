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

    std::cout << "Nfaces: " << m_anode->faces().size() << std::endl;

    // std::vector<torch::Tensor> m_pitch_tensors;
    // for (const auto& face : m_anode->faces()) {
    const auto & face = m_anode->face(m_face_index);
    m_raygrid_views = torch::zeros({5, 2, 2}/*, options*/);

    std::cout << "Face: " << face->ident() << std::endl;
    const auto & coords = face->raygrid();
    const auto & centers = coords.centers();
    const auto & pitch_dirs = coords.pitch_dirs();
    const auto & pitch_mags = coords.pitch_mags();
    auto next_rays = centers;
    for (int ilayer = 0; ilayer < coords.nlayers(); ++ilayer)
        next_rays[ilayer] += pitch_dirs[ilayer]*pitch_mags[ilayer];

    //Construct views by hand. We need to do this based off of our target plane.
    //We create the 2 trivial blobs first, then 
    std::vector<int> layers = {0, 1, m_aux_plane_l_index+2, m_aux_plane_m_index+2, m_target_plane_index+2};
    int layer_count = 0;
    for (const auto & ilayer : layers) {
        std::cout << "\tCenter: " << centers[ilayer] <<
        " Pitch Dir (Mag): " <<
        pitch_dirs[ilayer] << 
        " (" << pitch_mags[ilayer] << ")" << std::endl;

        // next_rays[ilayer] += pitch_dirs[ilayer]*pitch_mags[ilayer];

        std::cout << "\t\tNext ray: " << next_rays[ilayer] << std::endl;

        //Set the values in the tensor
        m_raygrid_views.index_put_({layer_count, 0, 0}, centers[ilayer][2]);
        m_raygrid_views.index_put_({layer_count, 0, 1}, centers[ilayer][1]);
        m_raygrid_views.index_put_({layer_count, 1, 0}, next_rays[ilayer][2]);
        m_raygrid_views.index_put_({layer_count, 1, 1}, next_rays[ilayer][1]);
        ++layer_count;
    }
    // }

    
    // std::vector<std::unordered_map<int, std::vector<int>>> m_chan_id_to_wires(3);
    for (const auto & plane : face->planes()) {
        auto & map = m_chan_id_to_wires[plane->ident()];
        int ichan = 0;
        for (const auto & plane_chan : plane->channels()) {
            std::cout << "Plane & Chan: " << plane->ident() << " " << ichan << " " << plane_chan->ident() << " wires" << std::endl;
            for (const auto & w : plane_chan->wires()) {
                std::cout << "\t" << w->index() << " " <<  w->planeid().face() << std::endl;
                if (w->planeid().face() == face->ident())
                    map[plane_chan->ident()].push_back(w->index());
            }
            ++ichan;
        }
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

    m_raygrid_views = m_raygrid_views.to(device);
    // m_raygrid_views = m_raygrid_views.to(device);

    // std::cout << m_raygrid_views[0] << std::endl;

    WireCell::Spng::RayGrid::Coordinates m_raygrid_coords =
            WireCell::Spng::RayGrid::Coordinates(m_raygrid_views);
    m_raygrid_coords.to(device);
    std::cout << "Active bounds: " << m_raygrid_coords.active_bounds() << std::endl;
    auto active_bounds = m_raygrid_coords.active_bounds();
    auto element_tensor_n = torch::zeros({active_bounds.index({4, 1}).item<int>()},torch::TensorOptions().dtype(torch::kInt32));
    auto element_tensor_m = torch::zeros({active_bounds.index({3, 1}).item<int>()},torch::TensorOptions().dtype(torch::kInt32));
    auto element_tensor_l = torch::zeros({active_bounds.index({2, 1}).item<int>()},torch::TensorOptions().dtype(torch::kInt32));
    auto element_m_accessor = element_tensor_m.accessor<int, 1>();
    auto element_l_accessor = element_tensor_l.accessor<int, 1>();
    auto element_n_accessor = element_tensor_n.accessor<int, 1>();

    // m_raygrid_coords.to(device);

    //Clone the inputs
    auto target_tensor_n = (*in->tensors())[m_target_plane_index]->tensor().clone().to(/*device*/torch::kCPU);
    auto target_tensor_map = (*in->tensors())[m_target_plane_index]->metadata()["channel_map"];

    auto aux_tensor_l = (*in->tensors())[m_aux_plane_l_index]->tensor().clone().to(/*device*/torch::kCPU);
    auto aux_tensor_l_map = (*in->tensors())[m_aux_plane_l_index]->metadata()["channel_map"];

    auto aux_tensor_m = (*in->tensors())[m_aux_plane_m_index]->tensor().clone().to(/*device*/torch::kCPU);
    auto aux_tensor_m_map = (*in->tensors())[m_aux_plane_m_index]->metadata()["channel_map"];

    //Transform into bool tensors (activities)
    auto tester = torch::zeros({1}).to(/*device*/torch::kCPU);
    torch::nn::MaxPool1d pool(torch::nn::MaxPool1dOptions(4));
    aux_tensor_l = (pool(aux_tensor_l) > tester);
    aux_tensor_m = (pool(aux_tensor_m) > tester);

    // auto active_l = (aux_tensor_l > tester).nonzero();
    // auto active_m = (aux_tensor_m > tester).nonzero();

    // std::cout << "active l\n" << active_l << std::endl;

    // std::cout << "l map " << aux_tensor_l_map.size() << std::endl;
    // auto l_map_tensor = torch::zeros(aux_tensor_l_map.size(), torch::TensorOptions().dtype(torch::kInt64));
    // auto l_map_accessor = l_map_tensor.accessor<long,1>();
    // for (const auto & key : aux_tensor_l_map.getMemberNames()) {
    //     int index = std::stoi(key);
    //     // std::cout << index << " " << aux_tensor_l_map[key] << std::endl;
    //     l_map_accessor[index] = aux_tensor_l_map[key].asInt();
    // }
    // l_map_tensor = l_map_tensor.to(device);


    torch::Tensor output_tensor_active = torch::zeros({target_tensor_n.size(0), element_tensor_n.size(0), target_tensor_n.size(-1)}, torch::TensorOptions(device).dtype(torch::kFloat64));
    torch::Tensor output_tensor_inactive = torch::zeros_like(output_tensor_active, torch::TensorOptions(device).dtype(torch::kFloat64));

    target_tensor_n = pool(target_tensor_n);

    //Apply the first two 'real' layers -- the order doesn't matter?
    // auto coords = m_raygrid_coords[0]; // For testing -- just one side of the APA

    // std::cout << "Trivial Blobs" << m_trivial_blobs << std::endl;
    for (long int irow = 0; irow < aux_tensor_l.sizes().back(); ++irow) {
        //Reset the elements
        element_tensor_l.index_put_({torch::indexing::Slice()}, 0);
        element_tensor_m.index_put_({torch::indexing::Slice()}, 0);
        element_tensor_n.index_put_({torch::indexing::Slice()}, 0);
        auto l_row = aux_tensor_l.index({0, torch::indexing::Slice(), irow});
        auto m_row = aux_tensor_m.index({0, torch::indexing::Slice(), irow});
        auto target_row = target_tensor_n.index({0, torch::indexing::Slice(), irow});
        
        auto l_row_nonzero = l_row.nonzero();
        if (l_row_nonzero.size(0) > 0) {
            std::cout << "nonzero\n" << l_row_nonzero << std::endl;
            // std::cout << "Row: " << irow << std::endl;
            // std::cout << "Nonzero l:\n" << l_row.nonzero() << std::endl;
            // std::cout << "Nonzero l converted:\n" << l_map_tensor.index({l_row.nonzero()}) << std::endl;

            auto nonzero_accessor = l_row_nonzero.accessor<long, 2>();
            int nonzero_size = nonzero_accessor.size(0);
            std::cout << "Nonzero size: " << nonzero_size << std::endl;
            for (long int inz = 0; inz < nonzero_size; ++inz) {
                if (nonzero_accessor.size(1) == 0) {
                    std::cerr << "Error somehow this is zero?" << std::endl;
                }
                else {
                    std::cout << "nonzero: " << nonzero_accessor[inz][0] << std::endl;
                    int mapped_channel = aux_tensor_l_map[std::to_string(nonzero_accessor[inz][0])].asInt();
                    const auto & wires = m_chan_id_to_wires[m_aux_plane_l_index][mapped_channel];
                    for (const auto & w : wires) {
                        std::cout << "\t" << w << std::endl;
                        element_l_accessor[w] = 1;
                    }
                }
            }
            // std::cout << element_tensor_l << std::endl;
        }

        auto m_row_nonzero = m_row.nonzero();
        if (m_row_nonzero.size(0) > 0) {
            std::cout << "nonzero\n" << m_row_nonzero << std::endl;
            // std::cout << "Row: " << irow << std::endl;
            // std::cout << "Nonzero l:\n" << m_row.nonzero() << std::endl;
            // std::cout << "Nonzero l converted:\n" << m_map_tensor.index({m_row.nonzero()}) << std::endl;

            auto nonzero_accessor = m_row_nonzero.accessor<long, 2>();
            int nonzero_size = nonzero_accessor.size(0);
            std::cout << "Nonzero size: " << nonzero_size << std::endl;
            for (long int inz = 0; inz < nonzero_size; ++inz) {
                if (nonzero_accessor.size(1) == 0) {
                    std::cerr << "Error somehow this is zero?" << std::endl;
                }
                else {
                    std::cout << "nonzero: " << nonzero_accessor[inz][0] << std::endl;
                    int mapped_channel = target_tensor_map[std::to_string(nonzero_accessor[inz][0])].asInt();
                    const auto & wires = m_chan_id_to_wires[m_target_plane_index][mapped_channel];
                    for (const auto & w : wires) {
                        std::cout << "\t" << w << std::endl;
                        element_m_accessor[w] = 1;
                    }
                }
            }
            // std::cout << element_tensor_m << std::endl;
        }

        auto target_row_nonzero = target_row.nonzero();
        if (target_row_nonzero.size(0) > 0) {
            std::cout << "nonzero\n" << target_row_nonzero << std::endl;
            // std::cout << "Row: " << irow << std::endl;
            // std::cout << "Nonzero l:\n" << target_row.nonzero() << std::endl;
            // std::cout << "Nonzero l converted:\n" << m_map_tensor.index({target_row.nonzero()}) << std::endl;

            auto nonzero_accessor = target_row_nonzero.accessor<long, 2>();
            int nonzero_size = nonzero_accessor.size(0);
            std::cout << "Nonzero size: " << nonzero_size << std::endl;
            for (long int inz = 0; inz < nonzero_size; ++inz) {
                if (nonzero_accessor.size(1) == 0) {
                    std::cerr << "Error somehow this is zero?" << std::endl;
                }
                else {
                    std::cout << "nonzero: " << nonzero_accessor[inz][0] << std::endl;
                    int mapped_channel = aux_tensor_m_map[std::to_string(nonzero_accessor[inz][0])].asInt();
                    const auto & wires = m_chan_id_to_wires[m_aux_plane_m_index][mapped_channel];
                    for (const auto & w : wires) {
                        std::cout << "\t" << w << std::endl;
                        element_n_accessor[w] = 1;
                    }
                }
            }
            // std::cout << element_tensor_n << std::endl;
        }
        // continue;
        // std::cout << l_row.sizes() << std::endl;
        // std::cout << torch::any(l_row) << std::endl;
        // std::cout << l_row << std::endl;

        // std::cout << m_row.sizes() << std::endl;
        // std::cout << torch::any(m_row) << std::endl;
        // std::cout << m_row << std::endl;

        // std::cout << target_row.sizes() << std::endl;
        // std::cout << torch::any(target_row) << std::endl;
        // std::cout << target_row << std::endl;

        auto raygrid_row_l = element_tensor_l.to(device);
        auto blobs = WireCell::Spng::RayGrid::apply_activity(m_raygrid_coords, m_trivial_blobs, raygrid_row_l);
        // std::cout << "First layer done" << std::endl;
        // std::cout << blobs.sizes() << std::endl;
        if (blobs.size(0) == 0) {
            // std::cout << "Found no blobs. Moving on" << std::endl;
            continue;
        }
        auto raygrid_row_m = element_tensor_m.to(device);
        blobs = WireCell::Spng::RayGrid::apply_activity(m_raygrid_coords, blobs, raygrid_row_m);
        // std::cout << "Second layer done" << std::endl;
        if (blobs.size(0) == 0) {
            // std::cout << "Found no blobs. Moving on" << std::endl;
            continue;
        }
        // continue;
        //For the last layer, get the bounds of the would-be created blobs
        //MP3 means our target plane has activity overlapping with blobs from the first 2 layers
        tester = tester.to(device);
        auto raygrid_row_n = element_tensor_n.to(device);
        auto target_active = (raygrid_row_n > tester);
        // std::cout << target_active << std::endl;
        auto one = torch::ones({1}).to(device);
        if (target_active.any().item<bool>()) {
            auto mp3_blobs = WireCell::Spng::RayGrid::apply_activity(
                m_raygrid_coords, blobs, target_active
            );

            // auto mp3_accessor = mp3_blobs.accessor<long, 3>();
            // std::cout << "MP3 Blobs: " << mp3_blobs.sizes() << std::endl;
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
        auto target_inactive = (raygrid_row_n == tester);
        // std::cout << target_inactive << std::endl;
        if (target_inactive.any().item<bool>()) {
            auto mp2_blobs = WireCell::Spng::RayGrid::apply_activity(
                m_raygrid_coords, blobs, target_inactive
            );
            
            // auto mp2_accessor = mp2_blobs.accessor<long, 3>();
            // std::cout << "MP2 Blobs: " << mp2_blobs.sizes() << std::endl;
            // std::cout << mp2_blobs << std::endl;
            for (int iblob = 0; iblob < mp2_blobs.size(0); ++iblob) {
                // std::cout << iblob << std::endl;
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