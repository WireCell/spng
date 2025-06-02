// TODO -- brief descrip

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';


// local wpidU0 = wc.WirePlaneId(wc.Ulayer, 0);
// local wpidV0 = wc.WirePlaneId(wc.Vlayer, 0);
// local wpidW0 = wc.WirePlaneId(wc.Wlayer, 0);

// local wpidU1 = wc.WirePlaneId(wc.Ulayer, 1);
// local wpidV1 = wc.WirePlaneId(wc.Vlayer, 1);
// local wpidW1 = wc.WirePlaneId(wc.Wlayer, 1);

// function make_wpid(tools)
{
    make_spng :: function(tools, debug_force_cpu=false) {

        local ROI_loose_lf = {
            data: {
                max_freq: 0.001,
                tau: 2.0e-06,
                use_negative_freqs: false,
            },
            name: "ROI_loose_lf",
            type: "LfFilter"
        },

        local ROI_tight_lf = {
            data: {
                max_freq: 0.001,
                tau: 1.6e-05,
                use_negative_freqs: false,
            },
            name: "ROI_tight_lf",
            type: "LfFilter"
        },
        local ROI_tighter_lf = {
            data: {
                max_freq: 0.001,
                tau: 8.0e-050,
                use_negative_freqs: false,
            },
            name: "ROI_tighter_lf",
            type: "LfFilter"
        },
        local gaus_tight = {
            data: {
                "flag": true,
                max_freq: 0.001,
                power: 2,
                sigma: 0,
                use_negative_freqs: false,

            },
            name: "Gaus_tight",
            type: "HfFilter"
        },
        local wiener_tight_u = {
            data: {
                flag: true,
                max_freq: 0.001,
                power: 6.55413,
                sigma: 0.000221933,
                use_negative_freqs: false,
            },
            name: "Wiener_tight_U",
            type: "HfFilter"
        },
        local wiener_tight_v = {
            data: {
                flag: true,
                max_freq: 0.001,
                power: 8.75998,
                sigma: 0.000222723,
                use_negative_freqs: false,
            },
            name: "Wiener_tight_V",
            type: "HfFilter"
        },

        local wiener_tight_w = {
            data: {
                flag: true,
                max_freq: 0.001,
                power: 3.47846,
                sigma: 0.000225567,
                use_negative_freqs: false,
            },
            name: "Wiener_tight_W",
            type: "HfFilter"
        },

        local wiener_wide_u = {
            data: {
                flag: true,
                max_freq: 0.001,
                power: 5.05429,
                sigma: 0.000186765,
                use_negative_freqs: false,
            },
            name: "Wiener_wide_U",
            type: "HfFilter"
        },

        local wiener_wide_v = {
            data: {
                flag: true,
                max_freq: 0.001,
                power: 5.77422,
                sigma: 0.0001936,
                use_negative_freqs: false,
            },
            name: "Wiener_wide_V",
            type: "HfFilter"
        },
        local wiener_wide_w = {
            data: {
                flag: true,
                max_freq: 0.001,
                power: 4.37928,
                sigma: 0.000175722,
                use_negative_freqs: false,
            },
            name: "Wiener_wide_W",
            type: "HfFilter"
        },

        local torch_roi_tight = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_roi_tight",
                uses: [ROI_tight_lf],
                data: {
                    spectra: [
                        wc.tn(ROI_tight_lf),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local torch_roi_tighter = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_roi_tighter",
                uses: [ROI_tighter_lf],
                data: {
                    spectra: [
                        wc.tn(ROI_tighter_lf),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local torch_roi_loose = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_roi_loose",
                uses: [ROI_loose_lf],
                data: {
                    spectra: [
                        wc.tn(ROI_loose_lf),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },

        local torch_wiener_tight_only_u = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_wiener_tight_only_u",
                uses: [wiener_tight_u],
                data: {
                    spectra: [
                        wc.tn(wiener_tight_u),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local torch_wiener_tight_only_v = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_wiener_tight_only_v",
                uses: [wiener_tight_v],
                data: {
                    spectra: [
                        wc.tn(wiener_tight_v),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local torch_wiener_tight_only_w = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_wiener_tight_only_w",
                uses: [wiener_tight_w],
                data: {
                    spectra: [
                        wc.tn(wiener_tight_w),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },

        local gaus_filter = {
                type: 'HfFilter',
                name: 'Gaus_wide',
                data: {
                    max_freq: 0.001,  // warning: units
                    power: 2,
                    flag: true,
                    sigma: 0.00012,  // caller should provide
                    use_negative_freqs: false,
                }
        },
        local torch_gaus_filter = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_gaus",
                uses: [gaus_filter],
                data: {
                    spectra: [
                        wc.tn(gaus_filter),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local wire_filters = [
            {
                type: 'HfFilter',
                name: 'Wire_ind',
                data: {
                    max_freq: 1,  // warning: units
                    power: 2,
                    flag: false,
                    sigma: 1.0 / wc.sqrtpi * 0.75,  // caller should provide
                }
            },
            {
                type: 'HfFilter',
                name: 'Wire_col',
                data: {
                    max_freq: 1,  // warning: units
                    power: 2,
                    flag: false,
                    sigma: 1.0 / wc.sqrtpi * 10.0,  // caller should provide
                }
            },
        ],

        local torch_wire_filters = [
            {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_ind",
                uses: [wire_filters[0]],
                data: {
                    spectra: [
                        wc.tn(wire_filters[0]),
                    ],
                    debug_force_cpu: debug_force_cpu,

                },
            },
            {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_col",
                uses: [wire_filters[1]],
                data: {
                    spectra: [
                        wc.tn(wire_filters[1]),
                    ],
                    debug_force_cpu: debug_force_cpu,

                },
            },
        ],
        local make_fanout(anode, name=null) = {
            ret : g.pnode({
                type: 'FrameToTorchSetFanout',
                name:
                    if std.type(name) == 'null'
                    then anode.name + 'torchfanout%d' % anode.data.ident
                    else name,
                data: {
                    anode: wc.tn(anode),
                    expected_nticks: 6000,

                    output_groups: [
                        [wc.WirePlaneId(wc.Ulayer, 0, anode.data.ident),
                        wc.WirePlaneId(wc.Ulayer, 1, anode.data.ident)],
                        
                        [wc.WirePlaneId(wc.Vlayer, 0, anode.data.ident),
                        wc.WirePlaneId(wc.Vlayer, 1, anode.data.ident)],
                        
                        [wc.WirePlaneId(wc.Wlayer, 0, anode.data.ident)],
                        
                        [wc.WirePlaneId(wc.Wlayer, 1, anode.data.ident)],
                    ],
                    debug_force_cpu: debug_force_cpu,


                }
            }, nin=1, nout=4, uses=[anode]),
        }.ret,

        # TODO -- Abstract away
        local nchans = [800, 800, 480, 480],

        

        local make_pipeline(anode, iplane) = {
            local the_field = if std.length(tools.fields) > 1 then tools.fields[anode.data.ident] else tools.fields[0],
            local torch_frer = {
                type: "TorchFRERSpectrum",
                name: "torch_frer%d_plane%d" % [anode.data.ident, iplane],
                uses: [
                    the_field,
                    tools.elec_resp
                ],
                data: {
                    field_response: wc.tn(the_field),#"FieldResponse:field%d"% anode.data.ident,
                    fr_plane_id: if iplane > 2 then 2 else iplane,
                    ADC_mV: 11702142857.142859,
                    inter_gain: 1.0,
                    default_nchans : nchans[iplane],
                    default_nticks: 6000,
                    default_period: 500.0, #512.0,
                    extra_scale: 1.0,
                    anode_num: anode.data.ident,
                    debug_force_cpu: debug_force_cpu,

                }
            },

            local the_wire_filter = if iplane > 1 then torch_wire_filters[1] else torch_wire_filters[0],

            local spng_decon = g.pnode({
                type: 'SPNGDecon',
                name: 'spng_decon_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    frer_spectrum: wc.tn(torch_frer),
                    wire_filter: wc.tn(the_wire_filter), #put in if statement
                    coarse_time_offset: 1000,
                    debug_no_frer: false,
                    debug_no_wire_filter: false,
                    debug_no_roll: false,
                    debug_force_cpu: debug_force_cpu,
                },
            },
            nin=1, nout=1,
            uses=[torch_frer, the_wire_filter]),


            local spng_gaus_app = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_gaus_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(torch_gaus_filter), #put in if statement
                    dimension: 1,
                },
            },
            nin=1, nout=1,
            uses=[torch_gaus_filter]),

            local torch_to_tensor = g.pnode({
                type: 'TorchToTensor',
                name: 'torchtotensor_%d_%d' % [anode.data.ident, iplane],
                data: {},
            }, nin=1, nout=1),

            local tensor_sink = g.pnode({
                type: 'TensorFileSink',
                name: 'tfsink_%d_%d' % [anode.data.ident, iplane],
                data: {
                    outname: 'testout_fan_apa%d_plane%d.tar' % [anode.data.ident, iplane],
                    prefix: ''
                },
            }, nin=1, nout=0),

            ret : g.pipeline(
                [
                    spng_decon,
                    spng_gaus_app,
                    torch_to_tensor,
                    tensor_sink,
                ]
            ),
        }.ret,

        // local tf_fans = [make_fanout(a) for a in tools.anodes],
        
        local spng_fanout(anode,) = {

            #FrameToTorchSetFanout
            local tf_fan = make_fanout(anode),

            #SPNGDecon + Output stuff -- per plane
            local pipelines = [
                make_pipeline(anode, iplane)
                for iplane in std.range(0, 3)
            ],

            ret : g.intern(
                innodes=[tf_fan],
                outnodes=pipelines,
                edges=[
                    g.edge(tf_fan, pipelines[0], 0),
                    g.edge(tf_fan, pipelines[1], 1),
                    g.edge(tf_fan, pipelines[2], 2),
                    g.edge(tf_fan, pipelines[3], 3),
                ]
              ),
        }.ret,
        
        ret : [
            spng_fanout(anode) for anode in tools.anodes
        ],
    }.ret
}