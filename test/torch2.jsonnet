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
    make_spng :: function(tools, debug_force_cpu=false, apply_gaus=true, do_roi_filters=false) {

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

        local wiener_tight_filters = [
            wiener_tight_u, wiener_tight_v, wiener_tight_w,

        ],
        local torch_wiener_tight_filters = [
            torch_wiener_tight_only_u, torch_wiener_tight_only_v, torch_wiener_tight_only_w,
        ],

        local torch_wiener_wide_only_u = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_wiener_wide_only_u",
                uses: [wiener_wide_u],
                data: {
                    spectra: [
                        wc.tn(wiener_wide_u),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local torch_wiener_wide_only_v = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_wiener_wide_only_v",
                uses: [wiener_wide_v],
                data: {
                    spectra: [
                        wc.tn(wiener_wide_v),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },
        local torch_wiener_wide_only_w = {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_wiener_wide_only_w",
                uses: [wiener_wide_w],
                data: {
                    spectra: [
                        wc.tn(wiener_wide_w),
                    ],
                    debug_force_cpu: debug_force_cpu,
                },
        },

        local wiener_wide_filters = [
            wiener_wide_u, wiener_wide_v, wiener_wide_w,
        ],
        local torch_wiener_wide_filters = [
            torch_wiener_wide_only_u, torch_wiener_wide_only_v, torch_wiener_wide_only_w,
        ],

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

        local make_replicator_post_tight(anode, iplane) = {
            ret : g.pnode({
                type: 'TorchTensorSetReplicator',
                name: 'post_tight_replicator_%d_%d' % [anode.data.ident, iplane],
                data: {
                    multiplicity: 5,
                }
            }, nin=1, nout=5, uses=[anode])
        }.ret,

        local make_replicator_post_gaus(anode, iplane) = {
            ret : g.pnode({
                type: 'TorchTensorSetReplicator',
                name: 'post_gaus_replicator_%d_%d' % [anode.data.ident, iplane],
                data: {multiplicity: 2,}

            }, nin=1, nout=2, uses=[anode])
        }.ret,
        
        local make_replicator_post_gaus_simple(anode, iplane) = {
            ret : g.pnode({
                type: 'TorchTensorSetReplicator',
                name: 'post_gaus_replicator_%d_%d' % [anode.data.ident, iplane],
                data: {multiplicity: (if iplane > 1 then 2 else 5)}
            }, nin=1, nout=(if iplane > 1 then 2 else 5), uses=[anode])
        }.ret,
        local make_pipeline(anode, iplane, apply_gaus=true, do_roi_filters=false) = {
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

            local the_wiener_tight = if iplane < 3 then wiener_tight_filters[iplane] else wiener_tight_filters[2],
            local the_torch_wiener_tight = if iplane < 3 then torch_wiener_tight_filters[iplane] else torch_wiener_tight_filters[2],
            local the_torch_wiener_wide = if iplane < 3 then torch_wiener_wide_filters[iplane] else torch_wiener_wide_filters[2],

            local torch_roi_tight = {
                    type: "Torch1DSpectrum",
                    name: "torch_1dspec_roi_tight",
                    uses: [the_wiener_tight, ROI_tight_lf],
                    data: {
                        spectra: [
                            wc.tn(the_wiener_tight),
                            wc.tn(ROI_tight_lf),
                        ],
                        debug_force_cpu: debug_force_cpu,
                    },
            },
            local torch_roi_tighter = {
                    type: "Torch1DSpectrum",
                    name: "torch_1dspec_roi_tighter",
                    uses: [the_wiener_tight, ROI_tighter_lf],
                    data: {
                        spectra: [
                            wc.tn(the_wiener_tight),
                            wc.tn(ROI_tighter_lf),
                        ],
                        debug_force_cpu: debug_force_cpu,
                    },
            },

            
            local torch_roi_loose = {
                    type: "Torch1DSpectrum",
                    name: "torch_1dspec_roi_loose",
                    uses: [the_wiener_tight, ROI_loose_lf, ROI_tight_lf],
                    data: {
                        spectra: [
                            wc.tn(the_wiener_tight),
                            wc.tn(ROI_loose_lf),
                            wc.tn(ROI_tight_lf),
                        ],
                        debug_force_cpu: debug_force_cpu,
                    },
            },


            local apply_wiener_tight = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_wiener_tight_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(the_torch_wiener_tight),
                    dimension: 1,
                    // target_tensor: "HfGausWide",
                    output_set_tag: "WienerTight",
                },
            },
            nin=1, nout=1,
            uses=[the_torch_wiener_tight]),
            
            local apply_wiener_wide = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_wiener_wide_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(the_torch_wiener_wide),
                    dimension: 1,
                    // target_tensor: "HfGausWide",
                    output_set_tag: "WienerWide",
                },
            },
            nin=1, nout=1,
            uses=[the_torch_wiener_wide]),
            
            local apply_loose_roi = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_loose_roi_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(torch_roi_loose),
                    dimension: 1,
                    // target_tensor: 'HfGausWide',
                    output_set_tag: 'ROILoose',
                },
            },
            nin=1, nout=1,
            uses=[torch_roi_loose]),
            
            local apply_tight_roi = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_tight_roi_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(torch_roi_tight),
                    dimension: 1,
                    // target_tensor: 'HfGausWide',
                    output_set_tag: 'ROITight',
                },
            },
            nin=1, nout=1,
            uses=[torch_roi_tight]),

            local apply_tighter_roi = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_tighter_roi_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(torch_roi_tighter),
                    dimension: 1,
                    // target_tensor: 'HfGausWide',
                    output_set_tag: 'ROITighter',
                },
            },
            nin=1, nout=1,
            uses=[torch_roi_tighter]),

            local spng_gaus_app = g.pnode({
                type: 'SPNGApply1DSpectrum',
                name: 'spng_gaus_apa%d_plane%d' % [anode.data.ident, iplane],
                data: {
                    base_spectrum_name: wc.tn(torch_gaus_filter), #put in if statement
                    dimension: 1,
                    // target_tensor: "Default",
                    output_set_tag: "HfGausWide",
                },
            },
            nin=1, nout=1,
            uses=[torch_gaus_filter]),

            local torch_to_tensor = g.pnode({
                type: 'TorchToTensor',
                name: 'torchtotensor_%d_%d' % [anode.data.ident, iplane],
                data: {},
            }, nin=1, nout=1),

            local tagger = g.pnode({
                type: 'SPNGTorchTensorSetTagger',
                name: 'test_tagger_%d_%d' % [anode.data.ident, iplane],
                data: {
                    allow_retagging: false,
                    tag_list: {
                        a: '1',
                        b: '2',
                    },
                }
            }, nin=1, nout=1),

            local tensor_sink = g.pnode({
                type: 'TensorFileSink',
                name: 'tfsink_%d_%d' % [anode.data.ident, iplane],
                data: {
                    outname: 'testout_fan_apa%d_plane%d.tar' % [anode.data.ident, iplane],
                    prefix: ''
                },
            }, nin=1, nout=0),


            local decon_and_gaus = [spng_decon] + (if apply_gaus then [spng_gaus_app] else []),
            local convert_and_sink = [torch_to_tensor, tensor_sink],

            local post_gaus_replicator = make_replicator_post_gaus(anode, iplane),
            local post_tight_replicator = if iplane < 2 then make_replicator_post_tight(anode, iplane) else null,

            local post_gaus_replicator_simple = make_replicator_post_gaus_simple(anode, iplane),

            local collator = g.pnode({
                type: 'TorchTensorSetCollator',
                name: 'collate_%d_%d' % [anode.data.ident, iplane],
                data: {
                    output_set_tag: 'collated',
                    // multiplicity: 2,
                    multiplicity: (if iplane < 2 then 5 else 2),
                },
            }, nin=if iplane < 2 then 5 else 2, nout=1),
            // }, nin=2, nout=1),

            local replicate_then_wiener = g.intern(
                innodes=[post_gaus_replicator],
                outnodes=[apply_wiener_tight, apply_wiener_wide],
                centernodes=[],
                edges = [
                    g.edge(post_gaus_replicator, apply_wiener_tight, 0),
                    g.edge(post_gaus_replicator, apply_wiener_wide, 1),
                ]
            ),
            // local post_gaus_filters = g.intern(
            //     innodes=[replicate_then_wiener],
            //     outnodes=[collator],
            //     centernodes=[],
            //     edges = [
            //         g.edge(replicate_then_wiener, collator, 0, 0),
            //         g.edge(replicate_then_wiener, collator, 1, 1),
            //     ],
            // ),

            // local post_gaus_filters = if iplane < 2 then g.intern(
            //     innodes=[replicate_then_wiener],
            //     outnodes=[collator],
            //     centernodes=[post_tight_replicator, apply_tight_roi, apply_tight_roi, apply_loose_roi],
            //     edges = [
            //         g.edge(replicate_then_wiener, post_tight_replicator, 0), #wiener_tight --> 3 (one of loose, tight, tighter roi)
                    
            //         g.edge(post_tight_replicator, apply_tight_roi, 0),
            //         g.edge(apply_tight_roi, collator, 0, 0),
                    
            //         g.edge(post_tight_replicator, apply_tighter_roi, 1),
            //         g.edge(apply_tighter_roi, collator, 0, 1),
                    
            //         g.edge(post_tight_replicator, apply_loose_roi, 2),
            //         g.edge(apply_loose_roi, collator, 0, 2),
                    
            //         g.edge(replicate_then_wiener, collator, 1, 3), #wiener_wide
            //     ],
            // ) else g.intern(
            //     innodes=[replicate_then_wiener],
            //     outnodes=[collator],
            //     centernodes=[],
            //     edges = [
            //         g.edge(replicate_then_wiener, collator, 0, 0),
            //         g.edge(replicate_then_wiener, collator, 1, 1),
            //     ],
            // ),

            local post_gaus_filters = if iplane < 2 then g.intern(
                innodes=[post_gaus_replicator_simple],
                outnodes=[collator],
                centernodes=[apply_wiener_tight, apply_wiener_wide, apply_tight_roi, apply_tighter_roi, apply_loose_roi],
                edges = [
                    g.edge(post_gaus_replicator_simple, apply_wiener_tight, 0, 0),
                    g.edge(apply_wiener_tight, collator, 0, 0),

                    g.edge(post_gaus_replicator_simple, apply_wiener_wide, 1, 0),
                    g.edge(apply_wiener_wide, collator, 0, 1),

                    g.edge(post_gaus_replicator_simple, apply_tight_roi, 2, 0),
                    g.edge(apply_tight_roi, collator, 0, 2),
                    
                    g.edge(post_gaus_replicator_simple, apply_tighter_roi, 3, 0),
                    g.edge(apply_tighter_roi, collator, 0, 3),

                    g.edge(post_gaus_replicator_simple, apply_loose_roi, 4, 0),
                    g.edge(apply_loose_roi, collator, 0, 4),
                ],
            ) else g.intern(
                innodes=[post_gaus_replicator_simple],
                outnodes=[collator],
                centernodes=[apply_wiener_tight, apply_wiener_wide],
                edges = [
                    g.edge(post_gaus_replicator_simple, apply_wiener_tight, 0, 0),
                    g.edge(apply_wiener_tight, collator, 0, 0),

                    g.edge(post_gaus_replicator_simple, apply_wiener_wide, 1, 0),
                    g.edge(apply_wiener_wide, collator, 0, 1),
                ],
            ),


            local full_pipeline = (
                decon_and_gaus + 
                (if do_roi_filters then [post_gaus_filters] else []) +
                convert_and_sink
            ),

            ret : g.pipeline(
                full_pipeline
                // [
                //     spng_decon,
                //     spng_gaus_app,
                //     torch_to_tensor,
                //     tensor_sink,
                // ]
            ),
            
            // ret : g.intern(
            //     innodes=[spng_decon],
            //     centernodes=[spng_gaus_app, torch_to_tensor, tensor_sink],
            //     outnodes=[tensor_sink],
            //     edges = [
            //         g.edge(spng_decon, spng_gaus_app),
            //         g.edge(spng_gaus_app, torch_to_tensor),
            //         g.edge(torch_to_tensor, tensor_sink),
            //     ]
            // )
        }.ret,

        // local tf_fans = [make_fanout(a) for a in tools.anodes],
        
        local spng_fanout(anode, apply_gaus=true, do_roi_filters=false) = {

            #FrameToTorchSetFanout
            local tf_fan = make_fanout(anode),

            #SPNGDecon + Output stuff -- per plane
            local pipelines = [
                make_pipeline(anode, iplane, apply_gaus, do_roi_filters)
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
            spng_fanout(anode, apply_gaus=apply_gaus, do_roi_filters=do_roi_filters) for anode in tools.anodes
        ],
    }.ret,


}