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
    make_spng :: function(tools) {
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
                ]
                },
            },
            {
                type: "Torch1DSpectrum",
                name: "torch_1dspec_col",
                uses: [wire_filters[1]],
                data: {
                spectra: [
                    wc.tn(wire_filters[1]),
                ]
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
                },
            },
            nin=1, nout=1,
            uses=[torch_frer, the_wire_filter]),

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