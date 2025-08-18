local wc = import 'wirecell.jsonnet';
local g = import 'pgraph.jsonnet';



function(){
    local SPNGTorchService = {
        type: "SPNGTorchService",
        name: "dnnroi",
        data: {
            model: "/nfs/data/1/abashyal/spng/spng_dev_050525/toolkit/spng/test/ts-model/test-2.3.ts",
            device: "gpu1",
        },
    },


    local torchfile = g.pnode({
        type: 'TorchFileSource',
        name: 'torchfilesource',
        data: {
            inname: 'testout_fan_apa0.tar',
            prefix: '',
            tag: 'ROILoose',
            "model_path": "/nfs/data/1/abashyal/spng/spng_dev_050525/Pytorch-UNet/ts-model-2.3/test-2.3.ts",
        },
    }, nin=0, nout=1),

    local spng_roi_planes = g.pnode({
        type: 'SPNGROITests',
        name: 'spng_roi_plane',
        data:{
            "apa": 0,
            "plane": 0,
            forward: wc.tn(SPNGTorchService),
            },
    }, nin=1, nout=1, uses=[SPNGTorchService]
    ),

    local tensor_sinks_planes = g.pnode({
    type: 'SPNGTorchTensorFileSink',
    name: 'tfsink_apa_plane',
    data: {
        outname: 'testout_fan_apa_plane.tar',
        prefix: ''
    },
}, nin=1, nout=0),   

local graph = g.pipeline([torchfile,spng_roi_planes,tensor_sinks_planes]),
local app = {
    type: 'Pgrapher',
    data: {
        edges: g.edges(graph),
    },
},

local plugins = ["WireCellSpng","WireCellPgraph","WireCellSio","WireCellRoot"],

local cmdline = {
    type: "wire-cell",
    data: {
        plugins: plugins,
        apps: ["Pgrapher"],
    }
},
seq: [cmdline] + g.uses(graph) + [app]
}.seq