
## tar -C /nfs/data/1/abashyal -xf /home/calcuttj/torch.tar
## Create a wct-dev-view and upload the environment
```bash
wcwc task wct-dev-view spng_dev_050525
cd spng_dev_050525 && direnv allow
```
## Install python libraries and the toolkit
```bash
cd python
pip install -e .
```
## Configuration and Installation
```bash
cd ../toolkit/
#export TDIR=/nfs/data/1/abashyal/opt/libtorch
export TDIR=$PREFIX/lib/python3.11/site-packages/torch 
./wcb configure --prefix=$PREFIX   --boost-mt --boost-libs=$PREFIX/lib   --boost-include=$PREFIX/include   --with-jsonnet-libs=gojsonnet   --with-cuda-lib=/usr/lib/x86_64-linux-gnu,$PREFIX/targets/x86_64-linux/lib   --with-cuda-include=/nfs/data/1/abashyal/spng/spng_dev_050525/local/targets/x86_64-linux/include/  --with-libtorch=$TDIR   --with-libtorch-include=$TDIR/include,$TDIR/include/torch/csrc/api/include   --with-root=$PREFIX
./wcb
./wcb install
```

## Note that if the warnings are treated as errors, add this in toolkit/wscript::configure(cfg) function towards the end

```python
##Remove -Werror from CXXFLAGS
if '-Werror' in cfg.env.CXXFLAGS:
    cfg.env.CXXFLAGS.remove('-Werror')
    info("Removing -Werror from CXXFLAGS")~
```
# With the SPNG Build

## Download SPNG and checkout Jake's Branch (Ideally in the spng_dev_2025)
git clone git@github.com:calcuttj/wire-cell-toolkit.git toolkit

git checkout feature/calcuttj_WirePlane_const_comps

## Run the wcb configuration and wcb build/install as mentioned above.

# Running the toolkit

## Make sure all the libraries are listed in the .envrc file
```bash
# Find built libs to avoid install.  Note, edit this list to taste.
path_add LD_LIBRARY_PATH $PWD/toolkit/build/{apps,aux,gen,iface,hio,img,pgraph,root,sig,sigproc,sio,spng,tbb,util} 
```

## Run the wire-cell jsonnet
In my case, the jsonnet file neeeded pdhd configuration from the cvmfs
```bash
cfg=/cvmfs/dune.opensciencegrid.org/products/dune/dunereco/v10_03_01d01/wire-cell-cfg/
export WIRECELL_PATH=$cfg:$WIRECELL_PATH 
wire-cell -l stdout -L debug -P $cfg -P ../../toolkit/cfg/ -C elecGain=7.8 ../../mytools/wct-sim-check.jsonnet
```

### Or whatever jsonnet file you have.

## Creating the workflow:

```bash
wirecell-pgraph dotify  -V elecGain=7.8 -V SPNG=1 wct-sim-check.jsonnet  pipeline.pdf
```

# Creating SPNG Input and Running SPNG Decon
```bash
# Create source for the SPNG
cfg=/cvmfs/dune.opensciencegrid.org/products/dune/dunereco/v10_03_01d01/wire-cell-cfg/
export WIRECELL_PATH=$cfg:$WIRECELL_PATH 
wire-cell -l stdout -L debug -P ../../toolkit/cfg/ -P $cfg -V SPNG=1 -C elecGain=7.8  wct-sim-framesink.jsonnet

# Reset cfg path and run the SPNG DECON
cfg=./
export WIRECELL_PATH=$cfg:$WIRECELL_PATH 
wire-cell -l stdout -L debug -P ../../toolkit/cfg/ -P $cfg -V SPNG=1 -C elecGain=7.8  wct-framesource_new.jsonnet 
```
## After Decon and collated:
```bash
# Create source for the SPNG
cfg=/cvmfs/dune.opensciencegrid.org/products/dune/dunereco/v10_03_01d01/wire-cell-cfg/
export WIRECELL_PATH=$cfg:$WIRECELL_PATH 
wire-cell -l stdout -L debug -P ../../toolkit/cfg/ -P $cfg -V SPNG=1 -C elecGain=7.8  wct-sim-framesink.jsonnet

# Reset cfg path and run the SPNG DECON
cfg=./
export WIRECELL_PATH=$cfg:$WIRECELL_PATH 
wire-cell -l stdout -L debug -P ../../toolkit/cfg/ -P $cfg -V SPNG=1 -C ApplyGaus=1 -C ROI=1 -C CollateAPAs=1 -C elecGain=7.8  wct-framesource_new.jsonnet 


```

# Create the SPNG Workflow
```bash
wirecell-pgraph dotify  -V elecGain=7.8 -V SPNG=1 wct-framesource_new.jsonnet output_spng.pdf
```

# Post Jake Edits:
```bash
wirecell-pgraph dotify -V elecGain=7.8 -V SPNG=1 -V ApplyGaus=1 -V ROI=1 -V CollateAPAs=1  wct-framesource_new.jsonnet output_spng_roi.pdf
```
## 
```bash
file /nfs/data/1/abashyal/spng/spng_dev_050525/toolkit/spng/test/ts-model/unet-l23-cosmic500-e50-new.ts
```

## Content of the ts file
```bash
 unzip -v /nfs/data/1/abashyal/spng/spng_dev_050525/toolkit/spng/test/ts-model/unet-l23-cosmic500-e50-new.ts 
```