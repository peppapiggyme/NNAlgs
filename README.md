# Introduction
- package for neural network algorithm training and tuning
- `uproot`+`keras`+`lmdb`
- bowen.zhang@cern.ch

# Layout
```text
TO BE FILLED
```

# Installation
- copy the following to my_env.yml
```yaml
name: decaymode
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=1_llvm
  - _tflow_select=2.1.0=gpu
  - absl-py=0.9.0=py37_0
  - astor=0.7.1=py_0
  - awkward=0.12.20=py_0
  - backcall=0.1.0=py_0
  - binutils_impl_linux-64=2.33.1=h53a641e_8
  - binutils_linux-64=2.33.1=h9595d00_17
  - blinker=1.4=py_1
  - c-ares=1.15.0=h516909a_1001
  - ca-certificates=2020.4.5.1=hecc5488_0
  - cachetools=3.1.1=py_0
  - certifi=2020.4.5.1=py37hc8dfbb8_0
  - cffi=1.14.0=py37hd463f26_0
  - chardet=3.0.4=py37_1003
  - click=7.0=py_0
  - cryptography=2.8=py37h72c5cf5_1
  - cudatoolkit=10.1.243=h6bb024c_0
  - cudnn=7.6.5=cuda10.1_0
  - cupti=10.1.168=0
  - cycler=0.10.0=py_2
  - decorator=4.4.2=py_0
  - deprecated=1.2.7=py_0
  - entrypoints=0.3=py37hc8dfbb8_1001
  - freetype=2.10.0=he983fc9_1
  - gast=0.2.2=py_0
  - gcc_impl_linux-64=7.3.0=hd420e75_5
  - gcc_linux-64=7.3.0=h553295d_17
  - google-auth=1.11.2=py_0
  - google-auth-oauthlib=0.4.1=py_2
  - google-pasta=0.1.8=py_0
  - grpcio=1.27.2=py37hf8bcb03_0
  - gxx_impl_linux-64=7.3.0=hdf63c60_5
  - gxx_linux-64=7.3.0=h553295d_17
  - h5py=2.10.0=nompi_py37h513d04c_102
  - hdf5=1.10.5=nompi_h3c11f04_1104
  - icu=64.2=he1b5a44_1
  - idna=2.9=py_1
  - ipykernel=5.1.4=py37h5ca1d4c_0
  - ipython=7.13.0=py37h5ca1d4c_0
  - ipython_genutils=0.2.0=py_1
  - jedi=0.16.0=py37hc8dfbb8_1
  - jpeg=9c=h14c3975_1001
  - jupyter_client=6.0.0=py_0
  - jupyter_core=4.6.3=py37hc8dfbb8_1
  - keras=2.3.1=py37_0
  - keras-applications=1.0.8=py_1
  - keras-preprocessing=1.1.0=py_0
  - kiwisolver=1.1.0=py37h99015e2_1
  - krb5=1.16.4=h2fd8d38_0
  - ld_impl_linux-64=2.33.1=h53a641e_8
  - libblas=3.8.0=15_openblas
  - libcblas=3.8.0=15_openblas
  - libcurl=7.68.0=hda55be3_0
  - libedit=3.1.20170329=hf8c457e_1001
  - libffi=3.2.1=he1b5a44_1006
  - libgcc-ng=9.2.0=h24d8f2e_2
  - libgfortran-ng=7.3.0=hdf63c60_5
  - libgomp=9.2.0=h24d8f2e_2
  - libgpuarray=0.7.6=h14c3975_1003
  - libiconv=1.15=h516909a_1005
  - liblapack=3.8.0=15_openblas
  - libopenblas=0.3.8=h5ec1e0e_0
  - libpng=1.6.37=hed695b0_0
  - libprotobuf=3.11.4=h8b12597_0
  - libsodium=1.0.17=h516909a_0
  - libssh2=1.8.2=h22169c7_2
  - libstdcxx-ng=9.2.0=hdf63c60_2
  - libtiff=4.1.0=hc3755c2_3
  - libuuid=2.32.1=h14c3975_1000
  - libxml2=2.9.10=hee79883_0
  - llvm-openmp=9.0.1=hc9558a2_2
  - lz4=3.0.2=py37hb076c26_1
  - lz4-c=1.8.3=he1b5a44_1001
  - mako=1.1.0=py_0
  - markdown=3.2.1=py_0
  - markupsafe=1.1.1=py37h8f50634_1
  - matplotlib-base=3.2.0=py37h250f245_1
  - mkl=2020.0=166
  - ncurses=6.1=hf484d3e_1002
  - ninja=1.10.0=hc9558a2_0
  - numpy=1.18.1=py37h8960a57_1
  - oauthlib=3.0.1=py_0
  - olefile=0.46=py_0
  - openssl=1.1.1g=h516909a_0
  - opt_einsum=3.2.0=py_0
  - pandas=1.0.1=py37hb3f55d8_0
  - parso=0.6.2=py_0
  - patsy=0.5.1=py_0
  - pexpect=4.8.0=py37hc8dfbb8_1
  - pickleshare=0.7.5=py37hc8dfbb8_1001
  - pillow=7.0.0=py37hefe7db6_0
  - pip=20.0.2=py_2
  - prompt-toolkit=3.0.4=py_0
  - prompt_toolkit=3.0.4=0
  - protobuf=3.11.4=py37he1b5a44_0
  - ptyprocess=0.6.0=py_1001
  - pyasn1=0.4.8=py_0
  - pyasn1-modules=0.2.7=py_0
  - pycparser=2.20=py_0
  - pygments=2.6.1=py_0
  - pygpu=0.7.6=py37hc1659b7_1000
  - pyjwt=1.7.1=py_0
  - pyopenssl=19.1.0=py_1
  - pyparsing=2.4.6=py_0
  - pysocks=1.7.1=py37_0
  - python=3.7.6=h357f687_4_cpython
  - python-dateutil=2.8.1=py_0
  - python-lmdb=0.96=py37he1b5a44_0
  - python-xxhash=1.4.3=py37h516909a_0
  - python_abi=3.7=1_cp37m
  - pytorch=1.4.0=py3.7_cuda10.1.243_cudnn7.6.3_0
  - pytz=2019.3=py_0
  - pyyaml=5.3=py37h516909a_0
  - pyzmq=19.0.0=py37hac76be4_1
  - readline=8.0=hf8c457e_0
  - requests=2.23.0=py37_0
  - requests-oauthlib=1.2.0=py_0
  - rsa=4.0=py_0
  - scipy=1.4.1=py37h921218d_0
  - seaborn=0.10.0=py_1
  - setuptools=46.0.0=py37hc8dfbb8_2
  - six=1.14.0=py_1
  - sqlite=3.30.1=hcee41ef_0
  - statsmodels=0.11.1=py37h8f50634_1
  - tensorboard=2.1.0=py3_0
  - tensorflow=2.1.0=gpu_py37h7a4bb67_0
  - tensorflow-base=2.1.0=gpu_py37h6c5654b_0
  - tensorflow-estimator=2.1.0=pyhd54b08b_0
  - tensorflow-gpu=2.1.0=h0d30ee6_0
  - termcolor=1.1.0=py_2
  - theano=1.0.4=py37he1b5a44_1001
  - tk=8.6.10=hed695b0_0
  - torchvision=0.5.0=py37_cu101
  - tornado=6.0.4=py37h8f50634_1
  - tqdm=4.43.0=py_0
  - traitlets=4.3.3=py37hc8dfbb8_1
  - uproot=3.11.3=py37_0
  - uproot-base=3.11.3=py37_0
  - uproot-methods=0.7.3=py_0
  - urllib3=1.25.7=py37_0
  - wcwidth=0.1.8=py_0
  - werkzeug=1.0.0=py_0
  - wheel=0.34.2=py_1
  - wrapt=1.12.1=py37h8f50634_1
  - xrootd=4.11.2=py37h0a84524_0
  - xz=5.2.4=h14c3975_1001
  - yaml=0.2.2=h516909a_1
  - zeromq=4.3.2=he1b5a44_2
  - zlib=1.2.11=h516909a_1006
  - zstandard=0.13.0=py37he1b5a44_0
  - zstd=1.4.4=h3b9ef0a_1
prefix: /opt/conda/envs/decaymode
```
- (updated on 02/02/2020 using)
```shell script
conda env export > my_env.yml
```
- then restore the environments
```shell script
conda env create -f my_env.yml
```

# [WIP] How to add a new project xxx
TO BE DONE


# Current workspace
>["work-2020" on Jupyter-Lab (US), valid until 28/09/2020](https://work-2020.atlas-ml.org/)
- 6 GPU (GeForce RTX 2080 Ti)
- 16 CPU (Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz)
- 64GB RAM

# TODOs
- **Testing**
  - More performance plots
  - Training history plots
- ~~using MxAOD instead of flat ROOT tree~~
- unify configuration files
- user can increase GPU usage. (e.g. GPU-Util[1])

[1] [What does GPU-Util means in nvidia-smi](https://stackoverflow.com/questions/40937894/nvidia-smi-volatile-gpu-utilization-explanation/40938696)
