This repository’s agent instructions live in [CLAUDE.md](./CLAUDE.md).

Before making changes, read `CLAUDE.md` and follow its guidance for build, test, style, and contribution workflows.

## Cursor Cloud specific instructions

- Python 3.10 is the `/opt/flashinfer-venv` virtualenv. Login shells prepend that `bin` directory and `/usr/local/cuda/bin` to `PATH`. `/usr/local/bin/python` and `/usr/local/bin/python3` are wrappers that exec the virtualenv interpreter. Do not replace those wrappers with symlinks: Python resolves `pyvenv.cfg` from `argv[0]`, and a symlink makes `import torch` fail.
- CUDA 12.9 is installed at `/usr/local/cuda` (`CUDA_HOME`). `nvcc` is `/usr/local/cuda/bin/nvcc`. Do not symlink `nvcc` into `/usr/local/bin`. FlashInfer sets `CUDA_HOME` from `dirname(dirname(which nvcc))`, so that symlink would point the toolkit at `/usr`.
- Cloud Agent VMs have no GPU. `flashinfer show-config` reports the CUDA runtime as unavailable and cannot register GPU modules (`No supported CUDA architectures found`). Kernel launches and GPU tests cannot run here. The CPU contract check is `pytest tests/utils/test_nvfp4_4over6_config.py`. A direct library call that does not need a device is `flashinfer.quantization.nvfp4_quantization_utils.make_nvfp4_global_scale`.
- The environment install passes `BUILD_NVEP=0`, the documented opt-out for the optional NIXL/NCCL expert-parallel native build. Reinstall without that variable when that build is required.
- `3rdparty/cutlass`, `spdlog`, `cccl`, and `nixl` are initialized during install. If a branch moves those gitlinks, run `git submodule update --init --recursive`.
