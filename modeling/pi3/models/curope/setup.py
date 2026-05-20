# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_arch_flags() -> list[str]:
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if not arch_list:
        return cuda.get_gencode_flags().replace("compute=", "arch=").split()

    flags: list[str] = []
    for arch in arch_list.replace(";", " ").split():
        arch = arch.strip()
        if not arch:
            continue
        arch = arch.replace("+PTX", "")
        major, minor = arch.split(".")
        code = f"{major}{minor}"
        flags.extend([f"-gencode=arch=compute_{code},code=sm_{code}"])
    return flags


all_cuda_archs = get_cuda_arch_flags()

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=["curope.cpp", "kernels.cu"],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
