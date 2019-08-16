from skbuild import setup
import torch
from torch.utils import cpp_extension
import re
from os import path as osp
import sys


def _s(s):
    match = re.fullmatch(r"([^:]*\:)(.+)", s)
    if " " in s and match is not None:
        return match[1] + '"' + match[2] + '"'
    else:
        return s


if torch.__version__.startswith("1.0"):
    pytorch_base_include_path_suffix = osp.sep + osp.join("lib", "include")
elif torch.__version__.startswith("1.1"):
    pytorch_base_include_path_suffix = osp.sep + osp.join("include")
elif torch.__version__.startswith("1.2"):
    pytorch_base_include_path_suffix = osp.sep + osp.join("include")
else:
    raise ValueError("unsupported PyTorch version {}".format(torch.__version__))

pytorch_install_prefix = None
for current_include_path in cpp_extension.include_paths(False):
    if ("torch" + pytorch_base_include_path_suffix) in current_include_path:
        offset_of_search_string = current_include_path.rfind("torch" + pytorch_base_include_path_suffix)
        current_include_path = current_include_path[:offset_of_search_string + len("torch")]
        if osp.isdir(osp.join(current_include_path, "share")):
            pytorch_install_prefix = current_include_path
            break

if pytorch_install_prefix is None:
    raise ValueError("could not determine pytorch install prefix")

cmake_args = [
        "-DPYTORCH_INCLUDES:PATH=" + osp.pathsep.join(cpp_extension.include_paths(False)),
        "-DPYTORCH_LDFLAGS:STRING=" + " ".join(map(_s, cpp_extension._prepare_ldflags([], False, False))),
        "-DTORCH_INSTALL_PREFIX:PATH=" + pytorch_install_prefix
    ]
try:
    cmake_args.extend([
        "-DPYTORCH_CUDA_INCLUDES:PATH=" + osp.pathsep.join(cpp_extension.include_paths(True)),
         "-DPYTORCH_CUDA_LDFLAGS:STRING=" + " ".join(map(_s, cpp_extension._prepare_ldflags([], True, False)))
    ])
except OSError:
    pass

setup(
    name="pytorch_register_op_minimal",
    version="0.0.0",
    author="Leopold Walkling",
    author_email="leopold_walkling@web.de",
    cmake_args=cmake_args,
    packages=["pytorch_register_op_minimal"],
    package_dir={"pytorch_register_op_minimal": "pytorch_register_op_minimal"}
)
