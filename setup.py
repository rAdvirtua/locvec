import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

def compile_cuda():
    base_dir = os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, "src", "locvec")
    ext = ".dll" if sys.platform == "win32" else ".so"
    output_path = os.path.join(target_dir, "liblocalvec" + ext)
    
    os.makedirs(target_dir, exist_ok=True)

    sources = [
        "src/cuda/ivf_cuda_backend.cu",
        "src/cuda/kmeans_trainer.cu",
        "src/bridge/ivf_wrapper.c",
        "src/bridge/ivf_allocator.c"
    ]

    print(f"Executing CUDA Build: {output_path}")
    cmd = ["nvcc", "-Xcompiler", "-fPIC", "-shared", "-o", output_path] + sources
    
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("Build Failed")
        sys.exit(1)

class CustomBuildPy(build_py):
    def run(self):
        compile_cuda()
        super().run()

class CustomDevelop(develop):
    def run(self):
        compile_cuda()
        super().run()

setup(
    name="locvec",
    version="0.1.0",
    packages=["locvec"],
    package_dir={"": "src"},
    package_data={"locvec": ["*.so", "*.dll"]},
    cmdclass={
        'build_py': CustomBuildPy,
        'develop': CustomDevelop,
    },
)