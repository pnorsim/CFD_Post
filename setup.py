"""Build the cfdpost C++ extension with CMake.

    pip install --no-build-isolation -e .      # development (module built in python/cfdpost)
    pip install --no-build-isolation .         # regular install

Environment: CMAKE_ARGS (extra CMake options, e.g. -DCFDPOST_NATIVE=ON),
CMAKE_BUILD_PARALLEL_LEVEL (build jobs).
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        out = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build = Path(self.build_temp).resolve() / "cmake"
        build.mkdir(parents=True, exist_ok=True)
        try:
            import pybind11
            pyb = ["-Dpybind11_DIR=" + pybind11.get_cmake_dir()]
        except ImportError:
            pyb = []
        args = ["-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}",
                f"-DCFDPOST_MODULE_DIR={out}", *pyb, *shlex.split(os.environ.get("CMAKE_ARGS", ""))]
        subprocess.check_call(["cmake", "-S", str(ROOT), "-B", str(build), *args])
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 2))
        subprocess.check_call(["cmake", "--build", str(build), "-j", jobs])


setup(
    ext_modules=[CMakeExtension("cfdpost._core")],
    cmdclass={"build_ext": CMakeBuild},
    package_dir={"": "python"},
    packages=["cfdpost"],
)
