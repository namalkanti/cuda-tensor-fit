from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

ext_modules = [Extension("cython_opt",
                    sources = ["cython_opt.pyx",
                                "cython_interface.c"],
                    include_dirs=[".", get_include()],
                    extra_compile_args=["-fopenmp"],
                    extra_link_args=["-fopenmp"])]

setup(
        name = "cython_opt",
        cmdclass = {"build_ext" : build_ext},
        ext_modules = ext_modules
        )
                        
