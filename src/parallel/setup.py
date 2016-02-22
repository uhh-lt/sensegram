from distutils.core import setup
#from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_module = Extension(
    "cython_knn",
    ["cython_knn.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    define_macros=[('CYTHON_TRACE', '1')]
)

setup(
    name = 'Cython KNN',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    include_dirs=[numpy.get_include()],)
