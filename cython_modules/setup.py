from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "clinical_utils", 
        ["clinical_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],  # Optimization flags
    ),
    Extension(
        "bezier_utils",
        ["bezier_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
    Extension(
        "roc_utils",
        ["roc_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
]


# Setup configuration
setup(
    name="cython_modules",
    version="1.0.0",
    ext_modules=cythonize(
        extensions,
        annotate=True,  # Generate HTML annotation of the C code
        language_level=3,  # Python 3 compatibility
    ),
    include_dirs=[np.get_include()]
)