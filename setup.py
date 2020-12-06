try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

subfolder_to_compile_dict = {
    # "MatrixFactorization",
    "baserec.base.similarity.compute_similarity_cython": "baserec/base/similarity/*.pyx",
    # "SLIM_BPR",
}

ext_modules = [
    Extension(
        name,
        [filepath],
        extra_compile_args=['-O2'],
        include_dirs=[numpy.get_include(), ],
    ) for name, filepath in subfolder_to_compile_dict.items()
]

setup(
    name='BaseRecommenders',
    version='0.0.1',
    packages=['baserec'],
    install_requires=[
        'scikit-learn>=0.21.2',
        'pandas>=1.1.4',
        'scikit-optimize>=0.8.1'
    ],
    extras_require={
        'wandb': ["wandb>=0.8.14"],
        'examples': ["typer>=0.3.2"]
    },
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish
        # 'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    ext_modules=cythonize(ext_modules),
)
