try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='MLPLibrary',
    version='0.0.1',
    description='A library for constructing MLP models, implemented with Numpy and Python.',
    author='Michael Podbury',
    url='https://github.com/more-th4n-life/MLPLibrary',
    packages=find_packages("."),
    package_dir={'network': 'network',
                 'network.loader': 'network/loader',
                 'network.model': 'network/model',
                 'network.dataset': 'network/dataset'
                 },
    install_requires=['numpy', 'tqdm']
)