from setuptools import setup, find_packages


setup(
    name='similarity-measures',
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith('similarity_measures')],
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'netrep @ git+https://github.com/ahwillia/netrep',
        'rsatoolbox',
        'config-utils @ git+https://github.com/nacloos/config-utils.git',
    ],
    description='',
    author=''
)