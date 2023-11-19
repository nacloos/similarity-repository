from setuptools import setup, find_packages


setup(
    name='similarity',
    version="0.0.1",
    packages=[package for package in find_packages()
              if package.startswith('similarity')],
    package_data={
        "similarity": ["configs/**/*.cue"]
    },
    include_package_data=True,
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'netrep @ git+https://github.com/ahwillia/netrep',
        'rsatoolbox',
        'config-utils @ git+https://github.com/nacloos/config-utils.git',
    ],
    description='A hub for similarity measures.',
    author='Nathan Cloos'
)
