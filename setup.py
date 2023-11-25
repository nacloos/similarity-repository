from setuptools import setup, find_packages

# TODO: install all the backend requirements.txt


# https://github.com/KhrulkovV/geometry-score
gs_requires = [
    "gs @ git+https://github.com/KhrulkovV/geometry-score.git",
    "gudhi"
]

# https://github.com/IlyaTrofimov/RTD/blob/38b9239c7e228c9ff70e0f8b3719efce0823cd05/README.md#installation
rtd_requires = [
    'rtd @ git+https://github.com/IlyaTrofimov/RTD',
    # TODO: requires cmake (doesn't work with latest cmake), CUDA, gcc (make the installation optional?)
    # TODO: didn't manage to install it on windows (use dockerfile?)
    """
    ...
    CMake Error at CMakeLists.txt:2 (project):
    Failed to run MSBuild command:

        MSBuild.exe

    to get the value of VCTargetsPath:

        The system cannot find the file specified
    """
    'risperplusplus @ git+https://github.com/simonzhang00/ripser-plusplus',
    'torch',
]

setup(
    name='similarity',
    version="0.0.1",
    packages=[package for package in find_packages()
              if package.startswith('similarity')],
    package_data={
        "similarity": ["similarity/**/*.cue"]
    },
    include_package_data=True,
    install_requires=[
        'pydantic',
        'scikit-learn',
        'matplotlib',
        'rsatoolbox',
        'netrep @ git+https://github.com/ahwillia/netrep',
        # TODO: pypi doesn't allow direct dependencies on github repos
        'config-utils @ git+https://github.com/nacloos/config-utils.git',
    ],
    # extras_require={
    #     'rtd': rtd_requires
    # },
    description='A hub for similarity measures.',
    author='Nathan Cloos'
)
