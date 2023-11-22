from setuptools import setup, find_packages


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
        # TODO: pypi doesn't allow direct dependencies on github repos
        'netrep @ git+https://github.com/ahwillia/netrep',
        'config-utils @ git+https://github.com/nacloos/config-utils.git',
    ],
    description='A hub for similarity measures.',
    author='Nathan Cloos'
)
