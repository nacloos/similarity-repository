from setuptools import setup, find_packages


requirements = {
    "platonic": [
        'torchaudio'
    ],
    "rsatoolbox": [
        'rsatoolbox'
    ],
    "netrep": [
        'netrep @ git+https://github.com/ahwillia/netrep'
    ],
    "repsim": [
        'repsim @ git+https://github.com/wrongu/repsim.git'
    ],
    "brainscore": [
        "six",
        "boto3",
        "tqdm",
        "Pillow",
        "entrypoints",
        "numpy",
        "pandas",
        "xarray<2022.6",  # groupby bug was introduced in index refactor: https://github.com/pydata/xarray/issues/6836
        "netcdf4"
    ],
    # "brainscore": [
    #     # copied from github brain-score/setup.py
    #     "numpy>=1.17",

    #     # "brainio @ git+https://github.com/brain-score/brainio",
    #     # use the exact same version as in brainscore_vision otherwise pip will raise conflict error when trying to install both similarity and brainscore_vision
    #     "brainio @ git+https://github.com/brain-score/brainio.git@main",

    #     "importlib-metadata<5",  # workaround to https://github.com/brain-score/brainio/issues/28
    #     "scikit-learn",
    #     # "scikit-learn<0.24",  # 0.24 breaks pls regression
    #     "scipy",
    #     "h5py",
    #     "tqdm",
    #     "gitpython",
    #     "fire",
    #     "networkx",
    #     "matplotlib",
    #     # "tensorflow",  # not needed here
    #     "result_caching @ git+https://github.com/brain-score/result_caching",
    #     "jupyter",
    #     "pybtex",
    #     "peewee",
    #     # "pillow<9.0.0",  # "AttributeError: module 'PIL' has no attribute 'Image'" when calling plt.savefig with old version of PIL
    #     "pillow",
    #     "psycopg2-binary"
    # ],
    "dsa": [
        # ERROR: No matching distribution found for kooplearn>=1.0.6
        # "dsa @ git+https://github.com/mitchellostrow/DSA.git"
    ],
    "imd": [
        "msid @ git+https://github.com/xgfs/imd.git"
    ],
    # https://github.com/KhrulkovV/geometry-score
    # "gs": [
    #     "gs @ git+https://github.com/KhrulkovV/geometry-score.git",
    #     "gudhi"
    # ],
    # https://github.com/IlyaTrofimov/RTD/blob/38b9239c7e228c9ff70e0f8b3719efce0823cd05/README.md#installation
    # "rtd": [
    #     'rtd @ git+https://github.com/IlyaTrofimov/RTD',
    #     # requires cmake (doesn't work with latest cmake), CUDA, gcc (make the installation optional?)
    #     # didn't manage to install it on windows (use dockerfile?)
    #     'risperplusplus @ git+https://github.com/simonzhang00/ripser-plusplus',
    #     'torch',
    # ]
    "neuroaimetrics": [
        "torchmetrics",
        "POT",
        "fastprogress"
    ],
    "resi": [
        # TODO: can't import the package directly because of conflicting 'repsim' name
        # "repsim @ git+https://github.com/mklabunde/resi.git",
        "einops",
        "loguru"
    ],
    "thingsvision": [
        "torchtyping",
        "numba"
    ]
}


install_requires = [
    'pydantic',
    'scikit-learn',
    'matplotlib',
    'requests',
    'gitpython',
    'seaborn',
    'numpy<2.0'
]

for k, v in requirements.items():
    install_requires += v


setup(
    name='similarity-repository',
    version="0.1.0",
    packages=[
        package for package in find_packages() if package.startswith('similarity')
    ],
    package_data={
        'similarity': ['registry/**/*'],
    },
    include_package_data=True,
    install_requires=install_requires,
    description='A repository for similarity measures.',
    author='Nathan Cloos'
)
