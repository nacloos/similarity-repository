"""
Specify a list of folders or files to download from GitHub repositories.
Useful when the code is not available as a pip package.
Can be used to quickly recover from accidental modifications of original content.
"""
import shutil
import git
from pathlib import Path
import tempfile


ROOT_DIR = Path(__file__).parent

# could specify commit hash in the url to increase reproducibility
download_list = [
    {
        "github_repo_url": "https://github.com/js-d/sim_metric",
        "github_path": "dists",
        "local_save_dir": "./similarity/registry/sim_metric"
    },
    # {
    #     "github_repo_url": "https://github.com/brain-score/brain-score",
    #     "github_path": "brainscore",
    #     "local_save_dir": "./similarity/registry/brainscore"
    # },
    {
        "github_repo_url": "https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment",
        "github_path": "CKA.py",
        "local_save_dir": "./similarity/registry/yuanli2333"
    },
    {
        "github_repo_url": "https://github.com/amzn/xfer",
        "github_path": "nn_similarity_index/sim_indices.py",
        "local_save_dir": "./similarity/registry/nn_similarity_index"
    },
    {
        "github_repo_url": "https://github.com/minyoungg/platonic-rep",
        "github_path": "metrics.py",
        "local_save_dir": "./similarity/registry/platonic"
    },
    {
        "github_repo_url": "https://github.com/ViCCo-Group/thingsvision",
        "github_path": "thingsvision/core",
        "local_save_dir": "./similarity/registry/thingsvision"
    },
    # {
    #     # TODO: automatically rename repsim folder to resi and replace "repsim.measures.utils" to "resi.measures.utils" in all import statements
    #     "github_repo_url": "https://github.com/mklabunde/resi",
    #     "github_path": "repsim/measures",
    #     "local_save_dir": "./similarity/registry/resi"
    # },
    {
        "github_repo_url": "https://github.com/Alxmrphi/correcting_CKA_alignment",
        "github_path": "metrics.py",
        "local_save_dir": "./similarity/registry/correcting_cka_alignment"
    },
    # {
    #     # TODO: have to comment out import 'ripserplusplus' and change 'cca_core' imports to relative imports
    #     "github_repo_url": "https://github.com/IlyaTrofimov/RTD",
    #     "github_path": "rtd",
    #     "local_save_dir": "./similarity/registry/rtd"
    # },
    # {
    #     # TODO: need to comment out line 9 to 12 of src/comparators/compare_functions/__init__.py
    #     "github_repo_url": "https://github.com/renyi-ai/drfrankenstein",
    #     "github_path": "src/comparators/compare_functions",
    #     "local_save_dir": "./similarity/registry/drfrankenstein"
    # }
    {
        "github_repo_url": "https://github.com/implicitDeclaration/similarity",
        "github_path": "metric",
        "local_save_dir": "./similarity/registry/implicitdeclaration_similarity"
    },
    # {
    #     # TODO: comment out function line 22 in alignment.py to prevent pytorch cuda error
    #     "github_repo_url": "https://github.com/pnnl/modelsym",
    #     "github_path": "model_symmetries/alignment/alignment.py",
    #     "local_save_dir": "./similarity/registry/modelsym"
    # },
    # {
    #     # TODO: comment out line 1 of stir/__init__.py; replace line 5 with relative import in stir/CKA_minibatch.py
    #     "github_repo_url": "https://github.com/nvedant07/STIR",
    #     "github_path": "stir",
    #     "local_save_dir": "./similarity/registry/stir"
    # },
    {
        "github_repo_url": "https://github.com/maroo-sky/FSD",
        "github_path": "metrics",
        "local_save_dir": "./similarity/registry/fsd"
    },
    {
        "github_repo_url": "https://github.com/uds-lsv/xRSA-AWEs",
        "github_path": "CKA.py",
        "local_save_dir": "./similarity/registry/xrsa_awes"
    },
    # {
    #     # TODO: change import to relative line 24 pwcca.py
    #     "github_repo_url": "https://github.com/technion-cs-nlp/ContraSim",
    #     "github_path": "",
    #     "local_save_dir": "./similarity/registry/contrasim/contrasim"
    # },
    # {
    #     # TODO: change line 7 qmvpa/rsa.py to relative import
    #     "github_repo_url": "https://github.com/qihongl/nnsrm-neurips18",
    #     "github_path": "qmvpa",
    #     "local_save_dir": "./similarity/registry/nnsrm_neurips18"
    # },
    {
        "github_repo_url": "https://github.com/mtoneva/brain_language_nlp",
        "github_path": "utils",
        "local_save_dir": "./similarity/registry/brain_language_nlp"
    },
    # {   
        # TODO: have to comment out llmcomp/measures/__init__.py
        # "github_repo_url": "https://github.com/mklabunde/llm_repsim",
        # "github_path": "llmcomp/measures",
        # "local_save_dir": "./similarity/registry/llm_repsim"
    # },
    {
        "github_repo_url": "https://github.com/neuroailab/mouse-vision",
        "github_path": "mouse_vision",
        "local_save_dir": "./similarity/registry/mouse_vision"
    },
    {
        # TODO: changed line 373 in rcca.py because of deprecation error
        "github_repo_url": "https://github.com/gallantlab/pyrcca",
        "github_path": "rcca/rcca.py",
        "local_save_dir": "./similarity/registry/pyrcca"
    },
    {
        "github_repo_url": "https://github.com/mgwillia/unsupervised-analysis",
        "github_path": "experiments/calculate_cka.py",
        "local_save_dir": "./similarity/registry/unsupervised_analysis"
    },
    {
        "github_repo_url": "https://github.com/nacloos/diffscore",
        "github_path": "diffscore/analysis/similarity_measures.py",
        "local_save_dir": "./similarity/registry/diffscore"
    }
]


def download_from_github(github_repo_url, github_path, local_save_dir):
    """
    Download a folder from a GitHub repository to a local path.
    Args:
        github_repo_url: URL of the GitHub repository
        github_path: path of the folder or file to download in the GitHub repository
        local_save_path: dir where to save the downloaded folder
    """
    local_save_path = ROOT_DIR / local_save_dir / github_path
    if local_save_path.exists():
        print(f"Folder {local_save_path} already exists. Skipping download.")
        return

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)

        print(f"Cloning {github_repo_url}...")
        # Clone the GitHub repository to the temporary directory
        git.Repo.clone_from(github_repo_url, tmp_dir)

        src_path = tmp_dir / github_path
        dst_path = local_save_path
        # Create the directory if it does not exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if folder_path is a file or a folder
        if src_path.is_file():
            # Move the file to the current directory
            shutil.move(src_path, dst_path)
        elif src_path.is_dir():
            # Copy the entire folder to the current directory
            # dst_path = dst_path / github_path
            shutil.copytree(src_path, dst_path)


if __name__ == "__main__":
    for item in download_list:
        github_repo_url = item["github_repo_url"]
        github_path = item["github_path"]
        local_save_dir = item["local_save_dir"]
        download_from_github(github_repo_url, github_path, local_save_dir)
