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
    {
        # TODO: automatically rename repsim folder to resi and replace "repsim.measures.utils" to "resi.measures.utils" in all import statements
        "github_repo_url": "https://github.com/mklabunde/resi",
        "github_path": "repsim/measures",
        "local_save_dir": "./similarity/registry/resi"
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
