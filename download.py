"""
Specify a list of folders or files to download from GitHub repositories.
Useful when the code is not available as a pip package.
Can be used to quickly recover from accidental modifications of original content.
"""
import shutil
import git
from pathlib import Path
import tempfile


# could specify commit hash in the url to increase reproducibility
download_list = [
    {
        "github_repo_url": "https://github.com/js-d/sim_metric",
        "github_path": "dists",
        "local_save_dir": "./similarity/backend/sim_metric"
    },
    {
        "github_repo_url": "https://github.com/brain-score/brain-score",
        "github_path": "brainscore",
        "local_save_dir": "./"
    },
    {
        "github_repo_url": "https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment",
        "github_path": "CKA.py",
        "local_save_dir": "./similarity/backend/yuanli2333"
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
    local_save_path = Path(local_save_dir) / github_path
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
        dst_path = Path(local_save_dir)

        # Check if folder_path is a file or a folder
        if src_path.is_file():
            # Move the file to the current directory
            shutil.move(src_path, dst_path)
        elif src_path.is_dir():
            # Copy the entire folder to the current directory
            dst_path = dst_path / github_path
            shutil.copytree(src_path, dst_path)


if __name__ == "__main__":
    for item in download_list:
        github_repo_url = item["github_repo_url"]
        github_path = item["github_path"]
        local_save_dir = item["local_save_dir"]
        download_from_github(github_repo_url, github_path, local_save_dir)
