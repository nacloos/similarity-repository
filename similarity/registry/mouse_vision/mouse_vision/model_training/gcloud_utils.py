# Adapted from https://github.com/danielkunin/neural-mechanics/blob/42f0ac8e8826cc63bf1e8f8c7af57fb5097bcd0d/utils/gcloud.py
import os
import subprocess
import glob
from google.cloud import storage


def lookup_tpu_ip_by_name(tpu_name, tpu_zone="us-central1-b"):
    gcloud_cmd = (
        f"/usr/bin/gcloud compute tpus list --zone={tpu_zone} | grep {tpu_name}"
    )

    try:
        out = subprocess.check_output(gcloud_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print("ERROR: gcloud tpu discovery command returned non-zero exist status")
        raise (e)

    ip = str(out).split()[3].split(":")[0]
    return ip


def configure_env_for_tpu(tpu_ip):
    os.environ["XRT_TPU_CONFIG"] = f"tpu_worker;0;{tpu_ip}:8470"
    print(f"\tXRT_TPU_CONFIG env variable set to: {os.environ['XRT_TPU_CONFIG']}")


def save_file_to_bucket(filename, autoremove=True, verbose=False, print_fn=print):
    assert filename[0:5] == "gs://"
    gcs = storage.Client()
    bucket_name = filename.split("gs://")[1].split("/")[0]
    bucket = gcs.get_bucket(bucket_name)

    if "epoch" in filename:
        glob_path = f"{filename.split('epoch')[0]}epoch*.{filename.split('.')[1]}"
        paths = glob.glob(glob_path)
        assert len(paths) > 0
    else:
        paths = [filename]

    for file in paths:
        remote_filename = "/".join(file.split(bucket_name)[1].split("/")[1:])
        blob = bucket.blob(remote_filename)
        blob.upload_from_filename(filename=file)
        if verbose:
            # On TPU, this function is called within an xm subprocess so we pass the 
            # master print fn in that case
            print_fn(f"File {file} posted to gcs")
        if autoremove:
            os.remove(file)  # remove locally


def download_file_from_bucket(filename, ordinal=None, verbose=False, print_fn=print):
    assert filename[0:5] == "gs://"

    gcs = storage.Client()
    bucket_name = filename.split("gs://")[1].split("/")[0]
    bucket = gcs.get_bucket(bucket_name)
    remote_filename = "/".join(filename.split(bucket_name)[1].split("/")[1:])
    blob = bucket.blob(remote_filename)

    if ordinal is None:
        local_filename = filename
    else:
        local_filename = filename.replace(".pt", f"_ord{ordinal}.pt")

    blob.download_to_filename(filename=local_filename)

    if verbose:
        # On TPU, this function is called within an xm subprocess so we pass the 
        # master print fn in that case
        print_fn(f"File {filename} downloaded from gcs to local path {local_filename}")

    return local_filename


def configure_tpu(tpu_name):
    print("Configuring ENV variables for TPU training")
    tpu_ip = lookup_tpu_ip_by_name(tpu_name)
    configure_env_for_tpu(tpu_ip)
