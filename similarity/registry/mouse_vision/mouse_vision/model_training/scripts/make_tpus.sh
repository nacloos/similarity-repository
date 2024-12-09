#!/bin/bash
for i in `seq $1 $2`; do
    printf -v tpu_name "mv-tpu-%02d" $i
    address="10.$(($i)).200.0/29"
    echo "attempting to create $tpu_name at $address"
    gcloud compute tpus create $tpu_name --accelerator-type=${3-"v3-8"} --zone=${4-"us-central1-b"} --range=$address --network=default --version=pytorch-1.6 &
done

wait
echo "all tpus created successfully"
