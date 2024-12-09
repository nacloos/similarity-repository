import os
import shutil

BASE_DIR = "/mnt/fs5/nclkong/allen_inst/dtd/"
for sp in range(1, 11):
    # sanity check the sets are distinct
    sp_dict = {}
    for prefix in ["train", "val", "test"]:
        filename = os.path.join(BASE_DIR, f"labels/{prefix}{sp}.txt")
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        sp_dict[prefix] = content

    curr_sets = list(sp_dict.values())
    # check that the sets have no repeating elements
    for u in curr_sets:
        assert(len(u) == len(set(u)))
    # sets are pairwise disjoint if their union is the sum of their sizes
    union = set().union(*curr_sets)
    n = sum(len(u) for u in curr_sets)
    assert(n == len(union))

    for prefix in sp_dict.keys():
        print(sp, prefix)
        for x in sp_dict[prefix]:
            assert(len(x.split('/')) == 2)
            source_file = os.path.join(BASE_DIR, f"images/{x}")
            dest_dir = os.path.join(BASE_DIR, f"splits/{prefix}{sp}/{x.split('/')[0]}")
            if not os.path.exists(dest_dir):
                print(f"Making {dest_dir}")
                os.makedirs(dest_dir)
            dest_file = os.path.join(dest_dir, f"{x.split('/')[1]}")
            shutil.copyfile(source_file, dest_file)
