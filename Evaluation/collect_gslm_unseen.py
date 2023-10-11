import os
import glob
from os.path import join
import tqdm
import json
train_metadata_dir = "/mnt/data/big-superb-train-data-renamed"
test_metadata_dir = "/mnt/data/big-superb-test-data-renamed"
testfile_dir = "/mnt/data/gslm-test-data"
seen_instr = set()
train_metadata_files = sorted(glob.glob(join(train_metadata_dir, "*", "train", "metadata.json")))
test_metadata_files = sorted(glob.glob(join(test_metadata_dir, "*", "test", "metadata.json")))
test_files = sorted(glob.glob(join(testfile_dir, "*", "test_files.txt")))
output_file = open("unseen_file.txt", 'w')
for train_metadata_file in train_metadata_files:
    metadata = json.load(open(train_metadata_file, 'r'))
    for key, info in metadata.items():
        instr = info['instruction'].split("The answer")[0].strip()
        seen_instr.add(instr)
assert len(test_metadata_files) == len(test_files)
unseen, seen_unseen_total = {}, {}
total, seen = 0, 0
for metadata_file, test_file in tqdm.tqdm(zip(test_metadata_files,test_files), total=len(test_files)):
    assert metadata_file.split('/')[-3] == test_file.split('/')[-2]
    task = metadata_file.split('/')[-3]
    if metadata_file.split('/')[-3] not in seen_unseen_total:
        seen_unseen_total[task] = {"seen_total":0, "unseen_total":0}
    metadata = json.load(open(metadata_file, 'r'))
    testfile_data = [l.strip() for l in open(test_file, 'r').readlines()]
    for key, info in metadata.items():
        instr = info['instruction'].split("The answer")[0].strip()
        assert info['file'] in testfile_data and info['file'] not in unseen, info['file']
        if instr in seen_instr:
            unseen[info['file']] = 0
            seen_unseen_total[task]["seen_total"] += 1
            seen += 1
        else:
            unseen[info['file']] = 1
            seen_unseen_total[task]["unseen_total"] += 1
        total += 1
    
    for file in testfile_data:
        output_file.write(f"{file} {unseen[file]}\n")
json.dump(seen_unseen_total, open('seen_unseen.json', 'w'))
print(f"Seen/Total {seen}/{total}")