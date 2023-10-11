import glob
from os.path import join
import json
import tqdm
label_file = 'espnet_label.txt'
data_dir = '/mnt/data/gslm-test-data'
with open(label_file, 'r') as f:
    label_table = {l.split(maxsplit=1)[0]:l.split(maxsplit=1)[-1].strip() for l in f.readlines()}

new_label_table = {}
test_files = glob.glob(join(data_dir, "*", 'test_files.txt'))
for test_file in tqdm.tqdm(test_files):
    files = open(test_file, 'r').readlines()
    files = [l.strip() for l in files]
    init_length = len(files)
    if '/' in files[0]:
        files = [l.split('/')[-1] for l in files]
    assert len(files) == init_length
    subset = test_file.split('/')[-2]
    for file in tqdm.tqdm(files):
        assert file not in new_label_table, file
        for key, label in label_table.items():
            if file in key and subset in key:
                new_label_table[file] = label
                break
        # assert done, f"file :{file}\ntest_file :{test_file}\n"
                # candidates.append(key)
        # assert len(candidates) == 1, f"file :{file}\ntest_file :{test_file}\n"
            
        # key = sorted(candidates, key=lambda x: len(x))[0]
        # if len(candidates) > 1:
        #     print(file)
        #     print(key)
        #     print(candidates)
        #     exit(0)
        
json.dump(new_label_table, open('gslm_label.json', 'w'))     
    
    
    