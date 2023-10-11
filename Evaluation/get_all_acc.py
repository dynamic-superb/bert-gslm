import argparse
from pathlib import Path
from os.path import join
import json
import glob
import os
import csv
import tqdm
unseen = open('unseen_file.txt', 'r').readlines()
unseen = {l.split()[0]:l.split()[1].strip() for l in unseen}
# print(unseen)
def get_acc(sample_dir):
    try:
        pred_file = json.load(open(join(sample_dir, "samples.json"), "r"))
    except:
        print(sample_dir)
        return "", "",""
    n_correct, n_seen_correct, n_seen_total,n_unseen_correct, n_unseen_total = 0, 0, 0, 0, 0
    for v in pred_file.values():
        assert v['file_name'] in unseen, v['file_name']
        # print(unseen[v['file_name']])
        if unseen[v['file_name']] == '1':
            
            n_unseen_total += 1
            if v['label'] == v["predict"]:
                n_correct += 1
                n_unseen_correct += 1
        else:
            n_seen_total += 1
            if v['label'] == v["predict"]:
                n_correct += 1
                n_seen_correct += 1
    
    acc = f"{n_correct*100 / len(pred_file):.2f}%"
    if n_seen_total > 0:
        seen_acc = f"{n_seen_correct*100/n_seen_total:.2f}%"
    else:
        seen_acc = ""
    
    if n_unseen_total > 0:
        unseen_acc = f"{n_unseen_correct*100/n_unseen_total:.2f}%"
    else:
        unseen_acc = ""
    
    return acc, seen_acc, unseen_acc  
parser = argparse.ArgumentParser()

parser.add_argument("--test_result_dir", type=Path,default="All_test_results")
parser.add_argument("--baseline_name", type=str,default="GSLM")
parser.add_argument("--output_csv", type=Path, default="bert_embedding.csv")
tasks = open('test_tasks.txt', 'r').readlines()
args = parser.parse_args()
all_result = [join(args.test_result_dir, task.strip(), "samples.json") for task in tasks]

with open(args.output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Inference Task', "acc", "seen_acc", 'unseen_acc'])
    for result in tqdm.tqdm(all_result):
        sample_dir = "/".join(result.split('/')[:-1])
        task = sample_dir.split('/')[-1]
        acc, seen_acc, unseen_acc = get_acc(sample_dir)
        writer.writerow([task, f"{acc}", f"{seen_acc}", f"{unseen_acc}"])