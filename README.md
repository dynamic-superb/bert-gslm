# Dynamic-SUPERB

## Download dataset

- You can download the dataset in the releases.
- The download dataset will looks like:

```
gslm-test-data
├── AccentClassification_AccentdbExtended
│   ├── dict.txt
│   ├── test_done.txt
│   ├── test_files.txt
│   ├── test.txt
└── BirdSoundDetection_Warblrb10k
    ├── dict.txt
    ├── test_done.txt
    ├── test_files.txt
    ├── test.txt
    
```

## Install

```shell
conda env create -f environment.yaml
conda activate prompt
```

### Inference from a checkpoint

- TEST_DATA_DIR= The path to gslm-test-data
- MODEL_DIR = The path to the download model
- SAVE_DIR = The path to the inference result dir
```shell
bash download_checkpoint.sh # Need to install gdown
cp checkpoint_best.pt MODEL_DIR
cd Inference
bash run_all_sample.sh MODEL_DIR SAVE_DIR TEST_DATA_DIR

```
## Calculate accuracy and format for google sheet

- See: `Evaluation/get_all_acc.py`
- `Evaluation/unseen_file.txt` is for calculating seen/unseen accuracy

```shell
cd Evaluation
python get_all_acc.py --test_result_dir SAVE_DIR --output_csv GSLM.csv
```

