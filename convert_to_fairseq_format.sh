#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Expected 2 arguments, but got $# arguments."
    exit 1
fi

DATA_DIR=$1
SAVE_DIR=$2

# Remove the trailing slash.
DATA_DIR="${DATA_DIR%/}"

TRAIN_PATH="${DATA_DIR}/train.txt"
VALID_PATH="${DATA_DIR}/validation.txt"
TEST_PATH="${DATA_DIR}/test.txt"
DICT_PATH="${DATA_DIR}/dict.txt"

WORKERS=16

fairseq-preprocess --only-source --srcdict "${DICT_PATH}" --trainpref "${TRAIN_PATH}" --validpref "${VALID_PATH}" --testpref "${TEST_PATH}" --destdir "${SAVE_DIR}" --workers "${WORKERS}"
