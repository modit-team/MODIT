#!/usr/bin/env bash

CODE_BASE=`realpath ../`;
DATA_DIR="${CODE_BASE}/data/PLBART_DATA";
langs="java,python,en_XX";

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID DATA_SIZE SRC TGT"
            echo
            echo "DATA_SIZE: small, medium"
            echo
            echo "SRC: parent_code, parent_commit, parent_buggy_only, parent_buggy_commit"
            echo
            echo "TGT: child_code, child_full_code"
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1;
DATA_SIZE=${2}.${3}.${4};

SOURCE=source;
TARGET=target;

PATH_2_DATA=${DATA_DIR}/${DATA_SIZE}/data-bin;

MODEL_DIR="${CODE_BASE}/models/PLBART";

SAVE_DIR=${MODEL_DIR}/${DATA_SIZE};
mkdir -p ${SAVE_DIR};
USER_DIR=${CODE_BASE}/src


function generate () {
    model=${SAVE_DIR}/checkpoint_best.pt
    FILE_PREF=${SAVE_DIR}/output_top_10
    RESULT_FILE=${SAVE_DIR}/result_top_10.txt

    fairseq-generate $PATH_2_DATA \
        --user-dir $USER_DIR \
        --path $model \
        --truncate-source \
        --task translation_in_same_language \
        --gen-subset test \
        -t $TARGET -s $SOURCE \
        --sacrebleu --remove-bpe 'sentencepiece' \
        --max-len-b 500 --beam 5 --nbest 5 \
        --batch-size 4 --langs $langs > $FILE_PREF;

    python parse_generated_top_k.py --input ${FILE_PREF} --output ${FILE_PREF}.hyp;

    ORIGINAL_TGT="${DATA_DIR}/${DATA_SIZE}/test.buggy-fixed.fixed";
    python evaluator_top_k.py \
            --ref ${ORIGINAL_TGT} \
            --pre ${FILE_PREF}.hyp \
            --beam 5 --nbest 1 2 5;

    python evaluator_top_k.py \
            --ref ${ORIGINAL_TGT} \
            --pre ${FILE_PREF}.hyp \
            --beam 5 --nbest 1 2 5 >> ${RESULT_FILE};
}

generate
