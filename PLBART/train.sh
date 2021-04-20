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

#checkpoint_3_25000.pt  checkpoint_6_50000.pt  checkpoint_8_75000.pt checkpoint_11_100000.pt
PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt;
PRETRAIN=${CODE_BASE}/models/pretrained/checkpoints/${PRETRAINED_MODEL_NAME};

PATH_2_DATA=${DATA_DIR}/${DATA_SIZE}/data-bin;

echo "Source: $SOURCE Target: $TARGET";

MODEL_DIR="${CODE_BASE}/models/PLBART";

SAVE_DIR=${MODEL_DIR}/${DATA_SIZE};
mkdir -p ${SAVE_DIR};
USER_DIR=${CODE_BASE}/src

#if [[ $DATA_SIZE == 'small' ]];
#then
#    BATCH_SIZE=16
#    UPDATE_FREQ=1
#else
#    BATCH_SIZE=8
#    UPDATE_FREQ=2
#fi

BATCH_SIZE=4;
UPDATE_FREQ=4;

function fine_tune () {
    OUTPUT_FILE=${SAVE_DIR}/finetune.log
    best_model=${SAVE_DIR}/checkpoint_best.pt
    if [[ -f ${best_model} ]]; then
	    echo "Found a trained checkpoint, Not performing training!";
	    return;
	  fi

    model=${SAVE_DIR}/checkpoint_last.pt;
    if [[ -f ${model} ]]; then
	    restore="";
    else
	    restore="--restore-file ${PRETRAIN} --reset-dataloader  --reset-optimizer --reset-meters --reset-lr-scheduler";
    fi
    fairseq-train $PATH_2_DATA \
        --user-dir $USER_DIR --truncate-source \
        --langs $langs --task translation_in_same_language \
        --arch mbart_base --layernorm-embedding \
        --source-lang $SOURCE --target-lang $TARGET \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --max-epoch 30 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay --lr 5e-05 --min-lr -1 \
        --warmup-updates 500 --max-update 100000 \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
        --seed 1234 --log-format json --log-interval 100 \
        ${restore} \
        --eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints --patience 5 \
        --ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE};
}


function generate () {
    model=${SAVE_DIR}/checkpoint_best.pt
    FILE_PREF=${SAVE_DIR}/output
    RESULT_FILE=${SAVE_DIR}/result.txt

    fairseq-generate $PATH_2_DATA \
        --user-dir $USER_DIR \
        --path $model \
        --truncate-source \
        --task translation_in_same_language \
        --gen-subset test \
        -t $TARGET -s $SOURCE \
        --sacrebleu --remove-bpe 'sentencepiece' \
        --max-len-b 500 --beam 5 \
        --batch-size 32 --langs $langs > $FILE_PREF;

    cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp
    ORIGINAL_TGT="${DATA_DIR}/${DATA_SIZE}/test.buggy-fixed.fixed";
    less ${ORIGINAL_TGT} > $FILE_PREF.ref
    printf "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
    python evaluator.py --ref ${FILE_PREF}.ref --pre ${FILE_PREF}.hyp
    python evaluator.py --ref ${FILE_PREF}.ref --pre ${FILE_PREF}.hyp >> ${RESULT_FILE}
    cd ..;
}

fine_tune
generate
