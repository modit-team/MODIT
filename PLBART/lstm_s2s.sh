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
            echo "SRC: parent_code, parent_commit"
            echo
            echo "TGT: child_code, child_full_code"
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1;
DATA_SIZE=${2}.${3}.${4};

SOURCE=source;
TARGET=target;


PATH_2_DATA=${DATA_DIR}/${DATA_SIZE}/data-bin

echo "Source: $SOURCE Target: $TARGET"

MODEL_DIR="${CODE_BASE}/models/Seq2Seq";

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

BATCH_SIZE=8;
UPDATE_FREQ=2;

function fine_tune () {
    best_model=${SAVE_DIR}/checkpoint_best.pt
    if [[ -f ${best_model} ]]; then
	    echo "Found a trained checkpoint, Not performing training!";
	    return;
	  fi
    OUTPUT_FILE=${SAVE_DIR}/finetune.log

    fairseq-train $PATH_2_DATA \
        --truncate-source \
        --task translation \
        --arch lstm \
        --source-lang $SOURCE --target-lang $TARGET \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --max-epoch 30 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay --lr 0.001 --min-lr -1 \
        --warmup-updates 500 --max-update 100000 \
        --dropout 0.1 --weight-decay 0.0 \
        --seed 1234 --log-format json --log-interval 100 \
        --eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints --patience 10 \
        --ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE}
}


function generate () {
    model=${SAVE_DIR}/checkpoint_best.pt
    FILE_PREF=${SAVE_DIR}/output
    RESULT_FILE=${SAVE_DIR}/result.txt

    fairseq-generate $PATH_2_DATA \
        --path $model \
        --truncate-source \
        --task translation \
        --gen-subset test \
        -t $TARGET -s $SOURCE \
        --sacrebleu --remove-bpe 'sentencepiece' \
        --max-len-b 500 --beam 5 \
        --batch-size 4  > $FILE_PREF

    cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp
    cat $FILE_PREF | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.ref
    sacrebleu -tok 'none' -s 'none' $FILE_PREF.ref < $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE}
    printf "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
    python evaluator.py --ref ${FILE_PREF}.ref --pre ${FILE_PREF}.hyp
    python evaluator.py --ref ${FILE_PREF}.ref --pre ${FILE_PREF}.hyp >> ${RESULT_FILE}
    echo "CodeBLEU Evaluation" > ${RESULT_FILE}
    cd CodeBLEU;
    python calc_code_bleu.py --refs $FILE_PREF.ref --hyp $FILE_PREF.hyp --lang java >> ${RESULT_FILE}
    python calc_code_bleu.py --refs $FILE_PREF.ref --hyp $FILE_PREF.hyp --lang java
    cd ..;
}

fine_tune
generate
