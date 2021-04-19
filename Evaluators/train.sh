DATA_BASE_DIR=`realpath "data/"`;
MODEL_BASE_DIR=`realpath "models/"`;
USER_DIR=`pwd`;
USER_DIR="${USER_DIR}/src";
PRETRAIN="${MODEL_BASE_DIR}/plbart.trained.pt";

BATCH_SIZE=8;
UPDATE_FREQ=4;
LEARNING_RATE=5e-5;

function train(){
    SPLIT=$1;
    DATA_NAME=$2; #
    SOURCE=$3;
    TARGET=$4;
    PATH_2_DATA="${DATA_BASE_DIR}/${SPLIT}/${DATA_NAME}";
    SAVE_DIR="${MODEL_BASE_DIR}/${SPLIT}/${DATA_NAME}-${SOURCE}-${TARGET}";
    OUTPUT_FILE="${SAVE_DIR}/train.log";
    mkdir -p $SAVE_DIR;
    fairseq-train $PATH_2_DATA \
        --task translation --arch transformer --layernorm-embedding --truncate-source \
        --source-lang ${SOURCE} --target-lang ${TARGET} --lr $LEARNING_RATE \
        --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt --min-lr -1 --warmup-updates 1000 --max-update 100000 \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
        --seed 1234 --log-format json --log-interval 10 \
        --eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints --patience 10 --ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE};
}

function train_tufano(){
    SPLIT="tufano-$1"; # small/medium
    DATA_NAME=$2; # original/binary-20000
    SOURCE=$3; # parent_code/parent_commit
    TARGET=$4; # child_code/child_full_code
    if [[ ${DATA_NAME} = "original" ]]; then
        DATA_PATH="original/${SOURCE}-${TARGET}";
    else
        DATA_PATH="tufano-${DATA_NAME}";
    fi
    PATH_2_DATA="${DATA_BASE_DIR}/${SPLIT}/binaries/${DATA_PATH}";
    SAVE_DIR="${MODEL_BASE_DIR}/${SPLIT}/${DATA_NAME}-${SOURCE}-${TARGET}";
    OUTPUT_FILE="${SAVE_DIR}/train.log";
    mkdir -p $SAVE_DIR;
    fairseq-train $PATH_2_DATA \
        --task translation --arch transformer --layernorm-embedding --truncate-source \
        --source-lang ${SOURCE} --target-lang ${TARGET} --lr $LEARNING_RATE \
        --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt --min-lr -1 --warmup-updates 1000 --max-update 100000 \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
        --seed 1234 --log-format tqdm --max-sentences-valid 16 \
        --no-epoch-checkpoints  --ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE};
}

function train_tufano(){
    SPLIT="tufano-$1"; # small/medium
    DATA_NAME=$2; # original/binary-20000
    SOURCE=$3; # parent_code/parent_commit
    TARGET=$4; # child_code/child_full_code
    if [[ ${DATA_NAME} = "original" ]]; then
        DATA_PATH="original/${SOURCE}-${TARGET}";
    else
        DATA_PATH="tufano-${DATA_NAME}";
    fi
    PATH_2_DATA="${DATA_BASE_DIR}/${SPLIT}/binaries/${DATA_PATH}";
    SAVE_DIR="${MODEL_BASE_DIR}/${SPLIT}/${DATA_NAME}-${SOURCE}-${TARGET}";
    OUTPUT_FILE="${SAVE_DIR}/train.log";
    mkdir -p $SAVE_DIR;
    fairseq-train $PATH_2_DATA \
        --user-dir $USER_DIR --truncate-source \
        --langs $langs --task translation_in_same_language \
        --arch mbart_base --layernorm-embedding \
        --source-lang ${SOURCE} --target-lang ${TARGET} --lr $LEARNING_RATE \
        --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt --min-lr -1 --warmup-updates 1000 --max-update 100000 \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
        --seed 1234 --log-format json --max-sentences-valid 16 \
        --no-epoch-checkpoints  --ddp-backend no_c10d --save-dir $SAVE_DIR \
        --restore-file $PRETRAIN --reset-dataloader \
        --reset-optimizer --reset-meters --reset-lr-scheduler \
        --eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --patience 10 \
        2>&1 | tee ${OUTPUT_FILE};
}


export CUDA_VISIBLE_DEVICES=$1;
export MKL_THREADING_LAYER=GNU;
#train time_split binary-30000 parent_commit child_full_code
#train_tufano small binary-30000 parent_code child_code;
#train_tufano small binary-30000 parent_commit child_code;
#train_tufano small binary-30000 parent_code child_full_code;
train_tufano small binary-30000 parent_commit child_full_code;
