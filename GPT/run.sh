#!/usr/bin/bash

cuda_device=$1;
data_name=$2;
source=$3;
target=$4;
model=$5;

if [[ ${data_name} = "small" ]]; then
    BLOCK_SIZE=512;
elif [[ ${data_name} = "medium" ]]; then
    BLOCK_SIZE=1024;
else
    echo "Data Name must be either small or medium";
    exit;
fi

if [[ $model = "normal" ]]; then
    PRETRAINDIR="microsoft/CodeGPT-small";
    MODEL_SHAPE="normal";
elif [[ $model = "adapted" ]]; then
    PRETRAINDIR="microsoft/CodeGPT-small-java-adaptedGPT2";
    MODEL_SHAPE="adapted";
else
    echo "Model Type Must be Eith \"normal\" or \"adapted\"";
    exit;
fi

data_dir=../data/PLBART_DATA/${data_name}.${source}.${target}/
export CUDA_VISIBLE_DEVICES=${cuda_device}
mkdir -p saved_models;
output_dir=saved_models/${data_name}-${source}-${target}-${MODEL_SHAPE};
mkdir -p ${output_dir};

function train(){
    CHECKPOINT_DIR=${output_dir}/checkpoint-best
    if [[ -f ${CHECKPOINT_DIR} ]]; then
        echo "Found a trained checkpoint, Not performing training!";
	      return;
    fi
    LANG=java;
    LOGFILE=${output_dir}/training.log;

    python run.py \
            --data_dir=${data_dir} \
            --langs=$LANG \
            --output_dir=${output_dir} \
            --pretrain_dir=$PRETRAINDIR \
            --log_file=$LOGFILE \
            --model_type=gpt2 \
            --block_size=${BLOCK_SIZE} \
            --do_train \
            --node_index 0 \
            --learning_rate=5e-5 \
            --weight_decay=0.01 \
            --evaluate_during_training \
            --per_gpu_train_batch_size=2 \
            --per_gpu_eval_batch_size=8 \
            --gradient_accumulation_steps=8 \
            --num_train_epochs=30 \
            --logging_steps=100 \
            --overwrite_output_dir \
            --seed=42;
}

function evaluate() {
    LANG=java
    PRETRAINDIR=${output_dir}/checkpoint-best
    LOGFILE=${output_dir}/evaluation.log

    python -u run.py \
            --data_dir=${data_dir} \
            --langs=$LANG \
            --output_dir=${output_dir} \
            --pretrain_dir=$PRETRAINDIR \
            --log_file=$LOGFILE \
            --model_type=gpt2 \
            --block_size=512 \
            --do_eval \
            --do_infer \
            --logging_steps=100 \
            --seed=42
}

train;
evaluate;
