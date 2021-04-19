cuda_device=$1;
data_name=$2;
lr=5e-5;

batch_size=16;
beam_size=5;
source_length=250;
target_length=250;
data_dir=../data/tufano-${data_name}
export CUDA_VISIBLE_DEVICES=${cuda_device}
mkdir -p saved_models;

source1=$3;
source2=$4;
target=$5;
output_dir=saved_models/${data_name}.${source1}.${source2}.${target}

function train () {
    train_file="${data_dir}/train/data.${source1},${data_dir}/train/data.${source2},${data_dir}/train/data.${target}";
    dev_file="${data_dir}/eval/data.${source1},${data_dir}/eval/data.${source2},${data_dir}/eval/data.${target}";
    eval_steps=5000
    train_steps=100000
    pretrained_model="microsoft/codebert-base";
    python run.py \
        --do_train --do_eval \
        --model_type roberta --config_name roberta-base --tokenizer_name roberta-base \
        --model_name_or_path $pretrained_model \
        --train_filename $train_file --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size --eval_batch_size 4 --gradient_accumulation_steps 4 \
        --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps;
}



function evaluate () {
    dev_file="${data_dir}/eval/data.${source1},${data_dir}/eval/data.${source2},${data_dir}/eval/data.${target}";
    test_file="${data_dir}/test/data.${source1},${data_dir}/test/data.${source2},${data_dir}/test/data.${target}";
    test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
    OUTPUT_FILE=${output_dir}/evaluation_result.txt;
    python run.py\
        --do_test \
        --model_type roberta --model_name_or_path roberta-base --config_name roberta-base \
        --tokenizer_name roberta-base  --load_model_path $test_model \
        --dev_filename $dev_file --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size 16 | tee ${OUTPUT_FILE};
}


train;
evaluate;
