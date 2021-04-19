cuda_device=$1;
data_name=$2;
source=$3;
target=$4;
lr=5e-5;

batch_size=16;
beam_size=5;
source_length=256;
target_length=256;
data_dir=../data/PLBART_DATA/${data_name}.${source}.${target}/
export CUDA_VISIBLE_DEVICES=${cuda_device}
mkdir -p saved_models;
output_dir=saved_models/${data_name}-${source}-${target}

function train () {
    best_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin;
    if [[ -f ${best_model} ]]; then
	    echo "Found a trained checkpoint, Not performing training!";
	    return;
	  fi
    train_file=$data_dir/train.buggy-fixed.buggy,$data_dir/train.buggy-fixed.fixed
    dev_file=$data_dir/eval.buggy-fixed.buggy,$data_dir/eval.buggy-fixed.fixed
    eval_steps=5000
    train_steps=100000
    pretrained_model="microsoft/graphcodebert-base";
    python run.py \
        --do_train --do_eval \
        --model_type roberta --config_name roberta-base --tokenizer_name roberta-base \
        --model_name_or_path $pretrained_model \
        --train_filename $train_file --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size --eval_batch_size 8 --gradient_accumulation_steps 2 \
        --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps;
}




function evaluate () {
    dev_file=$data_dir/eval.buggy-fixed.buggy,$data_dir/eval.buggy-fixed.fixed
    test_file=$data_dir/test.buggy-fixed.buggy,$data_dir/test.buggy-fixed.fixed
    test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
    python run.py\
        --do_test \
        --model_type roberta --model_name_or_path roberta-base --config_name roberta-base \
        --tokenizer_name roberta-base  --load_model_path $test_model \
        --dev_filename $dev_file --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size 8;
}


train;
evaluate;
