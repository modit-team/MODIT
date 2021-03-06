#!/bin/bash

# We assume that only 12 GB GPU is used for training and testing.

current_dir=$(pwd);
base_path=$(realpath ../);

PLBART_PATH="${base_path}/PLBART";
cd ${PLBART_PATH};
echo "#############################################################################################";
echo "Experiment for LSTM-S2S";
echo "=============================================================================================";
echo "Small Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash lstm_s2s.sh 0 small parent_contexed_commit child_code;
echo "Medium Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash lstm_s2s.sh 0 medium parent_contexed_commit child_code;
echo "#############################################################################################";

echo "#############################################################################################";
echo "Experiment for Transformer-S2S-Base";
echo "=============================================================================================";
echo "Small Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash transformer_s2s.sh 0 small parent_contexed_commit child_code;
echo "Medium Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash transformer_s2s.sh 0 medium parent_contexed_commit child_code;
echo "#############################################################################################";


echo "#############################################################################################";
echo "Experiment for PLBART";
echo "=============================================================================================";
echo "Small Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash train.sh 0 small parent_contexed_commit child_code;
echo "Medium Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash train.sh 0 medium parent_contexed_commit child_code;
echo "#############################################################################################";


CODEBERT_PATH="${base_path}/CodeBERT";
cd ${CODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for CodeBERT";
echo "=============================================================================================";
echo "Small Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash run.sh 0 small parent_contexed_commit child_code;
echo "Medium Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash run.sh 0 medium parent_contexed_commit child_code;
echo "#############################################################################################";


GRAPH_CODEBERT_PATH="${base_path}/GraphCodeBERT";
cd ${GRAPH_CODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for GraphCodeBERT";
echo "=============================================================================================";
echo "Small Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash run.sh 0 small parent_contexed_commit child_code;
echo "Medium Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash run.sh 0 medium parent_contexed_commit child_code;
echo "#############################################################################################";


CodeGPT="${base_path}/GPT";
cd ${GRAPH_CODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for CodeGPT";
echo "=============================================================================================";
echo "Small Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash run.sh 0 small parent_contexed_commit child_code adapted;
echo "Medium Dataset:"
echo "---------------------------------------------------------------------------------------------";
bash run.sh 0 medium parent_contexed_commit child_code adapted;
echo "#############################################################################################";
