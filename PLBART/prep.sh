#!/usr/bin/env bash

CODE_DIR=`realpath ../`;

SRCDIR="${CODE_DIR}/data/PLBART_DATA";
SPMDIR="${CODE_DIR}/models/pretrained";

function spm_preprocess () {
    for LANG in  small medium; do # small; do
        for src in parent_contexed_code; do #parent_buggy_only parent_buggy_commit; do #parent_code parent_commit; do
            for tgt in child_code; do # child_full_code; do
                for SPLIT in train eval test; do
                    if [[ $SPLIT == 'test' ]]; then
                        MAX_LEN=9999 # we do not truncate test sequences
                    else
                        MAX_LEN=512
                    fi
                    python encode.py \
                        --model-file ${SPMDIR}/sentencepiece.bpe.model \
                        --src_file ${SRCDIR}/${LANG}.${src}.${tgt}/${SPLIT}.buggy-fixed.buggy \
                        --tgt_file ${SRCDIR}/${LANG}.${src}.${tgt}/${SPLIT}.buggy-fixed.fixed \
                        --output_dir ${SRCDIR}/${LANG}.${src}.${tgt}/ \
                        --src_lang source --tgt_lang target \
                        --pref $SPLIT --max_len $MAX_LEN \
                        --workers 60;
                done
            done
        done
    done
}

function binarize () {
    for LANG in small medium; do # small; do
        for src in parent_contexed_commit; do #parent_buggy_only parent_buggy_commit; do # parent_code parent_commit; do
            for tgt in child_code; do # child_full_code; do
                fairseq-preprocess \
                    --source-lang source \
                    --target-lang target \
                    --trainpref ${SRCDIR}/${LANG}.${src}.${tgt}/train.spm \
                    --validpref ${SRCDIR}/${LANG}.${src}.${tgt}/eval.spm \
                    --testpref ${SRCDIR}/${LANG}.${src}.${tgt}/test.spm \
                    --destdir ${SRCDIR}/${LANG}.${src}.${tgt}/data-bin \
                    --thresholdtgt 0 \
                    --thresholdsrc 0 \
                    --workers 60 \
                    --srcdict ${SPMDIR}/dict.txt \
                    --tgtdict ${SPMDIR}/dict.txt;
            done
        done
    done
}

spm_preprocess
binarize
