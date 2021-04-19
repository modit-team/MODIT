DATA_BASE_DIR=`realpath "data/"`;
MODEL_BASE_DIR=`realpath "models/"`;

BATCH_SIZE=4;
UPDATE_FREQ=8;
LEARNING_RATE=5e-5;

function generate () {
	SPLIT=$1;
	DATA_NAME=$2;
	SOURCE=$3;
	TARGET=$4;
	PATH_2_DATA="${DATA_BASE_DIR}/${SPLIT}/${DATA_NAME}";
	SAVE_DIR="${MODEL_BASE_DIR}/${SPLIT}/${DATA_NAME}-${SOURCE}-${TARGET}";
	OUTPUT_FILE="${SAVE_DIR}/evaluation.log";
	model="${SAVE_DIR}/checkpoint_best.pt";
	FILE_PREF="${SAVE_DIR}/output";
	RESULT_FILE="${SAVE_DIR}/result.txt";
	TEST_TARGET_FILE=`echo "$TARGET" | tr "_" "."`;
	GOLD_TARGET_FILE="${DATA_BASE_DIR}/${SPLIT}/test/${TEST_TARGET_FILE}";
	GENERATED_CODE="${SAVE_DIR}/generated_results.txt";

	fairseq-generate $PATH_2_DATA \
  		--path $model \
  		--truncate-source \
  		--task translation \
  		--truncate-source \
  		--source-lang ${SOURCE} --target-lang ${TARGET} \
  		--gen-subset test \
  		--sacrebleu --remove-bpe 'sentencepiece' \
		  --max-len-b 500 --beam 10 --nbest 10 \
  		--batch-size 2 > $GENERATED_CODE;

  python parse_generated.py --input ${GENERATED_CODE} --output ${GENERATED_CODE}.hyp;
}

function evaluate () {
  	SPLIT=$1;
	DATA_NAME=$2;
	SOURCE=$3;
	TARGET=$4;
	SAVE_DIR="${MODEL_BASE_DIR}/${SPLIT}/${DATA_NAME}-${SOURCE}-${TARGET}";
	FILE_PREF="${SAVE_DIR}/output";
	RESULT_FILE="${SAVE_DIR}/result.txt";
	TEST_TARGET_FILE=`echo "$TARGET" | tr "_" "."`;
	GOLD_TARGET_FILE="${DATA_BASE_DIR}/${SPLIT}/test/${TEST_TARGET_FILE}";
	GENERATED_CODE="${SAVE_DIR}/generated_results.txt";
	
	if [ ! -f "${GENERATED_CODE}.hyp" ]; then
	    python parse_generated.py --input ${GENERATED_CODE} --output ${GENERATED_CODE}.hyp;
	fi

	printf "BLEU: \t" >> ${RESULT_FILE}
	python evaluator.py \
	        --ref ${GOLD_TARGET_FILE} \
	        --pre ${GENERATED_CODE}.hyp \
	        --beam 10 --nbest 1 2 5 10 >> ${RESULT_FILE}
	echo "CodeBLEU " >> ${RESULT_FILE}
	cd CodeBLEU;
	#python calc_code_bleu.py \
	#        --refs ${GOLD_TARGET_FILE} \
	#        --hyp ${GENERATED_CODE}.hyp \
	#        --lang java \
	#        --beam 10 --nbest 1 2 5 10 >> ${RESULT_FILE}
	#cd ..;
}

function test(){
	generate $1 $2 $3 $4;
	evaluate $1 $2 $3 $4;
}


export CUDA_VISIBLE_DEVICES=$1;
export MKL_THREADING_LAYER=GNU;
#generate time_split binary-30000 parent_commit child_code;
#evaluate time_split binary-30000 parent_commit child_code;

test time_split binary-30000 parent_code child_full_code;

