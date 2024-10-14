#!/bin/bash

hf_model_repo=$1
model_dir=$2
checkpoint_dir=$3
engine_dir=$4

DTYPE=${5:-"float16"}
TP_SIZE=${6:-1}
WORKERS=${7:-8}

echo $hf_model_repo
echo $model_dir
echo $checkpoint_dir
echo $engine_dir

echo $DTYPE
echo $TP_SIZE
echo $WORKERS
exit 0

if [[ $# -ne 4 ]] then
	echo "usage: $0 hf_model_repo model_dir checkpoint_dir engine_dir"
	exit 1
fi

if (python ./download_model.py --model_dir=$hf_model_repo --output_dir=$model_dir) then
	echo "Model download Complete. output Path : $model_dir"
else
	echo "Model download Fail."
	rm -rf $model_dir
fi

if (python ./convert_checkpoint.py --model_dir=$model_dir --output_dir=$checkpoint_dir --dtype=DTYPE --tp_size=TP_SIZE --workers=WORKERS) then
	echo "Convert checkpoint Complete. ckpt Path : $checkpoint_dir"
else
	echo "Convert Model Fail"
	rm -rf $model_dir $checkpoint_dir
fi

if (trtllm-build --checkpoint_dir $checkpoint_dir --output_dir $engine_dir --gemm-plugin auto --context_fmha disable --max_batch_size 1 --max_input_len 512 --max_output_len 128 --max_beam_width 1) then
	echo "trtllm-build complete. engine Path : $engine_dir"
else
	echo "trtllm-build Fail"
	rm -rf $model_dir $checkpoint_dir $engine_dir
fi
