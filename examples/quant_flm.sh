#! /bin/bash

# 检查命令行参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <JD_PATH> <MODEL_NAME>"
    exit 1
fi

# 从命令行参数获取路径和模型名称
JD_PATH="$1"
MODEL_NAME="$2"

# 0. set env
eval "$(conda shell.bash hook)"
conda activate autoawq
CUDA_VISIBLE_DEVICES=2,3,4,5

LOCAL_PREFIX="/data/ynt/models"

# 1. upload model from jd to ks3 and download to local
MODEL_PATH="$LOCAL_PREFIX/$MODEL_NAME"
KS3_PATH="ks3://baai-cofe/ynt/models/$MODEL_NAME"
echo "Uploading model from JD to ks3..."
ssh jd_infer ks3util cp -ru "$JD_PATH" "$KS3_PATH"
echo "Downloading model from ks3 to local..."
ks3util cp -ru "$KS3_PATH" "$LOCAL_PREFIX"

# 2. padd model
PADDED_MODEL_PATH="$LOCAL_PREFIX/$MODEL_NAME-padded"
echo "Padding model..."
python padding.py --model_path "$MODEL_PATH" --output_path "$PADDED_MODEL_PATH"

# 3. quant model
QUANT_MODEL_PATH="$LOCAL_PREFIX/$MODEL_NAME-AWQ"
echo "Quantizing model..."
python cli.py --hf_model_path "$PADDED_MODEL_PATH" --local_save_path "$QUANT_MODEL_PATH" --quant_name "$MODEL_NAME" --no-safetensors

# 4. cleanup
echo "Cleaning up..."
rm -rf "$PADDED_MODEL_PATH"

echo "Done"
