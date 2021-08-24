nvidia-smi
MODEL=NAONet_A_imagenet
OUTPUT_DIR=exp/$MODEL
DATA_DIR=imagenet/raw-data

mkdir -p $OUTPUT_DIR

fixed_arc="0 7 1 15 2 8 1 7 0 6 1 7 0 8 4 7 1 5 0 7 0 7 1 13 0 6 0 14 0 9 1 10 0 14 2 6 1 11 0 7"

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --batch_size=512 \
  --epochs=250 \
  --arch="$fixed_arc" \
  --channels=42 \
  --use_aux_head \
  --lr=0.4 \
  --keep_prob=1.0 \
  --drop_path_keep_prob=1.0 | tee -a $OUTPUT_DIR/train.log
