# VQA
## Download coco2017
python download_coco.py \
  --ann-root datasets/llava_image_data/annotations_trainval2017/annotations/ \
  --split train2017 \
  --out-dir datasets/llava_image_data/coco \
  --workers 4