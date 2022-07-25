# vlp_gat

![fig2](https://user-images.githubusercontent.com/75057781/180788286-191514ea-25ef-4e07-9ee6-f45b9354b45c.png)

** Need coco_ir folder **

oscar -> run_retrieval.py

(a) Train
CUDA_VISIBLE_DEVICES=0 python run_retrieval.py \
	--model_name_or_path /HDD/bpilseo/base-vg-labels/ep_67_588997 \
	--do_train \
	--do_lower_case \
	--evaluate_during_training \
	--num_captions_per_img_val 20 \
	--eval_caption_index_file minival_caption_indexs_top20.pt \
	--per_gpu_train_batch_size 32 \
	--learning_rate 0.00002 \
	--num_train_epochs 30 \
	--weight_decay 0.05 \
	--save_steps 5000 \
	--add_od_labels \
	--od_label_type vg \
	--max_seq_length 70 \
	--output_dir output/

(b) Inference
CUDA_VISIBLE_DEVICES=0 python run_retrieval.py \
	--do_test \
	--do_eval \
	--test_split test \
	--num_captions_per_img_val 5 \
	--eval_img_keys_file /HDD/bpilseo/coco_ir/test_img_keys_1k.tsv \
	--cross_image_eval \
	--per_gpu_eval_batch_size 64 \
	--eval_model_dir ./output/checkpoint-29-531060
