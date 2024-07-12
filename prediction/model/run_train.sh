#!/bin/bash
python model_train_vad.py --num_units 64 \
	--lr1 1e-5 \
	--lr2 1e-4 \
	--lr_decay 0.8 \
	--epoch 5 \
	--loss_weight 10 \
	--data_name data/audio_0426En \
	--is_diff False \
	--train_vgg True \
	--trained_layers 12 \
	--train_name asthma_model
