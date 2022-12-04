export EXP_NAME=exp14
export STEP=25000
export BEST_STEP=240000
export BEST_EPOCH=9
export PREFIX=$EXP_NAME
# export PREFIX=exp7
export CUDA_VISIBLE_DEVICES=7
# python run_func_OCD.py --eval 0 --config_path 
export PARENT=/a/home/cc/students/cs/ohadr/netapp/OCD/experiments
echo Starting

python run_func_OCD.py --eval 1 --train 0 --backbone_path ./checkpoints/checkpoint_lenet5.pth \
--config_path \./configs/train_bert2.json --data_train_path ./data/mnist --data_test_path ./data/mnist --datatype bert2 --precompute_all 0 \
--diffusion_model_path $PARENT/exp13model_checkpoint_epoch$BEST_EPOCH\_step$BEST_STEP\_databert2.pt \
--scale_model_path $PARENT/exp13scale_model_checkpoint_epoch$BEST_EPOCH\_loss$BEST_STEP\_databert2.pt \
  --tensorboard_path ./logs/$EXP_NAME
  
 --diffusion_model_path $PARENT/exp13ema_checkpoint_epoch$BEST_EPOCH\_step$BEST_STEP\_databert2.pt \
#  --diffusion_model_path $PARENT/$PREFIX"model_checkpoint_epoch"$BEST_EPOCH\_step$BEST_STEP\_databert.pt \

#  --scale_model_path $PARENT/$PREFIX"scale_model_checkpoint_epoch"$BEST_EPOCH\_loss$BEST_STEP\_databert.pt \
#  --diffusion_model_path $PARENT/$PREFIX"ema_checkpoint_epoch"$BEST_EPOCH\_step$BEST_STEP\_databert.pt \



# python run_func_OCD.py --eval 0 --backbone_path ./checkpoints/checkpoint_lenet5.pth \
# --config_path \./configs/train_mnist.json \
#  --data_train_path./data/mnist --data_test_path ./data/mnist --datatype mnist --precompute_all 0 \
# --diffusion_model_path ./experiments/$EXP_NAME --scale_model_path ./experiments/$EXP_NAME --tensorboard_path ./logs/$EXP_NAME