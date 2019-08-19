# launch instance
./setup_instance_from_snapshot_gcp.sh

# transfer repo on gcp
./transfer_on_gcp.sh

# install env
source config_gcp.sh
gcloud compute ssh $INSTANCE_NAME --command "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate recursion-cellular-image-classification && screen -d -m python -m tensorboard.main --logdir=board/"
# gcloud compute ssh $INSTANCE_NAME --command "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate recursion-cellular-image-classification && screen -d -m python main.py"

# CUDA_VISIBLE_DEVICES=1,2 python main.py

# Examples of 4 experiments (1 per GPU):
# CUDA_VISIBLE_DEVICES=0 screen -d -m python main.py
# CUDA_VISIBLE_DEVICES=1 screen -d -m python main.py --pretrain
# CUDA_VISIBLE_DEVICES=2 screen -d -m python main.py --scheduler
# CUDA_VISIBLE_DEVICES=3 screen -d -m python main.py --pretrain --scheduler

# CUDA_VISIBLE_DEVICES=0 screen -d -m python main.py --pretrain --scheduler --lr 0.1
# CUDA_VISIBLE_DEVICES=1 screen -d -m python main.py --pretrain --scheduler --lr 0.01
# CUDA_VISIBLE_DEVICES=2 screen -d -m python main.py --pretrain --scheduler --lr 0.001
# CUDA_VISIBLE_DEVICES=3 screen -d -m python main.py --pretrain --scheduler --lr 0.0001