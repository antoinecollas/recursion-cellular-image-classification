# launch instance
./setup_instance_from_snapshot_gcp.sh

# transfer repo on gcp
./transfer_on_gcp.sh

# install env
source config_gcp.sh
gcloud compute ssh $INSTANCE_NAME --command "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate recursion-cellular-image-classification && screen -d -m python main.py"
gcloud compute ssh $INSTANCE_NAME --command "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate recursion-cellular-image-classification && screen -d -m python -m tensorboard.main --logdir=board/"