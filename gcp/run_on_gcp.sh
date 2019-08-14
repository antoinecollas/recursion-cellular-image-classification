# launch instance
./setup_instance_gcp.sh

# transfer repo on gcp
./transfer_on_gcp.sh

# install env
source config_gcp.sh
gcloud compute ssh $INSTANCE_NAME --command "gsutil cp -r gs://rxrx1-dataset ."
gcloud compute ssh $INSTANCE_NAME --command "mv rxrx1-dataset data"
gcloud compute ssh $INSTANCE_NAME --command "unzip data/images/train.zip -d data/train"
gcloud compute ssh $INSTANCE_NAME --command "unzip data/images/test.zip -d data/test"
gcloud compute ssh $INSTANCE_NAME --command "source install_on_gcp.sh"
gcloud compute ssh $INSTANCE_NAME --command "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate recursion-cellular-image-classification && screen -d -m python main.py"
gcloud compute ssh $INSTANCE_NAME --command "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate recursion-cellular-image-classification && screen -d -m python -m tensorboard.main --logdir=board/"