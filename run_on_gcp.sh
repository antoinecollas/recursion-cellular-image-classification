# load configuration
cd ../scripts_gcloud/
source config_gcloud.sh

# launch instance
./setup_gpu_instance.sh
cd -

# transfer repo on gcp
./transfer_gcp.sh

# install env
gcloud compute ssh $INSTANCE_NAME --command "source install_on_gcp.sh"