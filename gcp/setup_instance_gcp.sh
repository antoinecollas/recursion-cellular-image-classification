#Tuto deep learning VM: https://cloud.google.com/deep-learning-vm/docs/quickstart-cli?hl=fr
#instances: https://cloud.google.com/compute/pricing?hl=fr
source config_gcp.sh

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --image-family=$IMAGE_FAMILY \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --custom-cpu=$NB_CPU \
    --custom-memory=$MEMORY_SIZE \
    --accelerator=$ACCELERATOR \
    --preemptible \
    --address=$IP_ADDRESS \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd