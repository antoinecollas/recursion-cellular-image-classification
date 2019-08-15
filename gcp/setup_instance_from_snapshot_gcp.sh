source config_gcp.sh

gcloud compute disks create $INSTANCE_NAME \
    --project=$PROJECT \
    --size=200GB \
    --zone $ZONE \
    --source-snapshot="recursion-cellular-image-classification" \
    --type=pd-ssd

gcloud beta compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --custom-cpu=$NB_CPU \
    --custom-memory=$MEMORY_SIZE \
    --maintenance-policy=TERMINATE \
    --accelerator=$ACCELERATOR \
    --disk=name=$INSTANCE_NAME,device-name=$INSTANCE_NAME,mode=rw,boot=yes,auto-delete=yes \
    --reservation-affinity=any \
    --preemptible \
    --address=$IP_ADDRESS