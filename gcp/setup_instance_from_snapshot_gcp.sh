source config_gcp.sh

gcloud beta compute disks create $INSTANCE_NAME \
    --project=$PROJECT \
    --type=pd-ssd \
    --size=500GB \
    --zone=$ZONE \
    --source-snapshot=recursion-cellular-image-classification-v2 \
    --physical-block-size=16384

gcloud beta compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --maintenance-policy=TERMINATE \
    --accelerator=$ACCELERATOR \
    --disk=name=$INSTANCE_NAME,device-name=$INSTANCE_NAME,mode=rw,boot=yes,auto-delete=yes \
    --reservation-affinity=any \
    --preemptible \
    --address=$IP_ADDRESS