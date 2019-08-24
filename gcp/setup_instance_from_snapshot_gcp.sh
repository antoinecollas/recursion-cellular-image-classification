source config_gcp.sh

gcloud compute disks create $INSTANCE_NAME \
    --project=$PROJECT \
    --size=200GB \
    --zone $ZONE \
    --source-snapshot="recursion-cellular-image-classification-v2" \
    --type=pd-ssd

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