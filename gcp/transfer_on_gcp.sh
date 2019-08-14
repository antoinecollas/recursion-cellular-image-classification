# load configuration
source config_gcp.sh

# prepare copy of repo
cd ../..
REPO=recursion-cellular-image-classification
TEMP_DIR=temp_dir
cp -r $REPO $TEMP_DIR
cd $TEMP_DIR
git clean -xdf
rm -rf .git

# transfer repo
gcloud compute scp --recurse * "$INSTANCE_NAME:~"
while [ $? -ne 0 ]; do
    sleep 15
    gcloud compute scp --recurse * "$INSTANCE_NAME:~"
done

# delete copy of repo
cd ..
rm -rf $TEMP_DIR