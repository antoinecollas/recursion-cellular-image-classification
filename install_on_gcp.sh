source /opt/anaconda3/etc/profile.d/conda.sh
conda create --name recursion-cellular-image-classification python=3.7.3 --yes
conda activate recursion-cellular-image-classification
PATH=/home/antoinecollas/.local/bin:$PATH
pip install --user -r requirement.txt