source /opt/anaconda3/etc/profile.d/conda.sh
conda create --name recursion-cellular-image-classification python=3.7.3 --yes
conda activate recursion-cellular-image-classification
PATH=/home/antoinecollas/.local/bin:$PATH
pip install --user -r requirement.txt
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user