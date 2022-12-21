cd ~
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2  --extra-index-url https://download.pytorch.org/whl/cu110
echo "Install mmcv packages….\n"

pip install -U openmim

mim install mmcv-full

pip install mmsegmentation

#기타 라이브러리 설치

echo "Install other necessary libraries….\n"

pip install wandb
pip install plotly
pip install matplotlib
pip install pandas
pip install tqdm

#Git repo clone 

MMSEG="mmsegmentation"

echo "Now cloning mmseg repository\n"

if [ -d $MMSEG ]; then
    echo "Already existing mmsegmentation. no need to clone."
else
    echo "No folder named mmseg. you should clone"
    git clone https://github.com/open-mmlab/mmsegmentation.git
fi
cd mmsegmentation
pip install -r  requirements.txt

# Prepare for training
wandb login