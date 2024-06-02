conda create --name dma python=3.10
conda activate dma

conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

conda install tensorflow=2.15 -c conda-forge

pip install "gymnasium[atari]"
ale-import-roms ./ROMS/ # The part of "./ROMS/" should be changed to the location of downloaded roms.

pip install gin-config
pip install tqdm
pip install git+https://github.com/google/dopamine.git

pip install causal-conv1d=1.2.0.post2
pip install mamba-ssm=1.2.0.post1

#pip install wandb
