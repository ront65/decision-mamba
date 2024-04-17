
conda create --name dma python=3.10
conda activate dma

#conda install pytorch cudatoolkit=11.3 -c pytorch
#conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge # This was a good one
#conda install pytorch==1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia #May need to install torchvision and torchaudio
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

conda install tensorflow=2.15 -c conda-forge

# pip install cmake
# pip install atari-py
pip install "gymnasium[atari]"
ale-import-roms ./ROMS/ # The part of "./ROMS/" should be changed to the location of downloaded roms.


#pip install pyprind
#pip install absl-py
pip install gin-config #Todo: May be removed, as it seems that  gin in imported but not used
#pip install gym==0.23.0
pip install tqdm
#pip install scikit-build
#pip install blosc #Todo: May be removed, as it seems that blosc is imported but not used
pip install git+https://github.com/google/dopamine.git

#need to install causal-conv1d and mamba
#need to download dqn dataset
pip install wandb
