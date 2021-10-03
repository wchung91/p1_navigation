# p1_navigation
Project Submission 


1. ENVRIONMENT DESCRIPTION
We use a docker to isolate the environment. Then, inside the docker, we use a virtual environment to 


DOCKER BUILD 
sudo docker build . --rm -t rl_env

-Need to change FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

DOCKER RUN 
sudo docker run -it -d --gpus all -v /home/wally-server/Documents/p1_navigation:/workspace --name p1_navigation_container rl_env /bin/bash

DOCKER EXEC
sudo docker exec -it p1_navigation_container /bin/bash

VIRTUAL ENVIRONMENT 
conda init bash

conda create -n p1_env -y python=3.6
 --> restart terminal
conda activate p1_env

INSTALL PYTORCH 
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install unityagents

wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip && unzip Banana_Linux.zip
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip && unzip Banana_Linux_NoVis.zip

#No need to do this 
INSTALL DEPENDECIES 
pip install .

git clone https://github.com/openai/gym.git
cd /gym
pip install -e .
cd ..

