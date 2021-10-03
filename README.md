#Project Name: Navigation
#1. Description of Environment
We need to train an agent to navigate and collect yellow bananas. For every yellow banana collected the reward is +1 and for every blue banana the reward is -1. The state space is 37 dimension. It includes the agent’s velocity and ray-based perception of objects in the front of the agent. The agent has a total of 4 actions. The agent can move forward, backward, left, right, and these actions are indicated in numerical values 0, 1, 2, 3. The problem is considered solved if the average reward is at least 13+ over 100 episodes. 

#2. Description of Installation 
We use a docker with the nvidia driver and isolate the environment. Inside the docker, we then create a virtual environment to use Python 3.6. In the virtual environment, we install pytorch and unityagents. The installation needs to be improved in the future works, but currently, the installation works. Improvements necessary for installation will be mentioned in future works. 

#3. Installation Guide 
3.1 This installation guide assumes that 
     -OS is Ubuntu 16.04. 
     -Docker is installed  (https://docs.docker.com/engine/install/ubuntu/) 
     -Nvidia driver is installed Cuda is installed 
     -Cudnn is installed 
     -nvidia-docker is installed (https://github.com/NVIDIA/nvidia-docker) 
     -git is installed 

3.2 Clone the repository 

   git clone https://github.com/wchung91/p1_navigation.git

3.3 Build the dockerfile. Run the command below in the terminal and it will create an image named rl_env.

   sudo docker build . --rm -t rl_env

If you use a different GPU from RTX 2080, you need to change the dockerfile. Open the dockerfile and change “pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime” to the docker image corresponding to the cuda and cudnn installed on your computer. 

3.4 Create a docker container from the docker image, but you need to change “/home/wally-server/Documents/p1_navigation” in the command below and then run the command. ““/home/wally-server/Documents/p1_navigation” is the directory of the volume. You should change “/home/wally-server/Documents/p1_navigation” to the path to the cloned repository. That way the docker container has access to the files cloned from the repository. One you changed the command, run the command. 

   sudo docker run -it -d --gpus all -v /home/wally-server/Documents/p1_navigation:/workspace --name p1_navigation_container rl_env /bin/bash

3.5 To access the container run,  
 
   sudo docker exec -it p1_navigation_container /bin/bash

3.6 Inside the container run the command below to initialize conda with bash 

   conda init bash

3.7 You need to close and reopen a new terminal. You can do that with the command from 3.5. Create a virtual environment named “p1_env” with python 3.6 with the following code 

   conda create -n p1_env -y python=3.6

3.8 Activate the environment “p1_env” with the command below. 

   conda activate p1_env

3.9 Inside the virtual environment, install pytorch with the command below. You’ll have to install the correct pytorch version depending on your cuda and cudnn version. 

   pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

3.10 Install unityagents with the following code. 

   pip install unityagents

3.11 Download the unity environments with the following commands. Since we are using a docker, you’ll have to use Banana_Linux_NoVis because no display is available. 

   wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip && unzip Banana_Linux.zip
   wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip && unzip Banana_Linux_NoVis.zip

3.12 To run the training code, go to main.py and set “Train = 1”. Then, run the command below

   python main.py 

The code will print the average scores, and it will create a figure called “ScoresTraining.png”

3.13 To run the testing code, go to main.py and set “Train = 0”. Then, run the command below 

   python main.py

The code will print the average scores, and it will create a figure called “TestScores.png”

#4. About the Code 
main.py - contains the main method for running the code. The code is divided into training code and testing code 
dqn_agent.py - contains the code for the dqn_agent and experience replay buffer. The dqn_agent stores data from experiences, chooses actions, and learns from the experiences.
model.py - contains the deep Q-Network. 

