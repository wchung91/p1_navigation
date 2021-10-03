from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from dqn_agent import Agent

#Load environment
#Use Banana_Linux_NoVis with docker
#On a regular computer use Banana_Linux
#env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")


"""
    Setup Environment
"""
# get the default agent
agent = Agent(state_size=37, action_size=4, seed=1)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

"""
    Mode to Run
    Change this in order to Train or Test the agent
    Train = 1 for training the agent
    Train = 0 for testing the agent
"""
Train = 0

"""
    Train the Agent
"""
if Train ==1:
    print("Using Training Mode")
    scores = []
    scores_window = deque(maxlen=100)                      # Used to average the scores over 100 episodes
    eps = 1.0                                              # epsilon for explore/choosing best action
    n_episodes = 2000                                      # Maximum number of episodes
    max_t = 1000                                           # Maximum number of steps
    #Run each episode
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name] # Initalize environment
        state = env_info.vector_observations[0]            # Get first state
        score = 0
        n_steps = 0

        for t in range(max_t):                             # Start steps
            action = agent.act(state, eps)                 # get action
            env_info = env.step(action)[brain_name]        # Update environment based on action
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state,action, reward, next_state, done) # update agent
            state = next_state                             #update next state
            score += reward                                # update the score
            if done:
                break
        eps_end = 0.01
        eps_decay =0.995
        eps = max(eps_end, eps_decay*eps)                  # decrease epsilon

        #print scores and average scores
        print("Score: {}".format(score))
        print("Number of Steps: {}".format(n_steps))
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        #If average score is greater than 15.0, save network and stop training
        if np.mean(scores_window)>=15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            print('\nSaved Network')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    # plot the scores
    # Figure will be saved as ScoresTraining
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("ScoresTraining")


"""
    Run the Trained Agent
"""

if Train != 1:
    print("Using Testing Mode")
    scores = []
    scores_window = deque(maxlen=100)                      # Used to average the scores over 100 episodes
    eps = 1.0                                              # epsilon is set to choose the best action
    n_episodes = 201                                       # maximum number of episodes
    max_t = 1000                                           # maximum number of steps

    #Load the trained network
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    #Start the episode
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]  # Reset the environment
        state = env_info.vector_observations[0]             # Initialize the state
        score = 0
        n_steps = 0

        for t in range(max_t):
            action = agent.act(state)                      # get action
            env_info = env.step(action)[brain_name]        # Update environment based on action
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            #agent.step(state,action, reward, next_state, done)
            state = next_state                             # update next_state
            score += reward                                # update the score
            if done:
                break

        #print scores
        print("Score: {}".format(score))
        print("Number of Steps: {}".format(n_steps))
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    # plot the scores
    # Figure will be saved as TestScores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("TestScores")

"""

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))

"""
