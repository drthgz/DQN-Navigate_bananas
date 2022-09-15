# Problem statement
Train an agent to navigate and collect bananas in Banana Collector Unity ML-Agent environment. Environment is a large, square world containing yellow bananas and blue bananas. Collecting a yellow banana returns a rewards of +1 and collecting a blue return a reward of -1. The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

# Description of learning algorithm
Deep Q-Learning was used the implementation of this solution. To take advantage of high-dimensional sensory inputs along end-to-end reinforcement learning. A Deep Q-network contains multilayer feedfoward fully connect network or Covolutional Network model initialized, then based on the sensory weights actions are chosen. The environment returns rewards or punishments, using these values the weights in model is updated and adjusted over several episodes.


# HyperParameters Used:
1) Number of episodes = 2000
2) Max timesteps = 1000
3) Batch size = 64
4) GAMMA = 0.99
5) TAU = 1e-3
6) Learning Rate = 5e-4
7) Buffer size = 1e5
8) Epsilon start = 1.0
9) Epsilon end = 0.01
10) Epsilon decay = 0.995
11) Target score = 14

# Neural Network Architecture Used:
    A multilayer forward popagating neural network with 2 hidden fully connected layers each containing 64 neurons
    ReLu (Rectified Linear Unit) Activiation function were used over output of each hidden fully connected layer.

# Plot of rewards
A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment. 
![Plot of Score over Episodes][output.png]
A score of greater than 13 was solved in 478 episodes. Final score: 14.03.
The environment was solved in fewer than 1800 episodes!

# Ideas for future work
1) Implementation a Convolutional Neural Network instead of Fully Connected Neural Network model
2) Implementation a Deep Learning more detailed feature extraction to train the agent
3) Implementation Duelling Deep Q-Network
4) Implementation of Double Deep Q-Network
5) Implementation of Rainbow Algorithm
6) Try different weight
    a) mini banches
    b) Quicker decay of epsilon

Implementation of any one of these or combination of these is likely to produce better results. 