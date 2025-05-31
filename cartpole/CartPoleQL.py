import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Discretization Configuration - we use fewer bins for dimensions that have less impact(for q learing we need discreate state 
#----------------------cartpole state-space is continuous)
BINS = (
    1,   # Cart position (simplified - not critical for balancing)
    3,   # Cart velocity    (slow,medium,fast)
    6,   # Pole angle (MOST important feature)
    3    # Pole angular velocity
)

# Create environment using Gymnasium
env = gym.make('CartPole-v1',render_mode="rgb_array")
n_actions = env.action_space.n

# Discretization function for continuous state space
def discretize_state(state):
    discretized = []
    for i in range(4):  # CartPole state-space has 4 dimensions
        # Normalize continuous value to [0, bin_count-1] range
        low = env.observation_space.low[i]
        high = env.observation_space.high[i]
        
        # Handle infinite bounds (Gymnasium has +/-inf for velocity)
        if low == -np.inf: low = -2.0  # Practical limits based on domain knowledge
        if high == np.inf: high = 2.0
        
        # Scale value to bin index
        scaled = np.interp(state[i], [low, high], [0, BINS[i]-1e-5])
        discretized.append(int(scaled))
    return tuple(discretized)

# Initialize Q-table with default value of zeros for unseen states
q_table = defaultdict(lambda: np.zeros(n_actions))

# Hyperparameters
EPISODES = 3000      # Total training episodes
ALPHA = 0.1            # Learning rate
GAMMA = 0.99           # Discount factor for future rewards
EPSILON_START = 1.0    # Initial exploration probability
EPSILON_MIN = 0.01     # Minimum exploration probability
EPSILON_DECAY = 0.995 # Exponential decay rate for exploration

# Tracking metrics
episode_rewards = []    # Total reward per episode
epsilon_values = []     # Epsilon value per episode
success_count = 0       # Count of episodes reaching max reward

# Training loop
epsilon = EPSILON_START
for episode in range(EPISODES):
    # Reset environment and discretize initial state
    state, info = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # Random action (exploration)
        else:
            action = np.argmax(q_table[state])  # Best known action (exploitation)
        
        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Discretize next state
        next_state_disc = discretize_state(next_state)
        
        # Q-learning update rule
        current_q = q_table[state][action]
        next_max_q = np.max(q_table[next_state_disc])
        
        # Calculate new Q-value using Bellman equation
        new_q = current_q + ALPHA * (reward + GAMMA * next_max_q - current_q)
        q_table[state][action] = new_q
        
        # Transition to next state
        state = next_state_disc
        total_reward += reward
        steps += 1
    
    # Update exploration rate (exponential decay)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    epsilon_values.append(epsilon)
    
    # Record metrics
    episode_rewards.append(total_reward)
    
    
    # Track successful episodes (max reward is 500 for CartPole-v1)
    if total_reward >= 475:
        success_count += 1
    
    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        success_rate = success_count / 100
        print(f"Episode {episode+1}/{EPISODES} | "
              f"Avg Reward: {avg_reward:.1f} | "
              f"Success Rate: {success_rate:.2f} | "
              f"Epsilon: {epsilon:.4f}")
        success_count = 0  # Reset counter


plt.figure(figsize=(12,6))
plt.plot(episode_rewards, alpha=0.4, label='Episode Reward')
# Calculate 100-episode moving average
moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
plt.plot(range(99, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='100-Episode Avg')
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
#plt.savefig("qlcartpole.png")
plt.show()


# Epsilon decay
plt.figure(figsize=(12,6))
plt.plot(range(EPISODES), 
            [max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** i)) for i in range(EPISODES)],
            color='purple', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Epsilon (Exploration Rate)')
plt.title('Epsilon Decay Over Time')
plt.legend()
#plt.savefig('qlcartpoleepsiolndecay.png')
plt.show()

# Save Q-table for later use
np.save('cartpole_q_table.npy', dict(q_table))
print("Training complete. Results saved to cartpole_qlearning_results.png")

import os
from gymnasium.wrappers import RecordVideo
env=RecordVideo(env,"video-cartpoleQL",episode_trigger=lambda ep :ep % 3 ==0)
for episode in range(10):
    state, info = env.reset()
    done = False
    total_reward=0
    state=np.array(state)
    while not done:
        disc_state = discretize_state(state)
        action = np.argmax(q_table[disc_state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward+=reward
        state=next_state


    print("Episode no :",episode, "total reward", total_reward)
# 4. Close the environment
env.close()



