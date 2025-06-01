import numpy as np
import gymnasium as gym
import random
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create the 8x8 environment
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array")

state_space = env.observation_space.n
action_space = env.action_space.n

print(f"State space: {state_space} (8x8 grid)")
print(f"Action space: {action_space}")

# Optimistic initialization to encourage exploration
def initialize_q_table(state_space, action_space, optimistic_value=10.0):
    return np.full((state_space, action_space), optimistic_value)

def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state][:])

def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(Qtable, state)
    return env.action_space.sample()

# ADJUSTED HYPERPARAMETERS FOR BETTER PERFORMANCE
n_training_episodes = 100000  # More episodes for larger state space
learning_rate = 0.85  # Slightly higher learning rate
max_steps = 250  # Increased max steps for larger grid
gamma = 0.99  # Higher discount factor to value future rewards

# EXPLORATION PARAMETERS
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0001  # Slower decay for more exploration

# Evaluation parameters
n_eval_episodes = 100
eval_seed = []

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    episode_rewards = []
    epsilon_values = []
    success_rate = []  # Track success rate during training
    successes = 0  # Count successful episodes
    
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        epsilon_values.append(epsilon)
        
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_rewards = 0
        episode_success = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward

            # Update Q-value
            current_q = Qtable[state][action]
            next_max_q = np.max(Qtable[new_state])
            
            # Bellman equation
            new_q = current_q + learning_rate * (reward + gamma * next_max_q - current_q)
            Qtable[state][action] = new_q

            if terminated or truncated:
                if reward == 1.0:  # Successfully reached goal
                    successes += 1
                    episode_success = True
                break
                
            state = new_state
            
        episode_rewards.append(total_rewards)
        
        # Track success rate every 1000 episodes
        if episode % 1000 == 0:
            success_rate.append(successes / 1000)
            successes = 0
            
    return Qtable, episode_rewards, epsilon_values, success_rate

# Initialize with optimistic values to encourage exploration
Qtable_frozenlake = initialize_q_table(state_space, action_space, optimistic_value=5.0)

# Train the agent
Qtable_frozenlake, episode_rewards, epsilon_values, success_rate = train(
    n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake
)

# Calculate moving average
window_size = 500
moving_avg = [np.mean(episode_rewards[max(0, i-window_size):i+1]) 
              for i in range(len(episode_rewards))]

# Plotting
plt.figure(figsize=(15, 12))

# Reward vs Episode
plt.subplot(3, 1, 1)
plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
plt.plot(moving_avg, 'r-', linewidth=2, label=f'{window_size}-episode Avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode (8x8 FrozenLake)')
plt.legend()
plt.grid(True)

# Epsilon vs Episode
plt.subplot(3, 1, 2)
plt.plot(epsilon_values)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay vs Episode')
plt.grid(True)

# Success Rate
plt.subplot(3, 1, 3)
plt.plot(np.arange(0, n_training_episodes, 1000), success_rate, 'g-')
plt.xlabel('Episode (every 1000)')
plt.ylabel('Success Rate')
plt.title('Success Rate During Training')
plt.grid(True)

plt.tight_layout()
plt.savefig('frozenlake_8x8_training.png')
plt.show()

# Evaluation function with progress bar
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    successes = 0
    
    for episode in tqdm(range(n_eval_episodes)):
        state, _ = env.reset()
        total_rewards_ep = 0
        episode_success = False

        for step in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                if reward == 1.0:
                    successes += 1
                    episode_success = True
                break
            state = new_state
            
        episode_rewards.append(total_rewards_ep)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_percentage = (successes / n_eval_episodes) * 100
    
    return mean_reward, std_reward, success_percentage

# Evaluate
mean_reward, std_reward, success_percentage = evaluate_agent(
    env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed
)

print("\n" + "="*50)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Success rate: {success_percentage:.2f}%")
print("="*50 + "\n")

# Record a successful episode
def record_successful_episode(env, Qtable, out_directory, max_attempts=100, fps=1):
    images = []
    success = False
    attempts = 0
    
    while not success and attempts < max_attempts:
        images = []
        state, _ = env.reset()
        img = env.render()
        images.append(img)
        terminated = False
        truncated = False
        
        for step in range(max_steps):
            action = np.argmax(Qtable[state][:])
            state, reward, terminated, truncated, _ = env.step(action)
            img = env.render()
            images.append(img)
            
            if terminated:
                if reward == 1.0:
                    success = True
                break
                
        attempts += 1
    
    if success:
        imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)
        print(f"Successfully recorded solution in {attempts} attempts")
    else:
        print("Failed to record a successful episode after 100 attempts")

# Record a video of a successful episode
record_successful_episode(env, Qtable_frozenlake, "frozenlake_8x8_solution.gif")
