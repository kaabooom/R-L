import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.bool8 = bool 
import random
import matplotlib.pyplot as plt
from collections import deque

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.Linear1 = nn.Linear(input_size, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        return self.Linear3(x)

# Hyperparameters
env_name = 'CartPole-v1'
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000
batch_size = 32
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update_frequency = 100
num_episodes = 500

# Initialize environment and DQN
env = gym.make(env_name)
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Experience replay buffer
replay_buffer = deque(maxlen=buffer_size)
epsilon = epsilon_start

# Training statistics
episode_rewards = []
moving_averages = []
epsilon_history=[]

def train():
    global epsilon
    step_count = 0
    
    for episode in range(num_episodes):
        state,info = env.reset()
        state = np.array(state)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()

            # Take action and observe reward and next state
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # Sample a batch from the replay buffer
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors and move to device
                states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                # Compute Q-values and target Q-values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q_values = target_net(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and update policy network
                loss = criterion(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_count += 1

                # Update target network
                if step_count % target_update_frequency == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done or truncated:
                break

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        epsilon_history.append(epsilon)

        # Record training progress
        episode_rewards.append(total_reward)
        moving_avg = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
        moving_averages.append(moving_avg)
        
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, Moving Avg: {moving_avg:.2f}")

    # Plot reward to episode
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(moving_averages, label='100-episode Moving Avg', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Performance on CartPole-v1')
    plt.legend()
    plt.grid()
    plt.savefig('dqn_training_cartPoleDQN.png')
    plt.show()

    # Plot epsilon decay during training
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_episodes), 
            [max(epsilon_min, epsilon_start * (epsilon_decay ** i)) for i in range(num_episodes)],
            color='purple', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('Epsilon Decay Over Time')
    plt.grid()
    plt.savefig('epsilon_decay_cartpoleDQN.png')
    plt.show()

    # Save model
    torch.save(policy_net.state_dict(), 'cartpole_dqn.pth')

'''if __name__ == "__main__":
    train()'''


env.close()


env = gym.make(env_name, render_mode='human')
model = DQN(input_size, output_size).to(device)
model.load_state_dict(torch.load("cartpole_dqn.pth", map_location=device))
model.eval()

# Debug: Print model architecture and device
print("Model Architecture:")
print(model)
print(f"Model Device: {next(model.parameters()).device}")

episodes = 3
for episode in range(1, episodes + 1):
    state, info = env.reset()
    state = np.array(state)  # Ensure state is NumPy array
    print("episode number :" ,episode)
    total_reward = 0

    for i in range (0,400):

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model(state_tensor).argmax().item()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            total_reward += reward
            state = np.array(observation)
            
    
    print(f"Episode {episode} finished with reward {total_reward:.2f}")

env.close()
