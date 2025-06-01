## 🧠 Environment Overview

CartPole is a classic control problem provided by [OpenAI Gym]([https://gym.openai.com/](https://gymnasium.farama.org/environments/classic_control/cart_pole/)). The goal is to balance a pole on a moving cart by applying forces to the cart.

- **Observation Space**: 4 continuous variables
  - Cart Position
  - Cart Velocity
  - Pole Angle
  - Pole Velocity at Tip
- **Action Space**: Discrete (2)
  - `0`: Push cart to the left
  - `1`: Push cart to the right
- **Reward**: +1 for every timestep the pole remains upright
- **Episode Termination**:
  - Pole angle > ±12°
  - Cart position > ±2.4 units
  - Episode length ≥ 500 (solved if average reward ≥ 475 over 100 episodes)

---
### We will solve this environment by~ Q-learning and Deep Q-network approach 
- # Q-L 
In Q-L we need a Q Table where states are in rows and actions are in coloumns, **Q-Learning works only with discrete states** , so we discretize the observation space.
  we discrete that using bins to map continuous observations into discrete buckets.   
For this env i used  ~ [1,3,6,3] values as bin  
1-Cart postion (least imp that why we give it one 1 bin (it does not matter much about the position for balancing the rod)    
3-Cart velocity (3 value low,med,high)      
6-Pole angle (most imp we need finely divided values)    
3-Pole angular velocity  (3 value low,med,high)      
  
for this discretization we make a fuction which take observation and bins as parameter and return a state tuple. Our Q-table will have 54 rows (1*3*6*3) and 2 coloumns.  
Then define Hyperparameter for learning     
-alpha = 0.1         # learning rate  
-gamma = 0.99         # discount factor  
-epsilon = 1.0        # initial exploration rate  
-epsilon_decay = 0.995  
-min_epsilon = 0.01  

Initialize Q-function Q(s, a) = 0 (ie~  put all values to 0 )  
pseudo code for training

    for t = 1, 2, ..., M do
        Choose a starting location
    
        while (not a goal state) do
            Check next possible N states
            
            Choose action according to epsilon-greedy policy:
                with probability ε:    a_t ← random action a ∈ A  (exploration)
                with probability 1-ε:  a_t ← argmaxₐ Q(s, a)     (exploitation)
            
            Perform the chosen action  
            Transition to the next state  
            Receive the immediate reward  
            Update the Q-table:
                Q(s, a) ← Q(s, a) + α [r + γ * maxₐ Q(s', a) − Q(s, a)]
    
        end while

    Update the exploration rate using a decaying function
    end for

The optimal policy is 🤧:
    π*(s) = argmaxₐ Q(s, a), ∀ s ∈ S  

## 📈Result from my training ~
![Screenshot 2025-05-31 223258](https://github.com/user-attachments/assets/7c47df20-7be6-473f-a15e-f15cb98fbd64)  ![Screenshot 2025-05-31 223307](https://github.com/user-attachments/assets/4c822b1d-da32-4be7-ade2-2ac853669628)

Optimal training EP is 1500 to 2000 ( i did it for 3000 😥)  
![ezgif-34e824e62639e7](https://github.com/user-attachments/assets/d8e6c701-4f5b-485b-aab1-fdde9f1707cd)
---
- # DQN
The Deep Q-Network (DQN) algorithm uses a neural network to approximate Q-values for the CartPole environment, allowing it to handle the continuous state space. It learns optimal actions by combining experience replay and a target network to stabilize training.  

step 1 :Is to define Q-network (using pytorch)  
step 2 :Initialize replay memory D to capacity N    
step 3 :Initialize action-value function Q with random weights θ    (Q network use for predicting)  
step 4 :Initialize target action-value function Q̂ with weights θ⁻ = θ      (target network for stablelizing)  
step 5 :training loop (each ep)

    For episode = (1 to M) do   #M is total no of episodes
        Initialize sequence s₁ = {x₁} and preprocessed sequence ϕ₁ = ϕ(s₁)   #x1 is the first obs of the ep
    
        For t = (1 to T) do     #T is max time steps per episodes
            With probability ε select a random action aₜ  
            Otherwise select aₜ = argmaxₐ Q(ϕ(sₜ), a; θ)  with probability 1-ε
    
            Execute action aₜ in emulator and observe reward rₜ and image xₜ₊₁  
            Set sₜ₊₁ = sₜ, aₜ, xₜ₊₁ and preprocess ϕₜ₊₁ = ϕ(sₜ₊₁)  
    
            Store transition (ϕₜ, aₜ, rₜ, ϕₜ₊₁) in D  
            Sample random minibatch of transitions (ϕⱼ, aⱼ, rⱼ, ϕⱼ₊₁) from D  
    
            Set target yⱼ =  
                rⱼ if episode terminates at step j+1  
                rⱼ + γ * maxₐ' Q̂(ϕⱼ₊₁, a'; θ⁻) otherwise  
    
            Perform a gradient descent step on (yⱼ - Q(ϕⱼ, aⱼ; θ))² w.r.t θ  
    
            Every C steps, reset Q̂ = Q  
        End For
    End For
#### More easy way of explanation ~
1. Create a memory (called replay buffer) to store experiences.
2. Build two neural networks:
   - Q Network (for predicting actions)
   - Target Q Network (used for stable learning)
   - Both are initially the same.

3. For each episode:
   a. Start from the initial state.
   
   b. For each step in the episode:
      i.    Choose an action:
            - With probability ε, choose a random action (explore).
            - Otherwise, pick the best action from the Q network (exploit).

      ii.   Do the action in the environment, get:
            - New state
            - Reward
            - Whether episode is done

      iii.  Save this experience (state, action, reward, next state) to memory.

      iv.   Randomly pick a batch of past experiences from memory.

      v.    For each experience in the batch:
            - Calculate the target Q value:
              → If it's the last step: target = reward
              → Else: target = reward + γ * max Q value from target network

      vi.   Update the Q network using gradient descent to reduce the difference between:
            - Predicted Q value and the target Q value

      vii.  Every few steps, update the target network to match the Q network.

4. Repeat for many episodes until the agent learns.
---
✅ Experience Replay: Transitions are stored in memory D and sampled randomly to break correlation.  
✅ Fixed Q-Target: The target network Q̂ (with weights θ⁻) is updated every C steps, not every step, to stabilize learning.

## 📈Result from my training ~
![dqn_training_cartPoleDQN](https://github.com/user-attachments/assets/3731ce19-59d3-4d06-9393-802491c46c1b)  
![epsilon_decay_cartpoleDQN](https://github.com/user-attachments/assets/7d292559-351c-44b4-bdee-887e26143ba5)  
![ezgif-240a8c09fab965](https://github.com/user-attachments/assets/859919fa-9a61-4aaf-832a-fe0b390db2fd)
PS - More stable than Qlearning one 😎














  


  
    
  


   




