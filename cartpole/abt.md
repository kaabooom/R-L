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

![video](https://github.com/user-attachments/assets/00a6909e-e03b-4a7d-9c20-50fe63c3a77a)









  


  
    
  


   




