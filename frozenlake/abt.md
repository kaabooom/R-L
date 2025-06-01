## About FROZENLAKE env 🥶
---
![frozen_lake](https://github.com/user-attachments/assets/5528cce6-0666-4711-83c0-78ff21e5c615)
### Action Space 🕹️
The action shape is (1,) in the range {0, 3} indicating which direction to move the player.  
0: Move left  
1: Move down  
2: Move right  
3: Move up  
**Action Space -> Discrete(4)**

### Observation Space
The observation is a value representing the player’s current position as current_row * ncols + current_col (where both the row and col start at 0).  
For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map.  
The observation is returned as an int().  
**Observation Space -> Discrete(16)**   ( for 4x4, for 8x8 it is Discrete(64). )

### Starting State  
The episode starts with the player in state [0] (location [0, 0]).  

### Rewards 
Reward schedule:    
Reach goal: +1  
Reach hole: 0  
Reach frozen: 0  

### Episode End  
The episode ends if the following happens:  
-Termination:
1. The player moves into a hole. 😵
2. The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).  

-Truncation:
(when using the time_limit wrapper):
The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

**is_slippery=True** : If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.  

For example, if action is left and is_slippery is True, then:  
P(move left)=1/3  
P(move up)=1/3  
P(move down)=1/3  


---
we are going to solve this env using Q-L approach   
APPORACH-  
1. Initialize:  
   - Q-table: 64 states (8x8 grid) × 4 actions (LEFT, DOWN, RIGHT, UP) → Initialize to zeros  
   - Hyperparameters:  
        α (learning rate) = 0.1       # Step size for updates  
        γ (discount factor) = 0.99     # Future reward importance  
        ε (exploration rate) = 1.0     # Start with 100% exploration  
        ε_decay = 0.999                # Decay per episode  
        ε_min = 0.01                   # Minimum exploration rate  
        episodes = 20000               # Training iterations  

2. For each episode:  
   - Reset environment: Agent starts at (0,0).  
   - While episode not terminated (goal or hole not reached):  
        a. Choose action:  
             - With probability ε: Random action (explore)  
             - Else: Action with max Q-value for current state (exploit)  
        
        b. Take action, observe:  
             - next_state, reward, done (terminated?), info  
        
        c. Update Q-table:  
             Q[state, action] = Q[state, action] + α * [reward + γ * max(Q[next_state]) - Q[state, action]]  
        
        d. Set state = next_state  
        
        e. If done:  
             - If goal reached: reward = +1    
             - If hole: reward = 0 (default)  
             - Break loop  

   - Decay exploration: ε = max(ε_min, ε * ε_decay)  

3. After training:  
   - Run evaluation episodes with ε=0 (pure exploitation) to test success rate.  
---
## Result from training 
![frozenlake_8x8_training](https://github.com/user-attachments/assets/c2ec961e-5d17-4921-99fd-188735c60696)
![frozenlake_8x8_solution](https://github.com/user-attachments/assets/a3c404df-e1d6-495d-9072-6b9a6c9dc084)
![frozenlake](https://github.com/user-attachments/assets/fc2e6c0d-27a5-4e1c-bef2-379f74ed673b)
