# About LunalLander env 🚀
![lunar_lander](https://github.com/user-attachments/assets/89206e29-eee2-43ce-8245-c67d793d1515) **before**

## Action Space
There are four discrete actions available:  
0: do nothing  
1: fire left orientation engine  
2: fire main engine  
3: fire right orientation engine  

## Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Rewards
After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.  

For each step, the reward:  

is increased/decreased the closer/further the lander is to the landing pad.  
is increased/decreased the slower/faster the lander is moving.  
is decreased the more the lander is tilted (angle not horizontal).  
is increased by 10 points for each leg that is in contact with the ground.  
is decreased by 0.03 points each frame a side engine is firing.  
is decreased by 0.3 points each frame the main engine is firing.  
The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.  

An episode is considered a solution if it scores at least 200 points.  

## Starting State  
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.  

## Episode Termination   
The episode finishes if:  
the lander crashes (the lander body gets in contact with the moon);  
the lander gets outside of the viewport (x coordinate is greater than 1);  
the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:  

---
Lets use some other algorithm apart from QL and DQN,  
lets try usign PPO ( which is proximal policy optimization )
![Screenshot 2025-06-01 220540](https://github.com/user-attachments/assets/9b7683df-1acb-4ec1-a80e-34c3baa083e0)
 
looks overwhelming right but, we dont have to write the overly complicated code for using PPO algorithm. We can use Stable-baselines3 lib to implement that without writing the whole code.   
You can choose which algo to use according to your environment.(see the table below)

![Screenshot 2025-06-01 215745](https://github.com/user-attachments/assets/4f569aae-1ebb-461d-9d50-71af2a40984b)

---
## code-
-required lib to import apart from gym

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy


-initialize the modle

    model = PPO(
        policy="MlpPolicy", env=env, n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )
-learning 

    model.learn(total_timesteps=1000000)
    # Save the model
    model_name = "ppo-LunarLander-v3"
    model.save(model_name)
    
-evaluation and loadmodel

    model=PPO.load("ppo-LunarLander-v3",env=env)
    env.reset()
    evaluate_policy(model,env,n_eval_episodes=2,render=True )
---
## Result from training
![ezgif-2da7de3344ecb4](https://github.com/user-attachments/assets/985699a9-3b5b-4a26-ae3a-f9924540b9ec) **after** 😎




    

