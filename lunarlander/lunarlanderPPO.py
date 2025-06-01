import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="human" )



'''episodes = 3
for episode in range(1, episodes + 1):
    observation, info = env.reset()
    done = False
    total_reward = 0

    print(f"Episode {episode} started.")
    
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward


    print(f"Episode {episode} finished with reward {total_reward:.2f}")

env.close()'''


# Create the environment
'''env = make_vec_env("LunarLander-v3", n_envs=4)'''

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)
'''
model.learn(total_timesteps=1000000)
# Save the model
model_name = "ppo-LunarLander-v3"
model.save(model_name)

'''
model=PPO.load("ppo-LunarLander-v3",env=env)
env.reset()
evaluate_policy(model,env,n_eval_episodes=2,render=True )
env.close()
