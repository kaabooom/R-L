## 🧠 Environment Overview

CartPole is a classic control problem provided by [OpenAI Gym](https://gym.openai.com/). The goal is to balance a pole on a moving cart by applying forces to the cart.

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



