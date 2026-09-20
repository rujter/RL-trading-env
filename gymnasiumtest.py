import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
import math

class TradingEnv(gym.Env):
    """Trading Environment with trend patterns and action masking."""
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, starting_cash=1000.0, trade_qty=10, max_steps=200):
        super().__init__()
        self.starting_cash = starting_cash
        self.trade_qty = trade_qty
        self.max_steps = max_steps
        
        # Actions: 0 = Buy, 1 = Sell, 2 = Hold
        self.action_space = spaces.Discrete(3)
        
        # State: [normalized_price, price_change, cash_ratio, position_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def _get_price(self, step):
        """Generates a predictable sine wave price trend with minor noise."""
        base_price = 100 + 20 * math.sin(step / 10.0)
        noise = np.random.normal(0, 0.5)
        return float(base_price + noise)

    def _get_obs(self):
        price_change = (self.price - self.last_price) / self.starting_price
        cash_ratio = self.cash / self.starting_cash
        position_value = (self.qty * self.price) / self.starting_cash
        norm_price = self.price / self.starting_price
        
        return np.array([norm_price, price_change, cash_ratio, position_value], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.starting_cash
        self.qty = 0
        
        self.starting_price = self._get_price(0)
        self.price = self.starting_price
        self.last_price = self.starting_price
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        value_before = self.cash + (self.qty * self.price)

        # Execute actions with validation
        if action == 0:  # BUY
            cost = self.trade_qty * self.price
            if self.cash >= cost:
                self.cash -= cost
                self.qty += self.trade_qty
        elif action == 1:  # SELL
            if self.qty >= self.trade_qty:
                self.cash += self.trade_qty * self.price
                self.qty -= self.trade_qty

        # Advance price to next step
        self.last_price = self.price
        self.price = self._get_price(self.current_step)

        value_after = self.cash + (self.qty * self.price)
        
        # Reward is percentage change in portfolio value
        reward = float((value_after - value_before) / self.starting_cash)

        terminated = bool(value_after < 0.10 * self.starting_cash)
        truncated = bool(self.current_step >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, {}


# 1. Verify environment
env = TradingEnv()
check_env(env)

# 2. Train with higher entropy coefficient (ent_coef=0.05) to force exploration
model = A2C("MlpPolicy", env, verbose=1, ent_coef=0.05, learning_rate=0.001)
model.learn(total_timesteps=30_000)

# 3. Test policy
obs= env.reset()
action_names = {0: "BUY", 1: "SELL", 2: "HOLD"}

print("\n--- EVALUATION RUN ---")
for step in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated = env.step(action)
    
    price = obs[0] * env.starting_price
    print(f"Step {step+1:02d} | Action: {action_names[int(action)]:<4} | Price: ${price:.2f} | Cash: ${env.cash:.2f} | Shares: {env.qty}")
    
    if terminated or truncated:
        break