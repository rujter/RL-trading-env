import pathlib
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from training import Environment, tort           
from Classes import policy_kwargs_def             
import config as C

MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR = pathlib.Path(C.EVAL_DIR)
EVAL_DIR.mkdir(exist_ok=True)


def make_env(a=0.0, b=0.0, lam=0.0):
    env = Environment(id=1, startingcash=C.STARTING_CASH)
    env.a, env.b, env.lam = a, b, lam
    env.test_ctr = 0 
    return env


def evaluate(agent, env, name, deterministic=True):
    prices, equity, shares, orders, ep_ids, tickers = [], [], [], [], [], []

    for ep in range(C.N_EVAL_EPISODES):
        obs, info = env.reset()
        tickers.append(str(env.ticker))
        ep_steps = 0
        while True:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            price, val = info.get("price"), info.get("val")

            prices.append(float(price))
            equity.append(float(val))
            shares.append(float(env.portfolio.holdings.get(env.stock.id, 0)))
            orders.append(int(info.get("ordernum") or 0))
            ep_ids.append(ep)
            ep_steps += 1

            if terminated or truncated:
                ret = (val - env.startingcash) / env.startingcash * 100
                print(f"  ep {ep:<2} {tickers[-1]:<8} steps={ep_steps:<5} "
                      f"final_val=${val:,.0f} return={ret:6.1f}%")
                break

    np.savez(
        EVAL_DIR / f"eval_results_{name}.npz",
        price=np.array(prices), equity=np.array(equity), shares=np.array(shares),
        order=np.array(orders), ep_id=np.array(ep_ids),
        tickers=np.array(tickers, dtype=str),
        startingcash=env.startingcash, deterministic=deterministic,
    )


def main():

    for seed in C.SEEDS:
        for a, b in C.TARGET_CONFIGS:
            name = f"{C.MODEL_PREFIX}_s{seed}_a{a:.2f}_b{b:.2f}"
            model_file = MODELS_DIR / f"{name}.zip"
            env = make_env(a, b, C.FIXED_LAMBDA)
            set_random_seed(seed)  # python + numpy + torch

            if tort:
                print(f"Training {name}")
                agent = PPO("MlpPolicy", env, seed=seed, verbose=1, ent_coef=0.005,
                            policy_kwargs=policy_kwargs_def, learning_rate=3e-4, device="cpu")
                checkpoint_callback = None
                if C.CHECKPOINT_FREQ > 0:
                    checkpoint_callback = CheckpointCallback(
                        save_freq=max(C.CHECKPOINT_FREQ // agent.n_envs, 1),
                        save_path=str(MODELS_DIR / "checkpoints" / name),
                        name_prefix=name,
                        verbose=2,
                    )
                agent.learn(total_timesteps=C.TOTAL_TIMESTEPS, log_interval=1,
                            callback=checkpoint_callback)
                agent.save(model_file)
            else:
                if not model_file.exists():
                    env.close()
                    continue
                print(f"Evaluating {name}")
                agent = PPO.load(model_file, env=env)
                evaluate(agent, env, name, deterministic=C.EVAL_DETERMINISTIC)
            env.close()

if __name__ == "__main__":
    main()
