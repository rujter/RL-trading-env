import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from training import Environment, tort
from Classes import CustomCNN, CustomRNN, policy_kwargs_cnn, policy_kwargs_rnn, policy_kwargs_def
from pathlib import Path
from plotonly import plot_results
import random


N_EVAL_EPISODES = 14
TOTAL_TIMESTEPS = 100000
START_SEED = 30
START_LAMBDA = 0.2

foldermodel = Path("models")
foldermodel.mkdir(exist_ok=True)

def evaluation(agent, env, name):
    xpoints, ypoints = [], []
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    hold_x, hold_y = [], []
    episode_arr, profitability_arr = [], []
    realx, realy = [], []
    episode_starts = []
    episode_tickers = []

    global_step = 0

    for episode in range(1, N_EVAL_EPISODES + 1):
        episode_starts.append(global_step + 1)
        obs, info = env.reset()
        episode_tickers.append(env.ticker)
        ep_reward = 0.0
        ep_steps = 0
        first_price = None

        while True:
            global_step += 1
            ep_steps += 1

            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            price = info.get("price")
            if first_price is None:
                first_price = price

            xpoints.append(global_step)
            ypoints.append(price)

            order = info.get("ordernum")
            if order == 1:
                buy_x.append(global_step); buy_y.append(price)
            elif order == 2:
                sell_x.append(global_step); sell_y.append(price)
            else:
                hold_x.append(global_step); hold_y.append(price)

            if terminated or truncated:
                portfolio_val = info.get("val", 0.0)
                profit_pct = (portfolio_val - env.startingcash) / env.startingcash * 100

                episode_arr.append(episode)
                profitability_arr.append(profit_pct)
                realx.append(episode)
                realy.append(price - first_price)

                cash = env.portfolio.cash
                shares = env.portfolio.holdings.get(env.stock.id, 0)
                print(
                    f"Episode {episode:<3} | steps: {ep_steps:<5} | "
                    f"final price: ${price:.2f} | cash: ${cash:.2f} | shares: {shares:.0f} | "
                    f"ep_reward: {ep_reward:.2f} | portfolioVal: ${portfolio_val:.2f} | "
                    f"profit: {profit_pct:.2f}%"
                )
                break

    results_path = f"eval_results_{name}.npz"
    np.savez(
        results_path,
        xpoints=np.array(xpoints), ypoints=np.array(ypoints),
        buy_x=np.array(buy_x), buy_y=np.array(buy_y),
        sell_x=np.array(sell_x), sell_y=np.array(sell_y),
        hold_x=np.array(hold_x), hold_y=np.array(hold_y),
        episode_arr=np.array(episode_arr),
        profitabilitypercentage_arr=np.array(profitability_arr),
        realx=np.array(realx), realy=np.array(realy),
        episode_starts=np.array(episode_starts),
        tickers=np.array(episode_tickers),
    )

    plot_results(seed=name, results_path=results_path)


def main():
    env = Environment(id=1, startingcash=10000.0)
    check_env(env)
    env.test_ctr = 0

    policy_kwargs = policy_kwargs_cnn

    seed = START_SEED
    for i in range(3):
        lambdaval = START_LAMBDA
        for j in range(3):
            MODEL_NAME = f"ppo100kcnn_s{seed}_l{lambdaval:.1f}"

            env.lam = lambdaval
            random.seed(seed)

            if tort:
                agent = PPO("MlpPolicy", env, seed=seed, verbose=1, ent_coef=0.005,
                            policy_kwargs=policy_kwargs, learning_rate=0.0003)
                agent.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1)
                agent.save(foldermodel / MODEL_NAME)
            else:
                print("evaluating", MODEL_NAME)
                env.test_ctr = 0
                agent = PPO.load(foldermodel / MODEL_NAME, env=env)
                evaluation(agent, env, MODEL_NAME)

            lambdaval = round(lambdaval + 0.1, 1)
        seed += 1

    env.close()


if __name__ == "__main__":
    main()