import sys
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from env_cancel import env_cancel, SEED
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 1. Instantiate & check Gymnasium compliance
    env = env_cancel(id=1, startingcash=1000.0)
    check_env(env)

    # 2. Train SAC agent (SAC uses built-in ent_coef entropy exploration instead of epsilon-decay)
    agent = SAC("MlpPolicy", env, seed=SEED, verbose=1, ent_coef="auto", learning_rate=0.0005)
    agent.learn(total_timesteps=100000, log_interval=1)
    

    # 3. Evaluation step — run multiple episodes so the plot spans ~10k steps
    num_eval_episodes = 100  # 100 episodes * 1000 steps/episode = 10,0000 steps total

    xpoints = np.array([])
    ypoints = np.array([])
    buy_x, buy_y = np.array([]), np.array([])
    sell_x, sell_y = np.array([]), np.array([])
    hold_x, hold_y = np.array([]), np.array([])
    cborder_x, cborder_y = np.array([]), np.array([])
    csoffer_x, csoffer_y = np.array([]), np.array([])

    episode_arr = np.array([])
    profitabilitypercentage_arr = np.array([])
    episodenum = 0

    global_step = 0

    for episode in range(num_eval_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            global_step += 1
            ep_steps += 1

            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            price = env.last_trade_price

            xpoints = np.append(xpoints, global_step)
            ypoints = np.append(ypoints, price)

            if info.get("ordernum") == 1:
                buy_x = np.append(buy_x, global_step)
                buy_y = np.append(buy_y, price)
            elif info.get("ordernum") == 2:
                sell_x = np.append(sell_x, global_step)
                sell_y = np.append(sell_y, price)
            elif info.get("ordernum") == 3:
                cborder_x = np.append(cborder_x, global_step)
                cborder_y = np.append(cborder_y, price)
            elif info.get("ordernum") == 4:
                csoffer_x = np.append(csoffer_x, global_step)
                csoffer_y = np.append(csoffer_y, price)
            else:
                hold_x = np.append(hold_x, global_step)
                hold_y = np.append(hold_y, price)

            if terminated or truncated:

                episodenum+= 1
                portfolio_val = info.get('val', 0.0)
                profitability_pct = (portfolio_val - env.startingcash) / env.startingcash * 100

                episode_arr = np.append(episode_arr, episodenum)
                profitabilitypercentage_arr = np.append(profitabilitypercentage_arr, profitability_pct)

                cash = env.portfolio.cash
                shares = env.portfolio.holdings.get(env.stock.id, 0)
                print(
                    f"Episode {episode:<3} | steps: {ep_steps:<5} | "
                    f"final price: ${price:.2f} | cash: ${cash:.2f} | shares: {shares:.0f} | "
                    f"ep_reward: {ep_reward:.2f} | portfolioVal: ${info.get('val', 0.0):.2f}"
                )
                break

    env.close()

    plt.figure(figsize=(20, 6)) 
    plt.plot(xpoints, ypoints, label="price")
    plt.legend()
    plt.ylim(40, 60)
    plt.savefig("pricemovement.png")
    plt.close()

    plt.figure(figsize=(20, 6))
    plt.plot(buy_x, buy_y, linestyle='', marker='o', ms=2, mec='r', mfc='r', label="buy")
    plt.plot(sell_x, sell_y, linestyle='', marker='o', ms=2, mec='g', mfc='g', label="sell")
    plt.plot(hold_x, hold_y, linestyle='', marker='o', ms=2, mec='b', mfc='b', label="hold")
    plt.plot(cborder_x, cborder_y, linestyle='', marker='x', ms=2, mec='orange', mfc='orange', label="cancel buy")
    plt.plot(csoffer_x, csoffer_y, linestyle='', marker='x', ms=2, mec='purple', mfc='purple', label="cancel sell")
    plt.legend()
    plt.ylim(40, 60)
    plt.savefig("orders.png")
    plt.close()

    plt.figure(figsize=(20, 6)) 
    plt.plot(episode_arr, profitabilitypercentage_arr)
    plt.xlabel("Episodes")
    plt.ylabel("Profitability")
    plt.legend()
    plt.ylim(40, 60)
    plt.savefig("profitability.png")
    plt.close()

if __name__ == "__main__":
    main()