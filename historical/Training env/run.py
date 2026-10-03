import sys
import os

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env

#depends on what enviroment we're importing from, when trainign or testing uncomment 
from training import Environment, SEED, CustomCNN, CustomRNN, policy_kwargs_cnn, policy_kwargs_rnn, policy_kwargs_def

#from testing import Environment, SEED, CustomCNN, CustomRNN, policy_kwargs_cnn, policy_kwargs_rnn, policy_kwargs_def
import matplotlib.pyplot as plt
import numpy as np
from training import tort


def main():
    # 1. Instantiate & check Gymnasium compliance
    #you put the import after the thing
    env = Environment(id=1, startingcash=10000.0)
    check_env(env)

    #policy_kwargs_default = policy_kwargs_def
    #policy_kwargs_default = policy_kwargs_rnn
    policy_kwargs_default = policy_kwargs_cnn
    if tort:
        agent = PPO("MlpPolicy", env, seed=SEED, verbose=1, ent_coef=0, policy_kwargs=policy_kwargs_default, learning_rate=0.0003)
        agent.learn(total_timesteps=1000000, log_interval=1) #to doloci kolko casa traja oz how many steps
        agent.save("cnn1mppo")
    else:
        agent = PPO.load("cnn1mppo", env=env)



    #agent = PPO("MlpPolicy", env, seed=SEED, verbose=1, ent_coef=0, policy_kwargs=policy_kwargs_default, learning_rate=0.0003)
    #agent = PPO.load("cnn1mppo", env=env)

    #agent = SAC("MlpPolicy", env, seed=SEED, verbose=1, ent_coef="auto", learning_rate=0.0005)
    #agent = SAC.load("META YTD agent 200k", env=env)

    #agent.learn(total_timesteps=1000000, log_interval=1) #to doloci kolko casa traja oz how many steps
    #najprej learna

    #agent.save("cnn1mppo")

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
    episode_starts = []

    episodenum = 1
    
    for i in range(episodenum): #to je potem evaluation
        episode_starts.append(global_step + 1)
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            global_step += 1
            ep_steps += 1

            action, _states = agent.predict(obs, deterministic=True)

            if ep_steps < 10:
                print(f"STEP {ep_steps} | action = {action}")

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            price = info.get("price")

            xpoints = np.append(xpoints, global_step)
            ypoints = np.append(ypoints, price)

            if info.get("ordernum") == 1:
                buy_x = np.append(buy_x, global_step)
                buy_y = np.append(buy_y, price)
            elif info.get("ordernum") == 2:
                sell_x = np.append(sell_x, global_step)
                sell_y = np.append(sell_y, price)
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
                    f"Episode {episodenum:<3} | steps: {ep_steps:<5} | "
                    f"final price: ${price:.2f} | cash: ${cash:.2f} | shares: {shares:.0f} | "
                    f"ep_reward: {ep_reward:.2f} | portfolioVal: ${info.get('val', 0.0):.2f}"
                )
                
                break

    env.close()
    
    np.savez("eval_results.npz",
            xpoints=xpoints, ypoints=ypoints,
            buy_x=buy_x, buy_y=buy_y,
            sell_x=sell_x, sell_y=sell_y,
            hold_x=hold_x, hold_y=hold_y,
            cborder_x=cborder_x, cborder_y=cborder_y,
            csoffer_x=csoffer_x, csoffer_y=csoffer_y,
            episode_arr=episode_arr, profitabilitypercentage_arr=profitabilitypercentage_arr,
            episode_starts=np.array(episode_starts))
    
if __name__ == "__main__":
    main()