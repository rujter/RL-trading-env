import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


data = np.load("eval_results.npz")
ticker = str(data["ticker"])
xpoints = data["xpoints"]
ypoints = data["ypoints"]
buy_x, buy_y = data["buy_x"], data["buy_y"]
sell_x, sell_y = data["sell_x"], data["sell_y"]
hold_x, hold_y = data["hold_x"], data["hold_y"]
episode_arr = data["episode_arr"]
profitabilitypercentage_arr = data["profitabilitypercentage_arr"]
episode_starts = data["episode_starts"]

avg_profitability = np.mean(profitabilitypercentage_arr)
sharpe = profitabilitypercentage_arr.mean() / profitabilitypercentage_arr.std()

T = 0.0  #threshold
temp = np.minimum(0, profitabilitypercentage_arr - T)**2
downside_dev = np.sqrt(np.mean(temp))
sortino = (profitabilitypercentage_arr.mean() - T) / (downside_dev + 1e-8)

profbyholding = ((ypoints[-1]-ypoints[0])/ypoints[0])*100

down = np.minimum(0,profitabilitypercentage_arr)

#pricemovement
plt.figure(figsize=(20, 6)) 
plt.plot(xpoints, ypoints, label="price")
for es in episode_starts:
    plt.axvline(x=es, color='gray', linestyle='--', alpha=0.4)
plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label="starting price")
plt.text(0.02, 0.8, f"tckr: {ticker}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')

plt.legend()
plt.savefig("pricemovement.png")
plt.close()


#orders
plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label="starting price")
plt.plot(buy_x, buy_y, linestyle='', marker='o', ms=4, mec='r', mfc='r', label="buy")
plt.plot(sell_x, sell_y, linestyle='', marker='o', ms=4, mec='g', mfc='g', label="sell")
plt.plot(hold_x, hold_y, linestyle='', marker='o', ms=1, mec='b', mfc='b', label="hold")

plt.text(0.02, 0.5, f"numhold: {hold_x.size:.2f}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')

plt.text(0.02, 0.45, f"buynum: {buy_x.size:.2f}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')

plt.text(0.02, 0.40, f"numsell: {sell_x.size:.2f}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.8, f"tckr: {ticker}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')

print(hold_x)
print("holdx\n")

#plt.plot(1000,50,linestyle='', marker='o', ms=10, mec='b', mfc='b', label="hold")
plt.legend()
plt.savefig("orders.png")
plt.close()

#profitability
plt.figure(figsize=(20, 6)) 
plt.text(0.02, 0.95, f"avg profitability: {avg_profitability:.2f}%", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.90, f"sharpe ratio: {sharpe:.2f}", transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.85, f"sortino ratio: {sortino:.2f}", transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.8, f"tckr: {ticker}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.75, f"prof by hold: {profbyholding:.2f}%", transform=plt.gca().transAxes, color='green', fontsize=10, va='top')

plt.plot(episode_arr, profitabilitypercentage_arr)
plt.legend()
plt.savefig("profitability.png")
plt.close()