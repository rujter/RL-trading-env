import numpy as np
import matplotlib.pyplot as plt
import math

data = np.load("eval_results.npz")
xpoints = data["xpoints"]
ypoints = data["ypoints"]
buy_x, buy_y = data["buy_x"], data["buy_y"]
sell_x, sell_y = data["sell_x"], data["sell_y"]
hold_x, hold_y = data["hold_x"], data["hold_y"]
cborder_x, cborder_y = data["cborder_x"], data["cborder_y"]
csoffer_x, csoffer_y = data["csoffer_x"], data["csoffer_y"]
episode_arr = data["episode_arr"]
profitabilitypercentage_arr = data["profitabilitypercentage_arr"]
episode_starts = data["episode_starts"]

avg_profitability = np.mean(profitabilitypercentage_arr)
sharpe = profitabilitypercentage_arr.mean() / profitabilitypercentage_arr.std()

T = 0.0  #threshold
temp = np.minimum(0, profitabilitypercentage_arr - T)**2
downside_dev = np.sqrt(np.mean(temp))


sortino = (profitabilitypercentage_arr.mean() - T) / (downside_dev + 1e-8)


down = np.minimum(0,profitabilitypercentage_arr)
sortino = profitabilitypercentage_arr.mean() / down.std()

plt.figure(figsize=(20, 6)) 
plt.plot(xpoints, ypoints, label="price")
for es in episode_starts:
    plt.axvline(x=es, color='gray', linestyle='--', alpha=0.4)
plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label="starting price")
plt.legend()
plt.ylim(40, 60)
plt.savefig("pricemovement.png")
plt.close()

plt.figure(figsize=(20, 6))
for es in episode_starts:
    plt.axvline(x=es, color='gray', linestyle='--', alpha=0.4)

plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label="starting price")
plt.plot(buy_x, buy_y, linestyle='', marker='o', ms=2, mec='r', mfc='r', label="buy")
plt.plot(sell_x, sell_y, linestyle='', marker='o', ms=2, mec='g', mfc='g', label="sell")
plt.plot(hold_x, hold_y, linestyle='', marker='o', ms=2, mec='b', mfc='b', label="hold")
plt.plot(cborder_x, cborder_y, linestyle='', marker='x', ms=2, mec='orange', mfc='orange', label="cancel buy")
plt.plot(csoffer_x, csoffer_y, linestyle='', marker='x', ms=2, mec='purple', mfc='purple', label="cancel sell")
plt.legend()
plt.ylim(30, 70)
plt.savefig("orders.png")
plt.close()

plt.figure(figsize=(20, 6)) 
plt.text(0.02, 0.95, f"avg profitability: {avg_profitability:.2f}%", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.90, f"sharpe ratio: {sharpe:.2f}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')
plt.text(0.02, 0.85, f"sortino ratio: {sortino:.2f}", 
        transform=plt.gca().transAxes, color='green', fontsize=10, va='top')

plt.plot(episode_arr, profitabilitypercentage_arr)
plt.legend()
plt.ylim(-50, 50)
plt.savefig("profitability.png")
plt.close()