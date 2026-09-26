import random
from Classes import Order, nextId, next_order_id
startingprice = 50
import math

#racunas mean ceno, ce gre vec kot to kupis dol, ce ne gor
def mean_reversion_trader(env, portfolio_id, mean, count):
    currprice = env.last_trade_price
    count += 1

    deviation = currprice - mean
    roll = random.uniform(-1, 1) - deviation * 0.3  # actually use mean here
    if roll < -0.1:
        action = "sell"
    elif roll > 0.1:
        action = "buy"
    else:
        action = "hold"

    qty = random.randint(1, 20)
    price_ref = env.last_trade_price if env.last_trade_price > 0 else startingprice
    price_offset = random.uniform(-0.5, 0.5)
    price = price_ref + price_offset
    max_valid_price = (env.array_size - 1) / 100 
    price = max(0.01, min(price, max_valid_price))

    o = Order(next_order_id(action == "buy"), price, qty,
              marketorlimit=1, stock_id=env.stock.id, portfolio_id=portfolio_id)

    env.LimitGOrder(o, env.stock)
    return count

def Ornstein_trader_step(env, portfolio_id, mean, count, alpha = 0.05, theta = 0.5, dt = 1, sigma = 1):

    count+= 1
    currprice = env.last_trade_price

    drift = theta*(mean - currprice) * dt #dt je tick, pac 1 ker en step
    Z = random.gauss(0,1)
    noise = sigma * (dt ** 0.5) * Z ##to je volatility dW
    dS = drift + noise

    if dS < -0.5: #act when the signal is large enough
        action = "sell"
    elif dS > 0.5:
        action = "buy"
    else:
        action = "hold"

    qty = random.randint(1, 20)
    price_ref = env.last_trade_price if env.last_trade_price > 0 else startingprice
    price_offset = random.uniform(-0.5, 0.5)
    price = price_ref + price_offset
    max_valid_price = (env.array_size - 1) / 100
    price = max(0.01, min(price, max_valid_price))

    o = Order(next_order_id(action == "buy"), price, qty,
              marketorlimit=1, stock_id=env.stock.id, portfolio_id=portfolio_id)

    env.LimitGOrder(o, env.stock)
    return count

def momentum_trader_step(env, portfolio_id):

    currprice = env.last_trade_price
    
    momentum_bias = 0.1 if env.last_trade_price > env.price_before_step else -0.1
    roll = random.uniform(-1, 1) + momentum_bias
    if roll < -0.1:
        action = "sell"
    elif roll > 0.1:
        action = "buy"
    else:
        action = "hold"


    qty = random.randint(1, 20)
    price_ref = env.last_trade_price if env.last_trade_price > 0 else startingprice
    price_offset = random.uniform(-0.5, 0.5)
    price = price_ref + price_offset
    max_valid_price = (env.array_size - 1) / 100 
    price = max(0.01, min(price, max_valid_price))

    o = Order(next_order_id(action == "buy"), price, qty,
              marketorlimit=1, stock_id=env.stock.id, portfolio_id=portfolio_id)

    env.LimitGOrder(o, env.stock)

def noise_trader_step(env, portfolio_id):

    action = random.choice(["buy", "sell", "hold"])
    if action == "hold":
        return

    qty = random.randint(1, 20)
    price_ref = env.last_trade_price if env.last_trade_price > 0 else startingprice
    price_offset = random.uniform(-0.5, 0.5)
    price = price_ref + price_offset
    max_valid_price = (env.array_size - 1) / 100 
    price = max(0.01, min(price, max_valid_price))

    o = Order(next_order_id(action == "buy"), price, qty,
              marketorlimit=1, stock_id=env.stock.id, portfolio_id=portfolio_id)

    env.LimitGOrder(o, env.stock)

def GBM_trader_step(env, portfolio_id, mu = 0.05, sigma = 0.2, dt = 1):

    currprice = env.last_trade_price

    Z = random.gauss(0, 1)
    predicted_next_price = currprice * math.exp((mu - 0.5*sigma**2)*dt + sigma*math.sqrt(dt)*Z)
    predicted_change = predicted_next_price - currprice

    threshold = 0.5  

    if predicted_change < -threshold:
        action = "sell"
    elif predicted_change > threshold:
        action = "buy"
    else:
        action = "hold"


    qty = min(20, max(1, int(abs(predicted_change)))) 

    price_ref = env.last_trade_price if env.last_trade_price > 0 else startingprice
    price_offset = random.uniform(-0.5, 0.5)
    price = price_ref + price_offset
    max_valid_price = (env.array_size - 1) / 100 
    price = max(0.01, min(price, max_valid_price))

    o = Order(next_order_id(action == "buy"), price, qty,
              marketorlimit=1, stock_id=env.stock.id, portfolio_id=portfolio_id)

    env.LimitGOrder(o, env.stock)

def poisson(lam, dt, n):
    return ((lam * dt) ** n * math.exp(-lam * dt)) / math.factorial(n)

def GBMandMerton_trader_step(env, portfolio_id, mu = 0.05, sigma = 0.2, dt = 1):

    currprice = env.last_trade_price

    Z = random.gauss(0, 1)

    lam = 0.005 #expected num of jumps per 
    sigma_j = 0.02
    mu_j = -0.05 

    k = math.exp(mu_j + 0.5 * (sigma_j**2)) - 1
    Yi = random.gauss(mu, sigma**2)

    u = random.random()
    Nt = 0
    cumulat = poisson(lam, dt, 0)

    #how many jumps in ts step
    while u > cumulat:
        Nt += 1
        cumulat += poisson(lam, dt, Nt)

    mertonsum = 0
    if Nt > 0:mertonsum = random.gauss(Nt * mu_j, math.sqrt(Nt) * sigma_j)

    predicted_next_price = currprice * math.exp(((mu - lam * k) - 0.5 * (sigma**2)) * dt + sigma * math.sqrt(dt) * Z + mertonsum)
    predicted_change = predicted_next_price - currprice

    threshold = 0.5  # vsak episode 1000 steps torej 5 na episode

    if predicted_change < -threshold:
        action = "sell"
    elif predicted_change > threshold:
        action = "buy"
    else:
        action = "hold"


    qty = min(20, max(1, int(abs(predicted_change)))) 

    price_ref = env.last_trade_price if env.last_trade_price > 0 else startingprice
    price_offset = random.uniform(-0.5, 0.5)
    price = price_ref + price_offset
    max_valid_price = (env.array_size - 1) / 100 
    price = max(0.01, min(price, max_valid_price))

    o = Order(next_order_id(action == "buy"), price, qty,
              marketorlimit=1, stock_id=env.stock.id, portfolio_id=portfolio_id)

    env.LimitGOrder(o, env.stock)