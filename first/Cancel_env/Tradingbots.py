import random
from Classes import Order, nextId, next_order_id
startingprice = 50

#racunas mean ceno, ce gre vec kot to kupis dol, ce ne gor
def mean_reversion_trader(env, portfolio_id, mean, count):
    currprice = env.last_trade_price
    count += 1
    mean = mean + (currprice - mean) / count

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
    return mean, count


def momentum_trader_step(env, portfolio_id):
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