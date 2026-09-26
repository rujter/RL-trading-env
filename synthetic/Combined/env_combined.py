import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
from Tradingbots import momentum_trader_step, mean_reversion_trader, noise_trader_step, GBM_trader_step, Ornstein_trader_step
import math

# Import shared models from Classes
from Classes import Portfolio, Stock, PPinpoint, Order, portfolios, next_order_id, nextId

SEED = 34
random.seed(SEED)
np.random.seed(SEED)

class Environment(gym.Env):

    def __init__(self, 
                    id=1,
                    startingcash=1000.0,
                    symbol="AAPL",
                    max_price=100.0,
                    tick_size=0.01,
                    startingprice = 50):
            
        super().__init__()
        self.id = id
        self.startingcash = startingcash
        self.symbol = symbol
        self.max_price = max_price
        self.tick_size = tick_size
        self.startingprice = startingprice

        #setting up price arrays
        self.ticks_per_unit = round(1 / tick_size)
        self.array_size = int(max_price * self.ticks_per_unit) + 1
        pp = PPinpoint()
        self.stock = Stock(
            id=symbol, pricepoint=pp, name=symbol,
            borderArr=[None] * self.array_size,
            sofferArr=[None] * self.array_size,
        )

        #agent portfolio setup        
        portfolios[self.id] = Portfolio(id=self.id, cash=startingcash)
        self.portfolio = portfolios[self.id]

        #trading bots setup
        self.noise_id_1 = 9001
        self.momentum = 9002
        self.meantrader = 9003
        self.ultrader = 9004
        self.GMUtrader = 9005
        self.GMUandMertontrader = 9006

        portfolios[self.noise_id_1] = Portfolio(id=self.noise_id_1, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.momentum] = Portfolio(id=self.momentum, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.meantrader] = Portfolio(id=self.meantrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.ultrader] = Portfolio(id=self.ultrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.GMUtrader] = Portfolio(id=self.GMUtrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.GMUandMertontrader] = Portfolio(id=self.GMUandMertontrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})

        self.mean = 0
        self.count = 0
        self.prevvolatiltiy = 0
        self.epsilon_prev = 0

        #parameters and experimental stuff
        self.addedOBS = True
        self.cancelled = True
 
        #setting up episode prices
        self.priceBefore2 = 0.0 #najstarsejsa
        self.priceBefore = 0.0 #novejsa

        self.last_trade_price = 0.0 #price of the most recent stock trade, se novejsa
        self.price_before_step = 0.0  #price before the step, najnovejsa
        self.step_count = 0        
        self.max_steps = 1000

        #setting up the enviroment for the agent
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        if not self.addedOBS: self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        else: self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)


    def _get_obs(self):

        best_bid = (self.stock.pricepoint.highestindx / 100.0) if self.stock.pricepoint.highestindx != -1 else self.last_trade_price
        best_ask = (self.stock.pricepoint.lowestindx / 100.0) if self.stock.pricepoint.lowestindx != -1 else self.last_trade_price

        if self.price_before_step > 0: price_change = (self.last_trade_price - self.price_before_step) / self.price_before_step
        else: price_change = 0.0

        if self.priceBefore > 0: price_change2 = (self.last_trade_price - self.priceBefore) / self.priceBefore
        else: price_change2 = 0.0

        if self.priceBefore2 > 0: price_change3 = (self.priceBefore - self.priceBefore2) / self.priceBefore2
        else: price_change3 = 0.0

        #Spread (Normalized relative to current price)
        spread = (best_ask - best_bid) / max(1.0, self.last_trade_price)

        #Cash Ratio (Normalized by starting cash)
        cash_ratio = self.portfolio.cash / self.startingcash

        #Position Value Ratio (Value of held shares relative to initial cash)
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        holdings_ratio = (qty * self.last_trade_price) / self.startingcash

        #calculated using garch
        self.epsilon_prev = (self.last_trade_price - self.price_before_step) / self.price_before_step
        volatility =  math.sqrt(0.00001+  0.07 * math.pow(self.epsilon_prev, 2) + 0.90 * math.pow(self.prevvolatiltiy, 2))
        self.prevvolatiltiy = volatility

        if self.addedOBS: return np.array([
            price_change,      # ~ -0.05 to +0.05 (Directional momentum)
            price_change2,
            price_change3,
            spread,            # ~ 0.000 to 0.050 (Order book depth/liquidity)
            cash_ratio,        # ~ 0.000 to 1.000+ (Available buying power)
            holdings_ratio,    # ~ 0.000 to 1.000+ (Current market exposure)
            volatility
        ], dtype=np.float32)
        else: return np.array([
            price_change,      # ~ -0.05 to +0.05 (Directional momentum)
            spread,            # ~ 0.000 to 0.050 (Order book depth/liquidity)
            cash_ratio,        # ~ 0.000 to 1.000+ (Available buying power)
            holdings_ratio,     # ~ 0.000 to 1.000+ (Current market exposure)
            volatility
        ], dtype=np.float32)
 
    def _portfolio_value(self):
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        return self.portfolio.cash + qty * self.last_trade_price
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #resetting the portfolios
        self.portfolio = Portfolio(id=self.id, cash=self.startingcash)
        portfolios[self.id] = self.portfolio 
        portfolios[self.noise_id_1] = Portfolio(id=self.noise_id_1, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.momentum] = Portfolio(id=self.momentum, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.meantrader] = Portfolio(id=self.meantrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.ultrader] = Portfolio(id=self.ultrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.GMUtrader] = Portfolio(id=self.GMUtrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})
        portfolios[self.GMUandMertontrader] = Portfolio(id=self.GMUandMertontrader, cash=1_000_000, holdings={self.stock.id: 1_000_000})

        self.stock.borderArr = [None] * self.array_size
        self.stock.sofferArr = [None] * self.array_size
        self.stock.pricepoint.highestindx = -1
        self.stock.pricepoint.lowestindx = -1

        #resetting the prices of trades
        self.last_trade_price = self.startingprice
        self.price_before_step = self.startingprice 
        self.priceBefore2 = self.startingprice
        self.priceBefore = self.startingprice
        self.step_count = 0

        self.mean = 0
        self.count = 0
        self.epsilon_prev = 0
        self.prevvolatiltiy = 0
 
        return self._get_obs(), {}

    def outstandingOrdersNum(self, port: Portfolio, delnica:Stock):
        #used as a block to calculate if the player has any networth left
        for i, level in enumerate(delnica.borderArr):
            if level is not None and len(level) > 0:
                for order in level:
                    if order.portfolio_id == port.id:
                        return True
                    
        for i, level in enumerate(delnica.sofferArr):
            if level is not None and len(level) > 0:
                for order in level:
                    if order.portfolio_id == port.id:
                        return True
                    
        return False


    def step(self, action):

        count+=1 #count of steps in the episode to calculate the mean

        self.mean = self.mean + 0.5 * (self.last_trade_price - self.mean) #calculating the mean, EWMA

        self.price_before_step = self.last_trade_price   
        value_before = self._portfolio_value()


        momentum_trader_step(self, self.momentum)
        self.count = mean_reversion_trader(self, self.meantrader, self.mean, self.count)
        self.count = Ornstein_trader_step(self, self.ultrader, self.mean, self.count)
        noise_trader_step(self, self.noise_id_1) 
        GBM_trader_step(self, self.GMUtrader)
        self.GMUandMertontrader(self, self.GMUandMertontrader)
        
        act = float(action[0]) if hasattr(action, "__len__") else float(action)
        order_action = "hold"

        #calculating what action the agent will take
        transaction_fee = 0.0005
        required_cash = self.last_trade_price * (1 + transaction_fee)

        if self.cancelled:
            if -0.05 <= act < -0.02:
                order_action = "cancel_soffers"
            elif 0.02 < act <= 0.05:
                order_action = "cancel_borders"
            elif act < -0.05 and self.portfolio.holdings.get(self.stock.id, 0) > 0:
                order_action = "sell"
            elif act > 0.05 and self.portfolio.cash >= required_cash:
                order_action = "buy"
            else:
                order_action = "hold"
        else: 
            if act < -0.05 and self.portfolio.holdings.get(self.stock.id, 0) > 0:
                order_action = "sell"
            elif act > 0.05 and self.portfolio.cash >= required_cash:
                order_action = "buy"
            else:
                order_action = "hold"

        num = min(1.0, abs(act))
        current_shares = self.portfolio.holdings.get(self.stock.id, 0)
        fallback_price = self.last_trade_price if getattr(self, 'last_trade_price', 0) > 0 else self.startingprice
        sharesbought = 0

        if order_action == "buy":
            price_ref = (self.stock.pricepoint.lowestindx / 100.0) if self.stock.pricepoint.lowestindx != -1 else fallback_price
            fee_factor = 1 + getattr(self, 'transaction_fee', 0.0)
            
            # Check if you can afford at least 1 share INCLUDING fees
            if self.portfolio.cash >= (price_ref * fee_factor):
                usable_cash = self.portfolio.cash / fee_factor
                ordersize = int((num * usable_cash) // price_ref) if price_ref > 0 else 0

                if ordersize <= 0: ordersize = 1

                o = Order(next_order_id(True), price_ref, ordersize, marketorlimit=1, stock_id=self.stock.id, portfolio_id=self.id)
                self.MarketGOrder(o, self.stock)

        elif order_action == "sell":
            current_holdings = self.portfolio.holdings.get(self.stock.id, 0)

            if current_holdings > 0:
                price_ref = (self.stock.pricepoint.highestindx / 100.0) if self.stock.pricepoint.highestindx != -1 else fallback_price
                ordersize = int(num * current_holdings)

                if ordersize <= 0: ordersize = 1

                o = Order(next_order_id(False), price_ref, ordersize, marketorlimit=1, stock_id=self.stock.id, portfolio_id=self.id)
                self.MarketGOrder(o, self.stock)  

        elif order_action == "cancel_borders": self.cancel_order(self.id, self.stock, bos=1)
        elif order_action == "cancel_soffers": self.cancel_order(self.id, self.stock, bos=2)

        value_after = self._portfolio_value()
        reward = (value_after - value_before) / 1000

        self.step_count += 1
        self.count = 0
        self.mean = 0

        #checking the networth of the player each step
        no_cash = self.portfolio.cash <= 0
        no_holdings = len(self.portfolio.holdings) == 0
        no_orders = not self.outstandingOrdersNum(self.portfolio, self.stock)
        terminated = bool(self._portfolio_value() <= 1)
 
        truncated = self.step_count >= self.max_steps 
        ordernum = 0

        #sending back diagnostics data
        if order_action == "buy": ordernum = 1
        elif order_action == "sell": ordernum = 2
        elif order_action == "cancel_borders": ordernum = 3
        elif order_action == "cancel_soffers": ordernum = 4
        
        info = { "reward": float(reward), "ordernum": float(ordernum), "val": float(self._portfolio_value())}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None

    #function to find the best either buy or sell offer
    def findNext(self, delnica: Stock, bos: int):

        if bos == 1:  # BUY side needs the next lowest sell price
            for i in range(delnica.pricepoint.lowestindx + 1, len(delnica.sofferArr)):
                if delnica.sofferArr[i] is not None and len(delnica.sofferArr[i]) > 0:
                    delnica.pricepoint.lowestindx = i
                    return i
            delnica.pricepoint.lowestindx = -1
            return -1
        
        else:  # SELL side needs the next highest buy price
            for i in range(delnica.pricepoint.highestindx - 1, -1, -1):
                if delnica.borderArr[i] is not None and len(delnica.borderArr[i]) > 0:
                    delnica.pricepoint.highestindx = i
                    return i
            delnica.pricepoint.highestindx = -1
            return -1

    #function to add an order into its queue
    def addToQueue(self, incoming:Order, delnica:Stock):
        bos = incoming.id // 1_000_000
        index = int(incoming.price * 100)

        if bos == 1:
            array = delnica.borderArr
        else:
            array = delnica.sofferArr

        if index < 0 or index >= len(array):
            print(f"Order price {incoming.price} out of valid range, rejecting")
            return

        if array[index] is None:
            array [index] = deque()

        array[index].append(incoming)

        if bos == 1:
            if delnica.pricepoint.highestindx == -1 or index > delnica.pricepoint.highestindx:
                delnica.pricepoint.highestindx = index
        else:
            if delnica.pricepoint.lowestindx == -1 or index < delnica.pricepoint.lowestindx:
                delnica.pricepoint.lowestindx = index

    #limit order function
    def LimitGOrder(self, incoming: Order, delnica: Stock):
        bos = incoming.id // 1_000_000
        if bos == 1:                                  
            indx = delnica.pricepoint.lowestindx        
            array = delnica.sofferArr                    
        else:                                            
            indx = delnica.pricepoint.highestindx       
            array = delnica.borderArr                    

        # ce ni buy/sellofferjev dodaj v queue
        if indx == -1:
            if bos == 1:
                self.addToQueue(incoming, delnica)     
            else:
                self.addToQueue(incoming, delnica)      
            return

        # pogledamo ce se nam splaca kaj fillat
        pricepoint = indx / 100

        if bos == 1:
            if not pricepoint <= incoming.price:         
                self.addToQueue(incoming, delnica)
                return
        else:
            if not pricepoint >= incoming.price:         
                self.addToQueue(incoming, delnica)
                return

        #while no orders left in the buy or sell offer array
        while indx != -1:
            if incoming.quantity == 0:
                break

            level = array[indx]
            if level is None or len(level) == 0:
                indx = self.findNext(delnica,bos) if bos == 1 else self.findNext(delnica,bos)
                continue

            current_price = indx / 100
            if bos == 1 and current_price > incoming.price:
                break
            if bos == 2 and current_price < incoming.price:
                break

            element = level.popleft()
            filled = self.apply_fill(incoming, element, level, incoming_is_buyer=(bos == 1))

            if filled is None:
                level.appendleft(element)
                break

            if filled == "element":
                # resting order's owner is deficient -- discard it, keep matching
                continue

            if filled == "incoming":
                # the other owner is deficient so invalid trade, add it back to borer or soffer
                level.appendleft(element) 
                break

            #if level is exhausted
            if len(level) == 0:
                if bos == 1:
                    delnica.pricepoint.lowestindx = self.findNext(delnica,bos)
                    indx = delnica.pricepoint.lowestindx
                else:
                    delnica.pricepoint.highestindx = self.findNext(delnica,bos)
                    indx = delnica.pricepoint.highestindx

        if incoming.quantity > 0:
            if bos == 1:
                self.addToQueue(incoming, delnica)
            else:
                self.addToQueue(incoming, delnica)

        return None

    def MarketGOrder(self, incoming: Order, delnica: Stock):

        bos = incoming.id // 1_000_000
        if bos == 1:                                    #BUY
            indx = delnica.pricepoint.lowestindx
            array = delnica.sofferArr
        else:                                            #SELL
            indx = delnica.pricepoint.highestindx
            array = delnica.borderArr

        #if no sell or buy orders, market order just fails -- nothing to fill against
        if indx == -1:
            return incoming.quantity

        while indx != -1:
            if incoming.quantity == 0:
                break

            level = array[indx]
            if level is None or len(level) == 0:
                if bos == 1:
                    delnica.pricepoint.lowestindx = self.findNext(delnica, bos)
                    indx = delnica.pricepoint.lowestindx
                else:
                    delnica.pricepoint.highestindx = self.findNext(delnica, bos)
                    indx = delnica.pricepoint.highestindx
                continue

            element: Order = level.popleft()
            filled = self.apply_fill(incoming, element, level, incoming_is_buyer=(bos == 1))

            # filled returna none ce je portfolio id enak torej self trade
            if filled is None:
                level.appendleft(element)
                break

            if filled == "element":
                # resting order's owner is deficient -- discard it, keep matching
                continue

            if filled == "incoming":
                # the other owner is deficient so invalid trade, add it back to borer or soffer
                level.appendleft(element)  
                break

            if len(level) == 0:
                if bos == 1:
                    delnica.pricepoint.lowestindx = self.findNext(delnica, bos)
                    indx = delnica.pricepoint.lowestindx
                else:
                    delnica.pricepoint.highestindx = self.findNext(delnica, bos)
                    indx = delnica.pricepoint.highestindx

        return incoming.quantity

    def apply_fill(self, incoming: Order, element: Order, level, incoming_is_buyer: bool):
        fill_price = element.price

        #reject self trading
        if incoming.portfolio_id == element.portfolio_id:
            return None

        #Determine fill quantity
        if incoming.quantity >= element.quantity:
            fill_qty = element.quantity
        else:
            fill_qty = incoming.quantity

        #Identify Buyer and Seller
        if incoming_is_buyer:
            buyer_order, seller_order = incoming, element
        else:
            buyer_order, seller_order = element, incoming

        buyer = portfolios[buyer_order.portfolio_id]
        seller = portfolios[seller_order.portfolio_id]
        stock_id = buyer_order.stock_id
        
        transaction_fee = 0.0005
        cost = fill_qty * fill_price
        fee = cost * transaction_fee

        #check if buyer and seller have cash or stocks respectively
        if buyer.cash < cost + fee:
            return "incoming" if incoming_is_buyer else "element"
        if seller.holdings.get(stock_id, 0) < fill_qty:  
            return "element" if incoming_is_buyer else "incoming"

        #Deduct Quantities
        if incoming.quantity >= element.quantity:
            incoming.quantity -= element.quantity
        else:
            element.quantity -= incoming.quantity
            incoming.quantity = 0
            level.appendleft(element)

        #Settle balances
        self.settle_trade(buyer_order, seller_order, fill_qty, fill_price)
        return fill_qty
    
    def settle_trade(self, buyer_order: Order, seller_order: Order, quantity: int, price: float):
        transaction_fee = 0.0005
        cost = quantity * price
        fee = quantity * price * transaction_fee
 
        buyer = portfolios[buyer_order.portfolio_id]
        seller = portfolios[seller_order.portfolio_id]
        stock_id = buyer_order.stock_id
 
        buyer.cash -= (cost + fee)
        buyer.holdings[stock_id] = buyer.holdings.get(stock_id, 0) + quantity 
 
        seller.cash += (cost - fee)
        seller.holdings[stock_id] = seller.holdings.get(stock_id, 0) - quantity  
        if seller.holdings[stock_id] == 0:
            del seller.holdings[stock_id]
 
        self.priceBefore2 = self.priceBefore
        self.priceBefore = self.last_trade_price
        self.last_trade_price = price
        
    def cancel_order(self, portfiolioid,delnica: Stock, bos:int):

        if bos == 1:
            for level in delnica.borderArr:
                if level is not None and len(level) > 0:
                    for order in list(level):  
                        if order.portfolio_id == portfiolioid:
                            level.remove(order)
                            if len(level) == 0:
                                delnica.pricepoint.highestindx = self.findNext(delnica, 1)

        if bos == 2:
            for level in delnica.sofferArr:
                if level is not None and len(level) > 0:
                    for order in list(level):
                        if order.portfolio_id == portfiolioid:
                            level.remove(order)
                            if len(level) == 0:
                                delnica.pricepoint.lowestindx = self.findNext(delnica, 2)
    
    def print_book(self, delnica: Stock):
        print(f"--- Order book for {delnica.name} ---")
        print(f"Best bid (highestindx): {delnica.pricepoint.highestindx} "
            f"({'none' if delnica.pricepoint.highestindx == -1 else delnica.pricepoint.highestindx / 100})")
        print(f"Best ask (lowestindx): {delnica.pricepoint.lowestindx} "
            f"({'none' if delnica.pricepoint.lowestindx == -1 else delnica.pricepoint.lowestindx / 100})")

        print("Buy orders (borderArr):")
        for i, level in enumerate(delnica.borderArr):
            if level is not None and len(level) > 0:
                qtys = [o.quantity for o in level]
                print(f"  price {i/100:.2f}: {qtys}")

        print("Sell offers (sofferArr):")
        for i, level in enumerate(delnica.sofferArr):
            if level is not None and len(level) > 0:
                qtys = [o.quantity for o in level]
                print(f"  price {i/100:.2f}: {qtys}")

    def printOutstandingOrders(self, port: Portfolio, delnica: Stock):
        # printaj vse orderje ki imajo isti portfolio id kot port

        print(f"--- Outstanding orders for portfolio {port.id} ---")

        print("Buy orders:")
        for i, level in enumerate(delnica.borderArr):
            if level is not None and len(level) > 0:
                for order in level:
                    if order.portfolio_id == port.id:
                        print(f"  id={order.id} price={i/100:.2f} qty={order.quantity}")

        print("Sell offers:")
        for i, level in enumerate(delnica.sofferArr):
            if level is not None and len(level) > 0:
                for order in level:
                    if order.portfolio_id == port.id:
                        print(f"  id={order.id} price={i/100:.2f} qty={order.quantity}")
        
    def print_portfolio(self, port: Portfolio):
        print(f"--- Portfolio {port.id} ---")
        print(f"Cash: {port.cash}")
        print(f"Holdings: {port.portfolio}")