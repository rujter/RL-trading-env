#dejmo rect da je cena 0 do 100 (0-10001)
from collections import deque
import time

class Portfolio:
    def __init__(self, portfolio:dict, cash):
        self.portfolio = portfolio
        self.cash = cash

portfolios = {}   # portfolio_id -> Portfolio

class Stock:
    def __init__(self, id, pricepoint, name, borderArr, sofferArr):
        self.id = id                                                 
        self.pricepoint = pricepoint
        self.name = name
        self.borderArr = borderArr
        self.sofferArr = sofferArr

class PPinpoint: ##ZA VSAKo delnico MORA IMET HIGHEST IN LOWEST PRICE indx
    def __init__(self, highestindx, lowestindx):
        self.highestindx = highestindx ##to uporabis ko prodajas in isces najvecji buyorder
        self.lowestindx = lowestindx #ko kupujes in isces najceneje selloffer

class Order:
    def __init__(self, id, price, quantity, marketorlimit, stock_id, portfolio_id):
        self.id = id
        self.price = price
        self.quantity = quantity
        self.marketorlimit = marketorlimit
        self.timestamp = time.time()
        self.stock_id = stock_id
        self.portfolio_id = portfolio_id

def create_stock(id, name, tick_size=0.01, max_price=100.0):
    ticks_per_unit = round(1 / tick_size)
    array_size = int(max_price * ticks_per_unit) + 1

    pp = PPinpoint(-1, -1)
    border = [None] * array_size
    soffer = [None] * array_size

    stock = Stock(id, pp, name, border, soffer)
    stock.tick_size = tick_size
    stock.ticks_per_unit = ticks_per_unit
    return stock

def print_book(delnica: Stock):
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

def printOutstandingOrders(port: Portfolio, delnica: Stock):
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
    
def print_portfolio(port: Portfolio):
    print(f"--- Portfolio {port.id} ---")
    print(f"Cash: {port.cash}")
    print(f"Holdings: {port.portfolio}")

nextId = 1  #global counter

def next_order_id(isbuy: bool):
    global nextId

    if isbuy: prefix = 1
    else: prefix = 2

    order_id = prefix * 1000000 + nextId
    nextId += 1
    return order_id

stocks = {}

def Whattodo():
    portfolio_id = int(input("Your portfolio id: "))
    ticker = input("Ticker: ")
    delnica = stocks[ticker]

    isbuy
    bos = int(input("Buy(1) or Sell(2) or Cancel order(3)"))
    if bos == 3:
        printOutstandingOrders(portfolios[portfolio_id], delnica)
        order_id_to_cancel = int(input("Enter the order id you'd like to cancel: "))
        cancel_order(order_id_to_cancel, delnica)

    #NAREDI FOREVER LOOP 
    price = float(input("Price: "))
    quantity = int(input("Quantity: "))
    marketorlimit = int(input("Market(1) or Limit(2) order"))

    if ticker not in stocks:
        print(f"No such stock: {ticker}")
        return

    if bos == 1: isbuy = True
    else: isbuy = False

    o = Order(next_order_id(isbuy), price, quantity, marketorlimit, 1, stock_id=delnica.id, portfolio_id=portfolio_id)  

    #Determins whether its a market or limit order
    if marketorlimit == 1: MarketGOrder(o, delnica)
    else: LimitGOrder(o, delnica)

#BUYING LOGIC#########################################################################
            
def findNxtSmallest(delnica:Stock): #najde najmanjsi naslednji index v soffer arrayu

    for i in range(delnica.pricepoint.lowestindx+1, len(delnica.sofferArr)):
        if delnica.sofferArr[i] is not None and len(delnica.sofferArr[i]) > 0:
            delnica.pricepoint.lowestindx = i
            return i
    
    delnica.pricepoint.lowestindx = -1
    return -1

def addToSofferQueue(soffer: Order, delnica: Stock):
    index = int(soffer.price * 100)

    if delnica.sofferArr[index] is None:
        delnica.sofferArr[index] = deque()

    delnica.sofferArr[index].append(soffer)

    if delnica.pricepoint.lowestindx == -1 or index < delnica.pricepoint.lowestindx:
        delnica.pricepoint.lowestindx = index

#SELLING LOGIC#########################################################################

def findNxtBiggest(delnica: Stock):  # najde naslednji najvecji index v border arrayu
    for i in range(delnica.pricepoint.highestindx - 1, -1, -1):   # walk DOWNWARD from current best buy price
        if delnica.borderArr[i] is not None and len(delnica.borderArr[i]) > 0:
            delnica.pricepoint.highestindx = i
            return i

    delnica.pricepoint.highestindx = -1
    return -1

def addToBorderQueue(border:Order, delnica:Stock):

    index = int(border.price * 100)

    if delnica.borderArr[index] is None:
        delnica.borderArr[index] = deque()

    delnica.borderArr[index].append(border)

    if delnica.pricepoint.highestindx == -1 or index > delnica.pricepoint.highestindx:
        delnica.pricepoint.highestindx = index

#IDENTICAL HELPER FUNCTIONS###################################3
def LimitGOrder(incoming: Order, delnica: Stock):

    bos = incoming.id // 1_000_000
    if bos == 1:                                    # BUY
        indx = delnica.pricepoint.lowestindx         # $$$$$$$$$$$$ FIXED: typo lowestindex -> lowestindx
        array = delnica.sofferArr                    # $$$$$$$$$$$$ FIXED: a buy checks the SELL book, not borderArr
    else:                                            # SELL
        indx = delnica.pricepoint.highestindx        # $$$$$$$$$$$$ FIXED: typo highestindex -> highestindx
        array = delnica.borderArr                    # $$$$$$$$$$$$ FIXED: a sell checks the BUY book, not sofferArr

    # ce ni buy/sellofferjev dodaj v queue
    if indx == -1:
        if bos == 1:
            addToBorderQueue(incoming, delnica)       # $$$$$$$$$$$$ FIXED: pass incoming (the Order), not array
        else:
            addToSofferQueue(incoming, delnica)       # $$$$$$$$$$$$ FIXED: correct queue function for sells
        return

    # pogledamo ce se nam splaca kaj fillat
    pricepoint = indx / 100

    if bos == 1:
        if not pricepoint <= incoming.price:          # $$$$$$$$$$$$ FIXED: incoming.price, not array.price
            addToBorderQueue(incoming, delnica)
            return
    else:
        if not pricepoint >= incoming.price:          # $$$$$$$$$$$$ FIXED: incoming.price, not array.price
            addToSofferQueue(incoming, delnica)
            return

    # from here on, matching loop would go -- walking `array`, using `indx` as the current position,
    # same structure as your original BuyOrder/SellOffer while-loops
    while indx != -1:
        if incoming.quantity == 0:
            break

        level = array[indx]
        if level is None or len(level) == 0:
            indx = findNxtSmallest(delnica) if bos == 1 else findNxtBiggest(delnica)
            continue

        current_price = indx / 100
        if bos == 1 and current_price > incoming.price:
            break
        if bos == 2 and current_price < incoming.price:
            break

        element = level.popleft()
        filled = apply_fill(incoming, element, level, incoming_is_buyer=(bos == 1))

        if filled is None:
            level.appendleft(element)
            break

        if filled == "element":
            # resting order's owner is deficient -- discard it, keep matching
            continue

        if filled == "incoming":
            # incoming order's owner is deficient -- put the resting order back, stop matching
            break

        if len(level) == 0:
            if bos == 1:
                delnica.pricepoint.lowestindx = findNxtSmallest(delnica)
                indx = delnica.pricepoint.lowestindx
            else:
                delnica.pricepoint.highestindx = findNxtBiggest(delnica)
                indx = delnica.pricepoint.highestindx

    if incoming.quantity > 0:
        if bos == 1:
            addToBorderQueue(incoming, delnica)
        else:
            addToSofferQueue(incoming, delnica)

    return None

def MarketGOrder(incoming: Order, delnica: Stock):

    bos = incoming.id // 1_000_000
    if bos == 1:                                    # BUY
        indx = delnica.pricepoint.lowestindx
        array = delnica.sofferArr
    else:                                            # SELL
        indx = delnica.pricepoint.highestindx
        array = delnica.borderArr

    # ce ni sellofferjev/buyorderjev, market order just fails -- nothing to fill against
    if indx == -1:
        return incoming.quantity

    while indx != -1:
        if incoming.quantity == 0:
            break

        level = array[indx]
        if level is None or len(level) == 0:
            if bos == 1:
                delnica.pricepoint.lowestindx = findNxtSmallest(delnica)
                indx = delnica.pricepoint.lowestindx
            else:
                delnica.pricepoint.highestindx = findNxtBiggest(delnica)
                indx = delnica.pricepoint.highestindx
            continue

        element: Order = level.popleft()
        filled = apply_fill(incoming, element, level, incoming_is_buyer=(bos == 1))

        # filled returna none ce je portfolio id enak torej self trade
        if filled is None:
            level.appendleft(element)
            break

        if filled == "element":
            # resting order's owner is deficient -- discard it, keep matching
            continue

        if filled == "incoming":
            # incoming order's owner is deficient -- put the resting order back, stop matching
            break

        if len(level) == 0:
            if bos == 1:
                delnica.pricepoint.lowestindx = findNxtSmallest(delnica)
                indx = delnica.pricepoint.lowestindx
            else:
                delnica.pricepoint.highestindx = findNxtBiggest(delnica)
                indx = delnica.pricepoint.highestindx

    return incoming.quantity

def apply_fill(incoming: Order, element: Order, level, incoming_is_buyer: bool):
    fill_price = element.price

    if incoming.portfolio_id == element.portfolio_id:
        return None

    if incoming.quantity >= element.quantity:
        fill_qty = element.quantity
    else:
        fill_qty = incoming.quantity

    if incoming_is_buyer:
        buyer_order, seller_order = incoming, element
    else:
        buyer_order, seller_order = element, incoming

    buyer = portfolios[buyer_order.portfolio_id]
    seller = portfolios[seller_order.portfolio_id]
    stock_id = buyer_order.stock_id
    cost = fill_qty * fill_price

    if buyer.cash < cost:
        return "incoming" if incoming_is_buyer else "element"
    if seller.portfolio.get(stock_id, 0) < fill_qty:
        return "element" if incoming_is_buyer else "incoming"

    
    if incoming.quantity >= element.quantity:
        incoming.quantity -= element.quantity
    else:
        element.quantity -= incoming.quantity
        incoming.quantity = 0
        level.appendleft(element)

    settle_trade(buyer_order, seller_order, fill_qty, fill_price)
    return fill_qty

def settle_trade(buyer_order: Order, seller_order: Order, quantity: int, price: float):
    transaction_fee = 0.0005 #0.05%

    buyer = portfolios[buyer_order.portfolio_id]
    seller = portfolios[seller_order.portfolio_id]
    stock_id = buyer_order.stock_id

    cost = quantity * price
    fee = quantity * price * transaction_fee

    buyer.cash -= (cost + fee)
    buyer.portfolio[stock_id] = buyer.portfolio.get(stock_id, 0) + quantity

    seller.cash += (cost - fee)
    seller.portfolio[stock_id] = seller.portfolio.get(stock_id, 0) - quantity
    if seller.portfolio[stock_id] == 0:
        del seller.portfolio[stock_id]

def cancel_order(order_id: int, delnica: Stock):
    # pejdi cez cel array od border in pol soffer
    for i in range(len(delnica.borderArr)):
        level = delnica.borderArr[i]
        if level is not None and len(level) > 0:
            for order in level:
                if order.id == order_id:
                    level.remove(order)
                    print(f"Cancelled order {order_id}")
                    return True

    for i in range(len(delnica.sofferArr)):
        level = delnica.sofferArr[i]
        if level is not None and len(level) > 0:
            for order in level:
                if order.id == order_id:
                    level.remove(order)
                    print(f"Cancelled order {order_id}")
                    return True

    print(f"Order {order_id} not found")
    return False
