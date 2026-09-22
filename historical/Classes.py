# Classes.py
import time
nextId = 1
def next_order_id(isbuy: bool):
    global nextId

    if isbuy: prefix = 1
    else: prefix = 2

    order_id = prefix * 1000000 + nextId
    nextId += 1
    return order_id

# Shared portfolio registry
portfolios = {}

class Portfolio:
    def __init__(self, id, cash, holdings=None):
        self.id = id
        self.cash = cash
        self.holdings = holdings if holdings is not None else {}

class Stock:
    def __init__(self, id, pricepoint, name, borderArr, sofferArr):
        self.id = id
        self.pricepoint = pricepoint
        self.name = name
        self.borderArr = borderArr
        self.sofferArr = sofferArr

class PPinpoint:
    def __init__(self, highestindx=-1, lowestindx=-1):
        self.highestindx = highestindx
        self.lowestindx = lowestindx

class Order:
    def __init__(self, id, price, quantity, marketorlimit, stock_id, portfolio_id):
        self.id = id
        self.price = price
        self.quantity = quantity
        self.marketorlimit = marketorlimit
        self.timestamp = time.time()
        self.stock_id = stock_id
        self.portfolio_id = portfolio_id
