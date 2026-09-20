import unittest
from pricematcher import (
    Stock, PPinpoint, Portfolio, Order,
    portfolios, LimitGOrder, MarketGOrder,
    create_stock, cancel_order, next_order_id, print_book,
    print_portfolio
)

class TestMyExchange(unittest.TestCase):

        def setUp(self):
            # runs automatically before EVERY test method -- fresh state each time
            portfolios.clear()

            self.stock = create_stock(1, "test")

            self.buyer = Portfolio({}, 1000)
            self.buyer.id = 1
            portfolios[1] = self.buyer

            self.seller = Portfolio({1: 100}, 0)
            self.seller.id = 2
            portfolios[2] = self.seller

            self.seller1 = Portfolio({1: 100}, 0)
            self.seller1.id = 3
            portfolios[3] = self.seller1

        def test_basictest(self):

            sell = Order(next_order_id(False), 10.0, 20, 1, stock_id=1, portfolio_id=2)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            print("\n")
            print_book(self.stock)

            buy = Order(next_order_id(1), 10.0, 20, 2, stock_id=1, portfolio_id=1)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            self.assertEqual(self.buyer.cash, 799.9)
            self.assertEqual(self.buyer.portfolio.get(1), 20)
            self.assertEqual(self.seller.cash, 199.9)
            self.assertEqual(self.seller.portfolio.get(1), 80)

        def test_cancelSelling(self):
            sell = Order(next_order_id(False), 10.0, 20, 1, stock_id=1, portfolio_id=2)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            print("\n")
            print_book(self.stock)

            cancel_order(sell.id, self.stock)

            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #prodajalec nima delnic,1000 deanrja
            #seller ima 1: 100 delnic in 0 denarja

            self.assertEqual(self.seller.portfolio.get(1), 100)
            self.assertEqual(self.stock.pricepoint.lowestindx, -1)
            self.assertEqual(self.stock.pricepoint.highestindx, 1000)

        def test_cancelBUying(self):
            sell = Order(next_order_id(False), 10.0, 20, 1, stock_id=1, portfolio_id=2)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(buy, self.stock)
            print("\n")
            print_book(self.stock)

            cancel_order(buy.id, self.stock)

            LimitGOrder(sell, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #prodajalec nima delnic,1000 deanrja
            #seller ima 1: 100 delnic in 0 denarja

            self.assertEqual(self.seller.portfolio.get(1), 100)
            self.assertEqual(self.stock.pricepoint.lowestindx, 1000)
            self.assertEqual(self.stock.pricepoint.highestindx, -1)

        def test_partialFIll(self):
            sell = Order(next_order_id(False), 10.0, 10, 1, stock_id=1, portfolio_id=2)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #seller ima 100 delnic
            #buyer ima 0 delnic (oboje na zacetku)

            self.assertEqual(self.stock.pricepoint.highestindx, 1000)
            self.assertEqual(self.stock.pricepoint.lowestindx, -1)
            self.assertEqual(self.seller.portfolio.get(1), 90)
            self.assertEqual(self.buyer.portfolio.get(1), 10)

            #MORAM UPDATAT VALUE ZA HIGHEST IN LOWEST INDEX V APPLY

        def test_2sellerssamepricepoint(self):
            sell = Order(next_order_id(False), 10.0, 10, 1, stock_id=1, portfolio_id=2)
            sell1 = Order(next_order_id(False), 10.0, 20, 1, stock_id=1, portfolio_id=3)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            LimitGOrder(sell1, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.seller1) 
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #seller ima 100 delnic
            #buyer ima 0 delnic (oboje na zacetku)

            self.assertEqual(self.stock.pricepoint.lowestindx, 1000)
            self.assertEqual(self.buyer.portfolio.get(1), 20)
            self.assertEqual(self.seller.portfolio.get(1), 90)
            self.assertEqual(self.seller1.portfolio.get(1), 90)

        def test_2sellerssDIFFRENTpricepoint(self):
            sell = Order(next_order_id(False), 10.0, 10, 1, stock_id=1, portfolio_id=2)
            sell1 = Order(next_order_id(False), 12.0, 20, 1, stock_id=1, portfolio_id=3)
            buy = Order(next_order_id(True), 15.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            LimitGOrder(sell1, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.seller1) 
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #seller ima 100 delnic
            #buyer ima 0 delnic (oboje na zacetku)

            self.assertEqual(self.stock.pricepoint.lowestindx, 1200)
            self.assertEqual(self.buyer.portfolio.get(1), 20)
            self.assertEqual(self.seller.portfolio.get(1), 90)
            self.assertEqual(self.seller1.portfolio.get(1), 90)

        def test_2sellerssBIGGERpricepoint(self):
            sell = Order(next_order_id(False), 10.0, 10, 1, stock_id=1, portfolio_id=2)
            sell1 = Order(next_order_id(False), 11.0, 20, 1, stock_id=1, portfolio_id=3)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            LimitGOrder(sell1, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.seller1) 
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #seller ima 100 delnic
            #buyer ima 0 delnic (oboje na zacetku)

            self.assertEqual(self.stock.pricepoint.lowestindx, 1100)
            self.assertEqual(self.buyer.portfolio.get(1), 10)
            self.assertEqual(self.seller.portfolio.get(1), 90)
            self.assertEqual(self.seller1.portfolio.get(1), 100)

        def test_2sellerssShouldntfill(self):
            sell = Order(next_order_id(False), 11.0, 10, 1, stock_id=1, portfolio_id=2)
            sell1 = Order(next_order_id(False), 11.0, 20, 1, stock_id=1, portfolio_id=3)
            buy = Order(next_order_id(True), 10.0, 20, 2, stock_id=1, portfolio_id=1)            

            LimitGOrder(sell, self.stock)
            LimitGOrder(sell1, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller)
            print("\n")
            print_portfolio(self.seller1) 
            print("\n")
            print_portfolio(self.buyer) 
            print("\n")

            #seller ima 100 delnic
            #buyer ima 0 delnic (oboje na zacetku)

            self.assertEqual(self.stock.pricepoint.lowestindx, 1100)
            self.assertEqual(self.buyer.portfolio.get(1), None)
            self.assertEqual(self.seller.portfolio.get(1), 100)
            self.assertEqual(self.seller1.portfolio.get(1), 100)

        def test_insufficientFunds(self):
            # buyer2 has no cash, tries to buy from a seller with plenty of stock
            self.seller3 = Portfolio({1: 100}, 0)
            self.seller3.id = 6
            portfolios[6] = self.seller3

            self.buyer2 = Portfolio({}, 5.0)   # only 5.0 cash, not enough for 10 @ 10.0 = 100
            self.buyer2.id = 7
            portfolios[7] = self.buyer2

            sell = Order(next_order_id(False), 10.0, 10, 1, stock_id=1, portfolio_id=6)
            buy = Order(next_order_id(True), 10.0, 10, 2, stock_id=1, portfolio_id=7)

            LimitGOrder(sell, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller3)
            print("\n")
            print_portfolio(self.buyer2)
            print("\n")

            # nothing should have filled: buyer can't cover the cost
            self.assertEqual(self.seller3.cash, 0)
            self.assertEqual(self.seller3.portfolio.get(1), 100)   # unchanged
            self.assertEqual(self.buyer2.cash, 5.0)                # unchanged
            self.assertEqual(self.buyer2.portfolio.get(1), None)   # never got shares

            # sell order should still be resting in the book since fill was rejected
            self.assertIn(sell, self.stock.sofferArr[1000])

        def test_insufficientStock(self):
            self.seller4 = Portfolio({1: 3}, 0)
            self.seller4.id = 8
            portfolios[8] = self.seller4

            self.buyer3 = Portfolio({}, 1000.0)
            self.buyer3.id = 9
            portfolios[9] = self.buyer3

            sell = Order(next_order_id(False), 10.0, 10, 1, stock_id=1, portfolio_id=8)
            buy = Order(next_order_id(True), 10.0, 10, 2, stock_id=1, portfolio_id=9)

            LimitGOrder(sell, self.stock)
            LimitGOrder(buy, self.stock)

            print("\n")
            print_book(self.stock)
            print("\n")
            print_portfolio(self.seller4)
            print("\n")
            print_portfolio(self.buyer3)
            print("\n")

            # nothing should have filled: seller doesn't have enough shares
            self.assertEqual(self.seller4.cash, 0)
            self.assertEqual(self.seller4.portfolio.get(1), 3)     # unchanged
            self.assertEqual(self.buyer3.cash, 1000.0)              # unchanged
            self.assertEqual(self.buyer3.portfolio.get(1), None)   # never got shares

            # seller was at fault, so their resting sell order gets discarded from the book
            self.assertTrue(
                self.stock.sofferArr[1000] is None or len(self.stock.sofferArr[1000]) == 0
            )

            # the buyer's order, having nothing left to match against, ends up resting instead
            self.assertIn(buy, self.stock.borderArr[1000])

if __name__ == "__main__":
    unittest.main()