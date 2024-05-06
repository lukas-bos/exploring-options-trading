# File for the strategy class

from risk_metrics import BlackScholes, Risk

class Leg:
    def __init__(self, underlying, strike, maturity, rate, price, option='call'):
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.price = price

        self.sigma = BlackScholes(underlying, strike, maturity, rate, option).implied_volatility(0.2, price)
        risks = Risk(underlying, strike, maturity, rate, self.sigma, option)
        self.delta = risks.delta()
        self.gamma = risks.gamma()
        self.vega = risks.vega()

        self.option = option

class Strategy:
    def __init__(self, legs):
        self.legs = legs

    def delta(self):
        """
        Calculate the delta of the option strategy.

        Returns:
        float: delta of the option strategy
        """
        delta = 0
        for leg in self.legs:
            delta += leg.delta
        return delta

    def gamma(self):
        """
        Calculate the gamma of the option strategy.

        Returns:
        float: gamma of the option strategy
        """
        gamma = 0
        for leg in self.legs:
            gamma += leg.gamma
        return gamma

    def vega(self):
        """
        Calculate the vega of the option strategy.

        Returns:
        float: vega of the option strategy
        """
        vega = 0
        for leg in self.legs:
            vega += leg.vega
        return vega


if __name__ == '__main__':
    leg1 = Leg(100, 100, 1, 0.05, 10)
    leg2 = Leg(100, 110, 1, 0.05, 5, option='put')

    strategy = Strategy([leg1, leg2])
    print(strategy.delta())
    print(strategy.gamma())
    print(strategy.vega())
    