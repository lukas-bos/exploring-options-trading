import numpy as np
from scipy.stats import norm
from scipy.optimize import fmin

class BlackScholes:
    def __init__(self, S, K, T, r, option='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.option = option

    def black_scholes(self, sigma):
        """
        Calculate the theoretical price of a European option using the Black-Scholes formula.

        Paramaters:
        S (float) = spot price of the underlying asset
        K (float) = strike price of the option
        T (float) = time to expiration in years
        r (float) = risk-free rate
        sigma (float) = volatility of the underlying asset
        option (str, optional) = type of option (call or put)

        Returns:
        float: Theoretical price of the option
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)

        if self.option == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    
    def implied_volatility(self, sigma, price):
        """
        Calculate the implied volatility for a European option using the Black-Scholes formula.

        Parameters:
        S (float) = spot price of the underlying asset
        K (float) = strike price of the option
        T (float) = time to expiration in years
        r (float) = risk-free rate
        sigma (float) = estimate of the volatility of the underlying asset
        price (float) = price of the option (premium)
        option (str, optional) = type of option (call or put)

        Returns:
        float: implied volatility of the option
        """
        target = lambda sigma: (self.black_scholes(sigma) - price) ** 2
        
        res = fmin(target, [sigma], disp = False)
        return res[0]


class Risk:
    def __init__(self, S, K, T, r, sigma, option='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.option = option
        self.sigma = sigma

    def delta(self):
        """
        Calculate the delta of a European option using the Black-Scholes formula.

        Parameters:
        S (float) = spot price of the underlying asset
        K (float) = strike price of the option
        T (float) = time to expiration in years
        r (float) = risk-free rate
        sigma (float) = volatility of the underlying asset
        option (str, optional) = type of option (call or put)

        Returns:
        float: delta of the option
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.option == 'call':
            return norm.cdf(d1)
        elif self.option == 'put':
            return norm.cdf(d1) - 1
    
    def gamma(self):
        """
        Calculate the gamma of a European option using the Black-Scholes formula.

        Parameters:
        S (float) = spot price of the underlying asset
        K (float) = strike price of the option
        T (float) = time to expiration in years
        r (float) = risk-free rate
        sigma (float) = volatility of the underlying asset

        Returns:
        float: gamma of the option
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """
        Calculate the vega of a European option using the Black-Scholes formula.

        Parameters:
        S (float) = spot price of the underlying asset
        K (float) = strike price of the option
        T (float) = time to expiration in years
        r (float) = risk-free rate
        sigma (float) = volatility of the underlying asset

        Returns:
        float: vega of the option
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100