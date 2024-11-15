import math
import pandas as pd
from scipy.stats import norm
import numpy as np


def calculate_black_scholes(
    price: float, ex_price: float, rfr: float, ttm: float, sigma: float
) -> float:
    """
    Calculate the theoretical price of a European call option using the Black-Scholes model.

    Parameters
    ----------
    price : float
        The current price of the underlying asset (e.g., stock price).

    ex_price : float
        The exercise (strike) price of the option. This is the price at which
        the option holder can buy the asset upon expiration.

    rfr : float
        The risk-free interest rate (annual), expressed as a decimal (e.g., 0.05 for 5%).
        This represents the return expected from a risk-free asset, such as a government bond.

    ttm : float
        Time to maturity of the option, in years. Represents the time remaining until
        the option's expiration. For example, if there are six months until expiration,
        this value would be 0.5.

    sigma : float
        The volatility of the underlying asset, expressed as a decimal. This represents
        the standard deviation of the asset's returns and is a measure of the asset's
        price fluctuations over time.

    Returns
    -------
    float
        The theoretical price of the European call option as calculated by the Black-Scholes model.

    Notes
    -----
    This implementation assumes that the option is a European-style call option, meaning it can
    only be exercised at expiration, not before. The Black-Scholes model uses several key
    assumptions, including a constant risk-free rate, constant volatility, and that the asset
    price follows a log-normal distribution.

    The formula for the Black-Scholes model is as follows:
        C = S_0 * N(d1) - X * exp(-r * T) * N(d2)
    where:
        - C is the call option price
        - S_0 is the current asset price
        - X is the exercise price
        - T is the time to maturity
        - r is the risk-free rate
        - sigma is the volatility of the asset
        - N(d1) and N(d2) are the cumulative distribution functions of a standard normal distribution

    The values `d1` and `d2` are calculated as:
        d1 = [ln(S_0 / X) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Examples
    --------
    >>> price = 100       # Current stock price
    >>> ex_price = 100    # Strike price of the option
    >>> rfr = 0.05        # Risk-free interest rate (5%)
    >>> ttm = 1           # Time to expiration in years
    >>> sigma = 0.2       # Volatility of the underlying asset (20%)
    >>> calculate_black_scholes(price, ex_price, rfr, ttm, sigma)
    10.450583572185565
    """

    d1 = (math.log(price / ex_price) + (rfr + 0.5 * sigma**2) * ttm) / (
        sigma * math.sqrt(ttm)
    )
    d2 = d1 - sigma * math.sqrt(ttm)

    call_price = price * norm.cdf(d1) - ex_price * math.exp(-rfr * ttm) * norm.cdf(d2)

    return call_price


class BlackScholesModel:
    """A class to represent the Black-Scholes option pricing model."""

    def __init__(self, S: float, K: float, T: float, sigma: float, r: float = 0.05):
        """
        Initialize the Black-Scholes model with the given parameters.

        Parameters
        ----------
        S : float
            The current price of the underlying asset (e.g., stock price).

        K : float
            The exercise (strike) price of the option.

        T : float
            Time to expiration in years.

        sigma : float
            The volatility of the underlying asset.

        r : float, optional
            The risk-free interest rate (default is 0.05 for 5%).
        """
        self.S = S  # Underlying asset price
        self.K = K  # Option strike price
        self.T = T  # Time to expiration in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

    def d1(self) -> float:
        """Calculate d1 in the Black-Scholes formula."""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    def d2(self) -> float:
        """Calculate d2 in the Black-Scholes formula."""
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_option_price(self) -> float:
        """Calculate the price of a European call option."""
        return self.S * norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(
            -self.r * self.T
        ) * norm.cdf(self.d2(), 0.0, 1.0)

    def put_option_price(self) -> float:
        """Calculate the price of a European put option."""
        return self.K * np.exp(-self.r * self.T) * norm.cdf(
            -self.d2(), 0.0, 1.0
        ) - self.S * norm.cdf(-self.d1(), 0.0, 1.0)


def calculate_historical_volatility(
    stock_data: pd.DataFrame, window: int = 252
) -> float:
    """Calculate the historical volatility of a stock based on its closing prices.

    Parameters
    ----------
    stock_data : pd.DataFrame
        A DataFrame containing stock price data with a 'Close' column.

    window : int
        The number of periods to use for calculating volatility (default is 252 for daily data).

    Returns
    -------
    float
        The historical volatility of the stock.
    """
    log_returns = np.log(stock_data["Close"] / stock_data["Close"].shift(1))
    volatility = np.sqrt(window) * log_returns.std()
    return volatility


class BlackScholesGreeks(BlackScholesModel):
    def delta_call(self):
        return norm.cdf(self.d1(), 0.0, 1.0)

    def delta_put(self):
        return -norm.cdf(-self.d1(), 0.0, 1.0)

    def gamma(self):
        return norm.pdf(self.d1(), 0.0, 1.0) / (self.S * self.sigma * np.sqrt(self.T))

    def theta_call(self):
        return -self.S * norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (
            2 * np.sqrt(self.T)
        ) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2(), 0.0, 1.0)

    def theta_put(self):
        return -self.S * norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (
            2 * np.sqrt(self.T)
        ) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2(), 0.0, 1.0)

    def vega(self):
        return self.S * norm.pdf(self.d1(), 0.0, 1.0) * np.sqrt(self.T)

    def rho_call(self):
        return (
            self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2(), 0.0, 1.0)
        )

    def rho_put(self):
        return (
            -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2(), 0.0, 1.0)
        )
