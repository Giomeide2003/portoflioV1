"""Simple backtest engine for a long-only trading strategy."""


def compute_returns(prices: list[float]) -> list[float]:
    """Compute simple period-to-period returns."""
    if len(prices) < 2:
        raise ValueError("at least two prices are required")
    if any(price <= 0 for price in prices):
        raise ValueError("prices must be positive")

    return [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]


def moving_average(prices: list[float], window: int) -> list[float | None]:
    """Return a simple moving average aligned with the input series."""
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(prices):
        raise ValueError("window cannot exceed the number of prices")

    averages: list[float | None] = [None] * (window - 1)
    averages.extend(
        sum(prices[i - window + 1 : i + 1]) / window
        for i in range(window - 1, len(prices))
    )
    return averages


def generate_long_only_signal(
    prices: list[float], short_window: int, long_window: int
) -> list[int]:
    """Generate a long-only signal from two moving averages.

    The strategy holds one unit of the asset when the short moving average
    is above the long moving average, and stays flat otherwise.
    """
    if short_window <= 0 or long_window <= 0:
        raise ValueError("moving-average windows must be positive")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    short_ma = moving_average(prices, short_window)
    long_ma = moving_average(prices, long_window)

    return [
        int(short is not None and long is not None and short > long)
        for short, long in zip(short_ma, long_ma)
    ]


def backtest_long_only(prices: list[float], short_window: int, long_window: int) -> list[float]:
    """Compute strategy returns using the previous-period signal.

    Shifting the signal by one period avoids using today's close to generate
    a position that is assumed to earn today's return.
    """
    asset_returns = compute_returns(prices)
    signals = generate_long_only_signal(prices, short_window, long_window)
    positions = signals[:-1]

    return [position * asset_return for position, asset_return in zip(positions, asset_returns)]
