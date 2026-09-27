"""Small example showing the option-pricing workflow."""

from src.black_scholes import black_scholes_price
from src.greeks import delta, gamma, theta, vega
from src.implied_volatility import implied_volatility


SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
VOLATILITY = 0.20


call_price = black_scholes_price(
    SPOT, STRIKE, MATURITY, RATE, VOLATILITY, "call"
)

print(f"Call price: {call_price:.4f}")
print(f"Delta: {delta(SPOT, STRIKE, MATURITY, RATE, VOLATILITY, 'call'):.4f}")
print(f"Gamma: {gamma(SPOT, STRIKE, MATURITY, RATE, VOLATILITY):.4f}")
print(f"Vega: {vega(SPOT, STRIKE, MATURITY, RATE, VOLATILITY):.4f}")
print(f"Theta: {theta(SPOT, STRIKE, MATURITY, RATE, VOLATILITY, 'call'):.4f}")

recovered_volatility = implied_volatility(
    market_price=call_price,
    spot=SPOT,
    strike=STRIKE,
    time_to_maturity=MATURITY,
    risk_free_rate=RATE,
    option_type="call",
)

print(f"Implied volatility: {recovered_volatility:.2%}")
