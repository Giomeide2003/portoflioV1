import unittest

from src.volatility_greeks import (
    Option,
    greeks,
    implied_volatility,
    option_price,
)


class VolatilityGreeksTests(unittest.TestCase):

    def setUp(self):
        self.option = Option(
            spot=100.0,
            strike=105.0,
            time_to_expiry=45 / 365,
            risk_free_rate=0.03,
            volatility=0.22,
            option_type="call",
        )

    def test_call_price_is_positive(self):
        self.assertGreater(option_price(self.option), 0)

    def test_put_call_delta_relationship(self):
        call = Option(100, 100, 30 / 365, 0.03, 0.20, "call")
        put = Option(100, 100, 30 / 365, 0.03, 0.20, "put")

        self.assertAlmostEqual(
            greeks(call)["delta"] - greeks(put)["delta"],
            1.0,
            places=10,
        )

    def test_gamma_and_vega_are_positive(self):
        values = greeks(self.option)
        self.assertGreater(values["gamma"], 0)
        self.assertGreater(values["vega"], 0)

    def test_implied_volatility_recovers_input(self):
        market_price = option_price(self.option)
        iv = implied_volatility(
            market_price,
            self.option.spot,
            self.option.strike,
            self.option.time_to_expiry,
            self.option.risk_free_rate,
            self.option.option_type,
        )

        self.assertAlmostEqual(iv, self.option.volatility, places=6)


if __name__ == "__main__":
    unittest.main()
