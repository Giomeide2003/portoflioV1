import unittest

import numpy as np

from src.gamma_exposure import (
    Option,
    aggregate_gamma_exposure,
    black_scholes_gamma,
    find_gamma_flip,
    gamma_exposure,
    gamma_profile,
)


class GammaExposureTests(unittest.TestCase):

    def test_black_scholes_gamma_is_positive(self):
        gamma = black_scholes_gamma(100, 100, 30 / 365, 0.20)
        self.assertGreater(gamma, 0)

    def test_call_and_put_have_same_gamma(self):
        # Black-Scholes gamma is identical for calls and puts with
        # the same strike, maturity and volatility.
        call = Option(100, 30 / 365, 0.20, 100, "call")
        put = Option(100, 30 / 365, 0.20, 100, "put")

        self.assertAlmostEqual(
            gamma_exposure(100, call),
            gamma_exposure(100, put),
        )

    def test_aggregate_exposure(self):
        option = Option(100, 30 / 365, 0.20, 100, "call")
        exposure = aggregate_gamma_exposure(100, [option])
        self.assertGreater(exposure, 0)

    def test_gamma_flip_interpolation(self):
        spots = np.array([99.0, 100.0, 101.0])
        exposure = np.array([-10.0, 0.0, 10.0])

        self.assertEqual(find_gamma_flip(spots, exposure), 100.0)

    def test_gamma_profile_shape(self):
        option = Option(100, 30 / 365, 0.20, 100, "call")
        spots = np.linspace(95, 105, 11)
        profile = gamma_profile(spots, [option])

        self.assertEqual(profile.shape, spots.shape)


if __name__ == "__main__":
    unittest.main()
