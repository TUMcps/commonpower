import unittest

from pathlib import Path
from datetime import timedelta

from commonpower.data_forecasting import *


class TestForecasting(unittest.TestCase):
    
    def setUp(self):
        
        self.horizon = timedelta(hours=2)
        self.freq = timedelta(hours=1)

        data_path = Path(__file__).parent / "data" / "1-LV-rural2--1-sw" / "LoadProfile.csv"
        data_path = data_path.resolve()

        self.ds = CSVDataSource(
            data_path,
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "p", "G1-B_qload": "q"},
            auto_drop=True,
            resample=self.freq,
        )
        
        self.ds2 = ConstantDataSource({"test": 2.5}, self.ds.get_date_range())

        self.test_time = "15.10.2016"

    def test_ConstantForecaster(self):
        dp = DataProvider(self.ds, ConstantForecaster(self.freq, self.horizon))

        expectation = {'q': np.array([0.010906, 0.010906, 0.010906]), 'p': np.array([0.030172, 0.030172, 0.030172])}
        fcst = dp.observe(self.test_time)

        np.testing.assert_equal(expectation, fcst)

        bounds = dp.observation_bounds(self.test_time)
        assert bounds

    def test_PersistenceForecaster(self):
        dp = DataProvider(self.ds, PersistenceForecaster(self.freq, self.horizon))

        expectation = {'q': np.array([0.010906, 0.0, 0.0]), 'p': np.array([0.030172, 0.034483, 0.025862])}
        fcst = dp.observe(self.test_time)

        np.testing.assert_equal(expectation, fcst)

        bounds = dp.observation_bounds(self.test_time)
        assert bounds

    def test_PerfectKnowledgeForecaster(self):
        dp = DataProvider(self.ds, PerfectKnowledgeForecaster(self.freq, self.horizon))

        expectation = {'q': np.array([0.010906, 0.0, 0.021812]), 'p': np.array([0.030172, 0.021552, 0.038793])}
        fcst = dp.observe(self.test_time)

        np.testing.assert_equal(expectation, fcst)

        bounds = dp.observation_bounds(self.test_time)
        assert bounds

    def test_NoisyForecaster(self):
        noise_bounds = [-0.3, 0.1]
        dp = DataProvider(self.ds, NoisyForecaster(self.freq, self.horizon, noise_bounds))

        truth = {'q': np.array([0.010906, 0.0, 0.021812]), 'p': np.array([0.030172, 0.021552, 0.038793])}
        fcst = dp.observe(self.test_time)

        for var in fcst.keys():
            assert (all(fcst[var] - truth[var] >= truth[var] * noise_bounds[0]) and
                    all(fcst[var] - truth[var] <= truth[var] * noise_bounds[1]))

        bounds = dp.observation_bounds(self.test_time)
        assert bounds
        
    def test_ConstantDataSource(self):
        dp = DataProvider(self.ds2, PerfectKnowledgeForecaster(self.freq, self.horizon))

        expectation = {'test': np.array([2.5, 2.5, 2.5])}
        fcst = dp.observe(self.test_time)

        np.testing.assert_equal(expectation, fcst)

        bounds = dp.observation_bounds(self.test_time)
        assert bounds


if __name__ == "__main__":
    unittest.main()
