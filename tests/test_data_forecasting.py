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
            aggregation_method='first',
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

    def test_ArrayDataSource(self):
        looping_data = ArrayDataSource({
            "day_night": [0.] * 6 + [1.] * 12 + [0.] * 6,
            "weekends": np.array([0.] * 24 * 5 + [1.] * 24 * 2),
            "prime_chaos": [0, 42, 73, 4, 5, 6, 7],
        },
            date_range=[pd.to_datetime('2024-01-08'), pd.to_datetime('2024-12-31')],
            frequency=timedelta(hours=1),
            start_date=pd.to_datetime('2024-01-01')  # 2024 starts with a Monday
        )

        expectation = np.array([
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 42, 73, 4, 5, 6, 7, 0, 42, 73, 4, 5, 6, 7, 0, 42, 73, 4, 5, 6, 7, 0, 42, 73],
        ]).T

        data = looping_data(pd.to_datetime('2024-01-08'), pd.to_datetime('2024-01-08 23:00:00'))
        np.testing.assert_equal(expectation, data)

        expectation = np.array([
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5, 6, 7, 0, 42, 73, 4, 5, 6, 7, 0, 42, 73, 4, 5, 6, 7, 0, 42, 73, 4, 5, 6, 7],
        ]).T

        data = looping_data(pd.to_datetime('2024-01-14'), pd.to_datetime('2024-01-14 23:00:00'))
        np.testing.assert_equal(expectation, data)


if __name__ == "__main__":
    unittest.main()
