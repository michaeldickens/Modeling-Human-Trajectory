"""

fit.py
------

Author: Michael Dickens
Created: 2020-06-24

"""

import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize
from scipy.stats import powerlaw
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def read_csv(filename: str) -> Dict[str, List[Optional[float]]]:
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)

        header = None
        csv_data = defaultdict(list)
        for row in reader:
            if header is None:
                header = row
                continue

            for i in range(len(row)):
                csv_data[header[i]].append(float(row[i]) if row[i] else None)

    return csv_data


def trim_missing_rows(xs: List[Optional[float]], ys: List[Optional[float]]) -> Tuple[List[float], List[float]]:
    if len(xs) != len(ys):
        raise Exception("trim_missing_rows: len(xs) != len(ys)")

    zipped = [(x, y) for (x, y) in zip(xs, ys) if x is not None and y is not None]
    return ([t[0] for t in zipped], [t[1] for t in zipped])


def gwp_minimize_mse(xs, ys):
    regression = LinearRegression()
    log_ys = np.log(ys)

    def mse(end_year):
        if end_year[0] - max(xs) <= 0:
            return 99999999
        log_xs = np.log([end_year[0] - x for x in xs]).reshape(-1, 1)
        regression.fit(log_xs, log_ys)
        predicted_ys = regression.predict(log_xs)
        return mean_squared_error(log_ys, predicted_ys)

    return optimize.minimize(mse, [2100])

def fit_gwp(csv_data, cutoff_row=None):
    '''
    Assume GWP follows a power law, culminating at some "doomsday" end year.
    Find the end year that minimizes mean squared error when
    log(end_year - year) and log(GWP) are fitted to a linear function.
    '''
    xs, ys = trim_missing_rows(csv_data['Year'], csv_data['GWP (billion 1990 $)'])
    if cutoff_row:
        xs = xs[:cutoff_row]
        ys = ys[:cutoff_row]

    opt = gwp_minimize_mse(xs, ys)
    end_year = opt.x[0]

    log_xs = np.log([end_year - x for x in xs]).reshape(-1, 1)
    log_ys = np.log(ys)
    regression = LinearRegression()
    regression.fit(log_xs, log_ys)
    predicted_ys = regression.predict(log_xs)

    # for i in range(len(log_xs)):
    #     print("{}\t{}\t{}".format(log_xs[i][0], log_ys[i], predicted_ys[i]))

    # print("Optimal end year:", end_year)
    # print("y = {:.2f} x + {:.2f}".format(regression.coef_[0], regression.intercept_))
    # print("MSE = {}, r^2 = {}".format(mean_squared_error(log_ys, predicted_ys), r2_score(log_ys, predicted_ys)))

    print("{}\t{}\t{}".format(xs[cutoff_row-1], end_year, mean_squared_error(log_ys, predicted_ys)))


csv_data = read_csv('data/GWP.csv')

for cutoff_row in range(42, 101):
    fit_gwp(csv_data, cutoff_row)
