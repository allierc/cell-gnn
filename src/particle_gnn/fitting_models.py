import numpy as np
from scipy.optimize import curve_fit


def linear_model(x, a, b):
    return a * x + b


def power_model(x, a, b):
    return a / (x ** b)


def linear_fit(x_data=[], y_data=[], threshold=10):

    if threshold == -1:
        x_data_ = x_data
        y_data_ = y_data
        relative_error = -1
        not_outliers = -1
    else:
        relative_error = np.abs(y_data - x_data) / np.abs(x_data)
        not_outliers = np.argwhere(relative_error < threshold)
        x_data_ = x_data[not_outliers[:, 0]]
        y_data_ = y_data[not_outliers[:, 0]]

    lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
    residuals = y_data_ - linear_model(x_data_, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return lin_fit, r_squared, relative_error, not_outliers, x_data, y_data


def boids_model(x, a, b, c):
    xdiff = x[:, 0:2]
    vdiff = x[:, 2:4]
    r = np.concatenate((x[:, 4:5], x[:, 4:5]), axis=1)

    total = a * xdiff + b * vdiff - c * xdiff / r
    total = np.sqrt(total[:, 0] ** 2 + total[:, 1] ** 2)

    return total
