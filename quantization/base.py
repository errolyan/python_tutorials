# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import numpy as  np

# Cumulative win, loss, avg win, avg loss
def win_rate(ret_1d):
    hit_rate = np.add.accumulate(ret_1d >= 0) / np.add.accumulate((ret_1d >= 0) | (ret_1d <= 0))
    return hit_rate


def loss_rate(ret_1d):
    miss_rate = np.add.accumulate(ret_1d <= 0) / np.add.accumulate((ret_1d >= 0) | (ret_1d <= 0))
    return miss_rate


def avg_win(ret_1d):
    average_win = np.add.accumulate(ret_1d[ret_1d >= 0]) / np.add.accumulate(ret_1d >= 0)
    average_win.fillna(method='ffill', inplace=True)
    return average_win


def avg_loss(ret_1d):
    average_loss = np.add.accumulate(ret_1d[ret_1d <= 0]) / np.add.accumulate(ret_1d <= 0)
    average_loss.fillna(method='ffill', inplace=True)
    return average_loss


# rolling win, loss, avg win, avg loss
def rolling_loss_rate(returns, window):
    bad_days = returns.copy()
    bad_days[bad_days > 0] = np.nan
    bad_days_rolling = (bad_days.rolling(window).count() / window)
    return bad_days_rolling


def rolling_avg_loss(returns, window):
    avg_bad_day = returns.copy()
    avg_bad_day[avg_bad_day > 0] = 0
    _avg_bad_day = (avg_bad_day.rolling(window).sum() / window)
    return _avg_bad_day


def rolling_win_rate(returns, window):
    good_days = returns.copy()
    good_days[good_days < 0] = np.nan
    good_days_rolling = (good_days.rolling(window).count() / window)
    return good_days_rolling


def rolling_avg_win(returns, window):
    avg_good_day = returns.copy()
    avg_good_day[avg_good_day < 0] = 0
    _avg_good_day = (avg_good_day.rolling(window).sum() / window)
    return _avg_good_day


# Gain expectancies and Kelly criterion
def arige(win_rate, avg_win, avg_loss):  # win% * avg_win% - loss% * abs(avg_loss%)
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def george(win_rate, avg_win, avg_loss):  # (1+ avg_win%)** win% * (1- abs(avg_loss%)) ** loss%  -1
    return (1 + avg_win) ** win_rate * (1 + avg_loss) ** (1 - win_rate) - 1


def kelly(win_rate, avg_win, avg_loss):  # Kelly = win% / abs(avg_loss%) - loss% / avg_win%
    return win_rate / np.abs(avg_loss) - (1 - win_rate) / avg_win

