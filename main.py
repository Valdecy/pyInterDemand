############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: Forecasting - Intermittent Demand
 
# GitHub Repository: <https://github.com/Valdecy>

############################################################################

# Required Libraries
import pandas as pd

from src.intermittent import plot_int_demand, classification, mase, rmse
from src.intermittent import croston_method
from src.intermittent import sba_method
from src.intermittent import sbj_method
from src.intermittent import tsb_method
from src.intermittent import hes_method
from src.intermittent import les_method
from src.intermittent import ses_method

############################################################################

# Load Dataset
data = {
        'DATE': pd.Series(['21/08/2020','22/08/2020', '23/08/2020', '24/08/2020', '25/08/2020', '26/08/2020', '27/08/2020', '28/08/2020', '29/08/2020', '30/08/2020', '31/08/2020', '01/09/2020']),
       'Value': pd.Series([5, 10, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0]),
       }
dataset = pd.DataFrame(data)
dataset['DATE'] = pd.to_datetime(dataset['DATE']).map(lambda x: x.strftime('%d-%m-%Y'))

############################################################################

# Prepare Time Series TS
ts       = dataset['Value'].copy(deep = True)
ts.index = pd.DatetimeIndex(dataset['DATE'])
ts       = ts.sort_index()
ts       = ts.reindex(pd.date_range(ts.index.min(), ts.index.max()), fill_value = 0)
ts       = ts.loc[ts[(ts != 0)].first_valid_index():]

print('')
print('Total Number of Observations: ', ts.shape[0])
print('Total Number of Zeros: ', len(ts[ts == 0]))
print('Start Date: ', ts.index[0])
print('End Date: '  , ts.index[-1])
print('')

############################################################################

# Time Series Classification
adi, cv_sq = classification(ts)

############################################################################

# Time Series Plot
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3)

############################################################################

# Croston
v, q, forecast = croston_method(ts, alpha = 0.5, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################

# SBA
v, q, forecast = sba_method(ts, alpha = 0.5, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################

# SBJ
v, q, forecast = sbj_method(ts, alpha = 0.2, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################

# TSB
v, q, forecast = tsb_method(ts, alpha = 0.5, beta = 0, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################

# HES
v, q, forecast = hes_method(ts, alpha = 0.5, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################

# LES
v, q, forecast = les_method(ts, alpha = 0.5, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################

# SES
v, q, forecast = ses_method(ts, alpha = 0.5, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

############################################################################


