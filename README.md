# pyInterDemand - Intermittent Demand Library

Demand forecasting is a critical component of supply chain management and business operations. While traditional demand forecasting methods are geared towards continuous and stable demand patterns, intermittent demand characterized by irregular or sporadic purchase events presents a unique set of challenges. pyInterDemand is a Python library designed to address these challenges by offering a comprehensive suite of algorithms tailored for intermittent demand forecasting. 

The supported algorithms are:

**Croston** - Croston's method separates the intermittent demand data into two separate sequences, one for the non-zero demand and another for the intervals between non-zero demands. The method then applies separate exponential smoothing on both.

**SBA (Syntetos & Boylan Approximation)** - This approach extends Croston's method by adjusting the smoothing parameter based on the bias in the forecast error.

**SBJ (Syntetos, Boylan & Johnston)** - An evolution of the SBA method, SBJ introduces an additional parameter to optimize the estimation further.

**TSB (Teunter, Syntetos & Babai)** - TSB offers a modification of Croston's method to improve forecast accuracy by dynamically updating the smoothing parameter.

**HES (Prestwich et al. 2014)** - This is a generalized exponential smoothing technique adapted for intermittent demand scenarios.

**LES (Linear Exponential Smoothing)** - LES employs a linear function to model the demand, smoothing the data points over time.

**SES (Simple Exponential Smoothing)** - The most straightforward among the techniques, SES applies an exponential decay to past observations.


## Usage

1. Install
```bash
pip install pyInterDemand
```
2. Import

```py3

# Import
from pyInterDemand.algorithm.intermittent import plot_int_demand, classification, mase, rmse
from pyInterDemand.algorithm.intermittent import croston_method

# Load Dataset
data = {
        'DATE': pd.Series(['21/08/2020','22/08/2020', '23/08/2020', '24/08/2020', '25/08/2020', '26/08/2020', '27/08/2020', '28/08/2020', '29/08/2020', '30/08/2020', '31/08/2020', '01/09/2020']),
       'Value': pd.Series([5, 10, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0]),
       }
dataset         = pd.DataFrame(data)
dataset['DATE'] = pd.to_datetime(dataset['DATE'], dayfirst = True).map(lambda x: x.strftime('%d-%m-%Y'))

# Prepare Time Series TS
ts       = dataset['Value'].copy(deep = True)
ts.index = pd.DatetimeIndex(dataset['DATE'], dayfirst = True)
ts       = ts.sort_index()
ts       = ts.reindex(pd.date_range(ts.index.min(), ts.index.max()), fill_value = 0)
ts       = ts.loc[ts[(ts != 0)].first_valid_index():]
print('')
print('Total Number of Observations: ', ts.shape[0])
print('Total Number of Zeros: ', len(ts[ts == 0]))
print('Start Date: ', ts.index[0])
print('End Date: '  , ts.index[-1])
print('')

# Time Series Classification
adi, cv_sq = classification(ts)

# Time Series Plot
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3)

# Croston
v, q, forecast = croston_method(ts, alpha = 0.5, n_steps = 4)
plot_int_demand(ts, size_x = 15, size_y = 10, bar_width = 0.3, prediction = forecast)

# Error
print('MASE = ', round(mase(ts, forecast), 3), ', RMSE = ', round(rmse(ts, forecast), 3))

```

3. Try it in **Colab**: 

- Croston ([ Colab Demo ](https://colab.research.google.com/drive/199MNV5EfwOaCYw3mWuVFk5wT-VUDGb8C?usp=sharing)) ( [ Paper ](https://doi.org/10.2307/3007885))
- SBA ([ Colab Demo ](https://colab.research.google.com/drive/1Ny4poBoZiq9sQYL5DbBkMICaWn1s_pD8?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.ijforecast.2004.10.001))
- SBJ ([ Colab Demo ](https://colab.research.google.com/drive/1dfUXTHqwXG8sJlJypZw1wt-jZ5fxfPuZ?usp=sharing)) ( [ Paper ](https://doi.org/10.1057/palgrave.jors.2602031))
- TSB ([ Colab Demo ](https://colab.research.google.com/drive/1P8txq1ET8bR5Frx3oiwQzMVteHUXuMea?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.ejor.2011.05.018))
- HES ([ Colab Demo ](https://colab.research.google.com/drive/1MDHobeHn6gWwCLVztrKoszDqpG4F3qxX?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.ijforecast.2014.01.006))
- LES ([ Colab Demo ](https://colab.research.google.com/drive/1LEGoN_kc03_hQp04Lv0Vce7dTUMCL7p6?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.ijforecast.2020.08.010))
- SES ([ Colab Demo ](https://colab.research.google.com/drive/1avarbQqMrVTDBR4M5A07JELRN1JgsMzp?usp=sharing)) ( [ Paper ](https://www.industrydocuments.ucsf.edu/tobacco/docs/#id=jzlc0130))
