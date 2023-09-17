############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Forecasting: Intermittent Demand
 
# GitHub Repository: <https://github.com/Valdecy>

############################################################################

# Required Libraries
import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np
import pandas as pd

############################################################################

# Function: Plot TS
def plot_int_demand(ts, test = [], size_x = 10, size_y = 10, bar_width = 1, prediction = []):
    plt.figure(figsize = (size_x, size_y))
    plt.xticks(rotation = 90)
    plt.rcParams.update({'font.size': 9}) 
    plt.scatter(ts[ts > 0].index.to_pydatetime(), ts[ts > 0], color = 'k', alpha = 0.7, marker = 's')
    plt.bar(ts[ts > 0].index.to_pydatetime(), ts[ts > 0], width = bar_width, color = 'purple', alpha = 0.3)
    if (len(test) > 0):
      plt.scatter(test[test > 0].index.to_pydatetime(), test[test > 0], color = 'k', alpha = 0.7, marker = 's')
      plt.bar(test[test > 0].index.to_pydatetime(), test[test > 0], width = bar_width, color = 'blue', alpha = 0.3)
    if (len(prediction) > 0):
        plt.scatter(prediction[:ts.shape[0]][prediction > 0].index.to_pydatetime(), prediction[:ts.shape[0]][prediction > 0], c = 'orange')
        plt.plot(prediction[:ts.shape[0]].index.to_pydatetime(), prediction[:ts.shape[0]], c = 'red')
        plt.scatter(prediction[ts.shape[0]:][prediction > 0].index.to_pydatetime(), prediction[ts.shape[0]:][prediction > 0], c = 'green')
    return

# Function: V & Q
def v_q_values(ts):
    v  = ts[ts > 0]
    q  = []
    x1 = 1
    for i in range(0, ts.shape[0]):
        if (ts[i] > 0):
            x2    = i+1
            zeros = (ts[x1:x2] == 0).sum(axis = 0)
            if (zeros >= 0):
                q.append(zeros)
                x1 = x2-1
    return v, np.asarray(q)

# Function: Classification
def classification(ts):
    v, q         = v_q_values(ts)
    adi          = sum(q)/len(v)
    cv_squared   = ( sum( ( (v - ts.mean() )**2)/ len(ts) )/ ts.mean() )
    f_type = 'Smooth'
    if (adi > 1.32 and cv_squared < 0.49 ):
      f_type = 'Intermittent'
    elif (adi > 1.32 and cv_squared > 0.49 ):
      f_type = 'Lumpy'
    elif (adi < 1.32 and cv_squared > 0.49 ):
      f_type = 'Erratic'
    print('ADI: ', round(adi, 3), ', CV: ', round(cv_squared, 3), ', Type: ', f_type)
    return adi, cv_squared

# Function: MASE (Mean Absolute Scaled Error)
def mase(ts, prediction):
    divisor = 0
    for i in range(1, ts.shape[0]):
        divisor = divisor + abs(ts[i] - ts[i-1])
    divisor = divisor/(ts.shape[0] - 1)
    diff    = abs(ts - prediction[:ts.shape[0]])/divisor
    mase    = diff.mean()
    return mase

# Function: RMSE (Root Mean Squared Error)
def rmse(ts, prediction):
    diff = (ts - prediction[:ts.shape[0]])**2
    mse  = diff.mean()
    return mse**(1/2)

############################################################################

# Function: Croston Method ( https://doi.org/10.2307/3007885 )
def croston_method(ts, alpha = 0.1, n_steps = 1, freq = '1d'):
    v, q          = v_q_values(ts)
    v_i           = ts.copy(deep = True)
    q_i           = ts.copy(deep = True)
    f_i           = ts.copy(deep = True)
    v_i.values[:] = 0
    q_i.values[:] = 0
    f_i.values[:] = 0
    date_idx      = ts.index  
    v_i[0]        = ts[0]  
    q_i[0]        = 1 
    f_i[0]        = v_i[0]/q_i[0]
    for i in range(0, ts.shape[0]-1):
        if (ts[i] > 0):
            idx_1        = ts.index.get_loc(date_idx[i])
            idx_2        = v.index.get_loc (date_idx[i])
            v_i[idx_1+1] = alpha*v[idx_2] + (1 - alpha)*v_i[idx_1]
            q_i[idx_1+1] = int(alpha*q[idx_2] + (1 - alpha)*q_i[idx_1])
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = v_i[idx_1+1]/q_i[idx_1+1]
            else:
                f_i[idx_1+1] = v_i[idx_1]
        else:
            idx_1        = ts.index.get_loc(date_idx[i])
            v_i[idx_1+1] = v_i[idx_1]
            q_i[idx_1+1] = q_i[idx_1]
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = v_i[idx_1+1]/q_i[idx_1+1]
            else:
                f_i[idx_1+1] = v_i[idx_1]
    idx        = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    new_series = pd.Series(np.repeat(f_i[-1], len(idx)), index = idx)
    f_i        = pd.concat([f_i, new_series]) 
    return v_i, q_i, f_i

# Function: SBA (Syntetos & Boylan Approximation) Method (  https://doi.org/10.1016/j.ijforecast.2004.10.001 )
def sba_method(ts, alpha = 0.1, n_steps = 1, freq = '1d'):
    v, q          = v_q_values(ts)
    v_i           = ts.copy(deep = True)
    q_i           = ts.copy(deep = True)
    f_i           = ts.copy(deep = True)
    v_i.values[:] = 0
    q_i.values[:] = 0
    f_i.values[:] = 0
    date_idx      = ts.index  
    v_i[0]        = ts[0]  
    q_i[0]        = 1 
    f_i[0]        = v_i[0]/q_i[0]
    for i in range(0, ts.shape[0]-1):
        if (ts[i] > 0):
            idx_1        = ts.index.get_loc(date_idx[i])
            idx_2        = v.index.get_loc(date_idx[i])
            v_i[idx_1+1] = alpha*v[idx_2] + (1 - alpha)*v_i[idx_1]
            q_i[idx_1+1] = int(alpha*q[idx_2] + (1 - alpha)*q_i[idx_1])
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = (1 - alpha/(2))*(v_i[idx_1+1]/q_i[idx_1+1])
            else:
                f_i[idx_1+1] = (1 - alpha/(2))*v_i[idx_1]
        else:
            idx_1        = ts.index.get_loc(date_idx[i])
            v_i[idx_1+1] = v_i[idx_1]
            q_i[idx_1+1] = q_i[idx_1]
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = (1 - alpha/(2))*v_i[idx_1+1]/q_i[idx_1+1]
            else:
                f_i[idx_1+1] = (1 - alpha/(2))*v_i[idx_1]
    idx        = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    new_series = pd.Series(np.repeat(f_i[-1], len(idx)), index = idx)
    f_i        = pd.concat([f_i, new_series])    
    return v_i, q_i, f_i

# Function: SBJ (Shale, Boylan & Johnston) Method ( https://doi.org/10.1057/palgrave.jors.2602031 )
def sbj_method(ts, alpha = 0.1, n_steps = 1, freq = '1d'):
    v, q          = v_q_values(ts)
    v_i           = ts.copy(deep = True)
    q_i           = ts.copy(deep = True)
    f_i           = ts.copy(deep = True)
    v_i.values[:] = 0
    q_i.values[:] = 0
    f_i.values[:] = 0
    date_idx      = ts.index  
    v_i[0]        = ts[0]  
    q_i[0]        = 1 
    f_i[0]        = v_i[0]/q_i[0]
    for i in range(0, ts.shape[0]-1):
        if (ts[i] > 0):
            idx_1        = ts.index.get_loc(date_idx[i])
            idx_2        = v.index.get_loc(date_idx[i])
            v_i[idx_1+1] = alpha*v[idx_2] + (1 - alpha)*v_i[idx_1]
            q_i[idx_1+1] = int(alpha*q[idx_2] + (1 - alpha)*q_i[idx_1])
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = (1 - alpha/(2 - alpha))*(v_i[idx_1+1]/q_i[idx_1+1])
            else:
                f_i[idx_1+1] = (1 - alpha/(2 - alpha))*v_i[idx_1]
        else:
            idx_1        = ts.index.get_loc(date_idx[i])
            v_i[idx_1+1] = v_i[idx_1]
            q_i[idx_1+1] = q_i[idx_1]
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = (1 - alpha/(2 - alpha))*v_i[idx_1+1]/q_i[idx_1+1]
            else:
                f_i[idx_1+1] = (1 - alpha/(2 - alpha))*v_i[idx_1]
    idx        = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    new_series = pd.Series(np.repeat(f_i[-1], len(idx)), index = idx)
    f_i        = pd.concat([f_i, new_series])    
    return v_i, q_i, f_i

# Function: TSB (Teunter, Syntetos & Babai) Method ( https://doi.org/10.1016/j.ejor.2011.05.018 )
def tsb_method(ts, alpha = 0.1, beta = 0.1, n_steps = 1, freq = '1d'):
    v, q          = v_q_values(ts)
    v_i           = ts.copy(deep = True)
    q_i           = ts.copy(deep = True)
    f_i           = ts.copy(deep = True)
    v_i.values[:] = 0
    q_i.values[:] = 0
    f_i.values[:] = 0
    date_idx      = ts.index  
    v_i[0]        = ts[0]    
    q_i[0]        = 1 
    f_i[0]        = v_i[0]*q_i[0]
    for i in range(0, ts.shape[0]-1):
        if (ts[i] > 0):
            idx_1        = ts.index.get_loc(date_idx[i])
            idx_2        = v.index.get_loc(date_idx[i])
            v_i[idx_1+1] = alpha*v[idx_2] + (1 - alpha)*v_i[idx_1]
            q_i[idx_1+1] = beta + (1 - beta)*int(q_i[idx_1])
            f_i[idx_1+1] = v_i[idx_1+1]*q_i[idx_1+1]
        else:
            idx_1        = ts.index.get_loc(date_idx[i])
            v_i[idx_1+1] = v_i[idx_1]
            q_i[idx_1+1] = (1 - beta)*q_i[idx_1]
            f_i[idx_1+1] = v_i[idx_1+1]*q_i[idx_1+1]
    idx        = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    new_series = pd.Series(np.repeat(f_i[-1], len(idx)), index = idx)
    f_i        = pd.concat([f_i, new_series])  
    return v_i, q_i, f_i

# Function: HES (Prestwich et al. 2014) Method ( https://doi.org/10.1016/j.ijforecast.2014.01.006 )
def hes_method(ts, alpha = 0.1, n_steps = 1, freq = '1d'):
    v, q          = v_q_values(ts)
    v_i           = ts.copy(deep = True)
    q_i           = ts.copy(deep = True)
    f_i           = ts.copy(deep = True)
    v_i.values[:] = 0
    q_i.values[:] = 0
    f_i.values[:] = 0
    date_idx      = ts.index  
    v_i[0]        = ts[0]  
    q_i[0]        = 1 
    f_i[0]        = v_i[0]/q_i[0]
    for i in range(0, ts.shape[0]-1):
        if (ts[i] > 0):
            idx_1        = ts.index.get_loc(date_idx[i])
            idx_2        = v.index.get_loc(date_idx[i])
            v_i[idx_1+1] = alpha*v[idx_2] + (1 - alpha)*v_i[idx_1]
            q_i[idx_1+1] = int(alpha*q[idx_2] + (1 - alpha)*q_i[idx_1])
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = v_i[idx_1+1]/q_i[idx_1+1]
            else:
                f_i[idx_1+1] = v_i[idx_1]
        else:
            idx_1        = ts.index.get_loc(date_idx[i])
            v_i[idx_1+1] = v_i[idx_1]
            q_i[idx_1+1] = q_i[idx_1]
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = v_i[idx_1+1]/(q_i[idx_1+1] + alpha*q[idx_2]/2)
            else:
                f_i[idx_1+1] = v_i[idx_1]
    idx = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    f_i = f_i.append(pd.Series(np.repeat(f_i[-1], len(idx)), index = idx))    
    return v_i, q_i, f_i

# Function: LES (Linear-Exponential Smoothing) Method ( https://doi.org/10.1016/j.ijforecast.2020.08.010 )
def les_method(ts, alpha = 0.1, n_steps = 1, freq = '1d'):
    v, q          = v_q_values(ts)
    v_i           = ts.copy(deep = True)
    q_i           = ts.copy(deep = True)
    f_i           = ts.copy(deep = True)
    v_i.values[:] = 0
    q_i.values[:] = 0
    f_i.values[:] = 0
    date_idx      = ts.index  
    v_i[0]        = ts[0]  
    q_i[0]        = 1 
    f_i[0]        = v_i[0]/q_i[0]
    idx_1         = 0
    idx_2         = 0
    for i in range(0, ts.shape[0]-1):
        if (ts[i] > 0):
            idx_1        = ts.index.get_loc(date_idx[i])
            idx_2        = v.index.get_loc(date_idx[i])
            v_i[idx_1+1] = alpha*v[idx_2] + (1 - alpha)*v_i[idx_1]
            q_i[idx_1+1] = int(alpha*q[idx_2] + (1 - alpha)*q_i[idx_1])
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = v_i[idx_1+1]/q_i[idx_1+1]
            else:
                f_i[idx_1+1] = v_i[idx_1]
        else:
            idx_1        = ts.index.get_loc(date_idx[i])
            v_i[idx_1+1] = v_i[idx_1]
            q_i[idx_1+1] = q_i[idx_1]
            if (q_i[idx_1+1] != 0):
                f_i[idx_1+1] = (v_i[idx_1+1]/q_i[idx_1+1]) * (1 -  alpha*q[idx_2]/(2*q_i[idx_1+1]))
            else:
                f_i[idx_1+1] = v_i[idx_1]
    idx = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    f_i = f_i.append(pd.Series(np.repeat(f_i[-1], len(idx)), index = idx))    
    return v_i, q_i, f_i

# Function: SES (Simple Exponential Smoothing) Method ( https://www.industrydocuments.ucsf.edu/tobacco/docs/#id=jzlc0130 )
def ses_method(ts, alpha = 0.1, n_steps = 1, freq = '1d'):
    f_i           = ts.copy(deep = True)
    date_idx      = ts.index 
    f_i.values[:] = 0
    f_i[0]        = ts[0]
    for i in range(0, ts.shape[0]-1):
        idx        = ts.index.get_loc(date_idx[i])
        f_i[idx+1] =  alpha*ts[idx] +  (1 -  alpha)*(f_i[idx])
    idx = pd.date_range(f_i.index[-1], periods = n_steps, freq = freq)[1:]
    f_i = f_i.append(pd.Series(np.repeat(f_i[-1], len(idx)), index = idx))  
    return f_i

############################################################################