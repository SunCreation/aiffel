#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')

print('슝=3')
# %%
dataset_filepath = 'data/daily-min-temperatures.csv' 
df = pd.read_csv(dataset_filepath) 
print(type(df))
df.head()
# %%
# 이번에는 Date를 index_col로 지정해 주었습니다. 
df = pd.read_csv(dataset_filepath, index_col='Date', parse_dates=True)
print(type(df))
df.head()
# %%
ts1 = df['Temp']  # 우선은 데이터 확인용이니 time series 의 이니셜을 따서 'ts'라고 이름 붙여줍시다!
print(type(ts1))
ts1.head(20)
# %%
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 13, 6    # matlab 차트의 기본 크기를 13, 6으로 지정해 줍니다.

# 시계열(time series) 데이터를 차트로 그려 봅시다. 특별히 더 가공하지 않아도 잘 그려집니다.
plt.plot(ts1)
# %%
ts1[ts1.isna()]  # 시계열(Time Series)에서 결측치가 있는 부분만 Series로 출력합니다.
# %%
# 결측치가 있다면 이를 보간합니다. 보간 기준은 time을 선택합니다. 
ts1=ts1.interpolate(method='time')

# 보간 이후 결측치(NaN) 유무를 다시 확인합니다.
print(ts1[ts1.isna()])

# 다시 그래프를 확인해봅시다!
plt.plot(ts1)
# %%
def plot_rolling_statistics(timeseries, window=12):
    
    rolmean = timeseries.rolling(window=window).mean()  # 이동평균 시계열
    rolstd = timeseries.rolling(window=window).std()    # 이동표준편차 시계열

     # 원본시계열, 이동평균, 이동표준편차를 plot으로 시각화해 본다.
    orig = plt.plot(timeseries, color='blue',label='Original')    
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
print('슝=3')
# %%
plot_rolling_statistics(ts1, window=12)
# %%
dataset_filepath = 'data/airline-passengers.csv' 
df = pd.read_csv(dataset_filepath, index_col='Month', parse_dates=True).fillna(0)  
print(type(df))
df.head()
# %%
ts2 = df['Passengers']
plt.plot(ts2)
# %%
plot_rolling_statistics(ts2, window=12)
# %%
from statsmodels.tsa.stattools import adfuller

def augmented_dickey_fuller_test(timeseries):
    # statsmodels 패키지에서 제공하는 adfuller 메소드를 호출합니다.
    dftest = adfuller(timeseries, autolag='AIC')  
    
    # adfuller 메소드가 리턴한 결과를 정리하여 출력합니다.
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    
print('슝=3')
# %%
augmented_dickey_fuller_test(ts1)
# %%
adfuller(ts1)
# %%
augmented_dickey_fuller_test(ts2)
# %%
ts_log = np.log(ts2)
plt.plot(ts_log)
# %%
augmented_dickey_fuller_test(ts_log)
# %%
# 계절성 제거
moving_avg = ts_log.rolling(window=12).mean()  # moving average구하기 
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
# %%
ts_log_moving_avg = ts_log - moving_avg # 변화량 제거
ts_log_moving_avg.head(15)
# %%
ts_log_moving_avg.dropna(inplace=True)
ts_log_moving_avg.head(15)
# %%
plot_rolling_statistics(ts_log_moving_avg)
# %%
augmented_dickey_fuller_test(ts_log_moving_avg)
# %%
moving_avg_6 = ts_log.rolling(window=6).mean()
ts_log_moving_avg_6 = ts_log - moving_avg_6
ts_log_moving_avg_6.dropna(inplace=True)
print('슝=3')
# %%
plot_rolling_statistics(ts_log_moving_avg_6)
# %%
augmented_dickey_fuller_test(ts_log_moving_avg_6)
# %%
# 차분
ts_log_moving_avg_shift = ts_log_moving_avg.shift()

plt.plot(ts_log_moving_avg, color='blue')
plt.plot(ts_log_moving_avg_shift, color='green')
# %%
ts_log_moving_avg_diff = ts_log_moving_avg - ts_log_moving_avg_shift
ts_log_moving_avg_diff.dropna(inplace=True)
plt.plot(ts_log_moving_avg_diff)
# %%
plot_rolling_statistics(ts_log_moving_avg_diff)
# %%
augmented_dickey_fuller_test(ts_log_moving_avg_diff)
# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend # 추세(시간 추이에 따라 나타나는 평균값 변화 )
seasonal = decomposition.seasonal # 계절성(패턴이 파악되지 않은 주기적 변화)
residual = decomposition.resid # 원본(로그변환한) - 추세 - 계절성

plt.rcParams["figure.figsize"] = (11,6)
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
# %%
plt.rcParams["figure.figsize"] = (13,6)
plot_rolling_statistics(residual)
# %%
residual.dropna(inplace=True)
augmented_dickey_fuller_test(residual)
# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
plt.show()
# p,q,d 값을 정수로만 하는 건 아닐텐데;;
# %%
# 1차 차분 구하기
diff_1 = ts_log.diff(periods=1).iloc[1:]
diff_1.plot(title='Difference 1st')

augmented_dickey_fuller_test(diff_1)
# %%
train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(ts_log, c='r', label='training dataset')  # train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택
plt.plot(test_data, c='b', label='test dataset')
plt.legend()
# %%
print(ts_log[:2])
print(train_data.shape)
print(test_data.shape)
# %%
# p = 1, d = 1, q = 0으로 ARIMA모델 사용
import warnings
warnings.filterwarnings('ignore') #경고 무시

from statsmodels.tsa.arima.model import ARIMA
# Build Model
model = ARIMA(train_data, order=(1, 1, 0))  
fitted_m = model.fit() 

print(fitted_m.summary())
# %%
fitted_m = fitted_m.predict()
fitted_m = fitted_m.drop(fitted_m.index[0])
plt.plot(fitted_m, label='predict')
plt.plot(train_data, label='train_data')
plt.legend()
# %%
model = ARIMA(train_data, order=(10, 1, 0))  # p값을 10으로 테스트
fitted_m = model.fit() 
fc= fitted_m.forecast(len(test_data), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)   # 예측결과

# Plot
plt.figure(figsize=(9,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, c='b', label='actual price')
plt.plot(fc_series, c='r',label='predicted price')
plt.legend()
plt.show()
# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse = mean_squared_error(np.exp(test_data), np.exp(fc))
print('MSE: ', mse)

mae = mean_absolute_error(np.exp(test_data), np.exp(fc))
print('MAE: ', mae)

rmse = math.sqrt(mean_squared_error(np.exp(test_data), np.exp(fc)))
print('RMSE: ', rmse)

mape = np.mean(np.abs(np.exp(fc) - np.exp(test_data))/np.abs(np.exp(test_data)))
print('MAPE: {:.2f}%'.format(mape*100))
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Time Series 생성
# [[YOUR CODE]]

ts.head()

#%%
# 결측치 처리
# [[YOUR CODE]]
ts[ts.isna()]  # Time Series에서 결측치가 있는 부분만 Series로 출력합니다.

#%%

# 로그 변환
# [[YOUR CODE]]

#%%

# 정성적 그래프 분석
plot_rolling_statistics(ts_log, window=12)

#정량적 Augmented Dicky-Fuller Test
augmented_dickey_fuller_test(ts_log)

#시계열 분해 (Time Series Decomposition)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log, model='multiplicative', period = 30) 
# [[YOUR CODE]]

#%%

# Residual 안정성 확인
# [[YOUR CODE]]

#%%

train_data, test_data = # [[YOUR CODE]]

#%%

# ACF, PACF 그려보기 -> p,q 구하기
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# [[YOUR CODE]]

#%%

# 차분 안정성 확인 -> d 구하기
# [[YOUR CODE]]

augmented_dickey_fuller_test(diff_1)

#%%

from statsmodels.tsa.arima_model import ARIMA

# Build and Train  Model
# [[YOUR CODE]]

#%%

# Forecast : 결과가 fc에 담깁니다. 
# [[YOUR CODE]]

# Make as pandas series
# [[YOUR CODE]]

# Plot
# [[YOUR CODE]]

#%%

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse = # [[YOUR CODE]]
print('MSE: ', mse)

mae = # [[YOUR CODE]]
print('MAE: ', mae)

rmse = # [[YOUR CODE]]
print('RMSE: ', rmse)

mape = # [[YOUR CODE]]
print('MAPE: {:.2f}%'.format(mape*100))


#%%

# [[YOUR CODE]]

#%%

class bong_abbang:
    def __init__(N) -> None:
        hi = N
    
    
    def printhi(hi):
        print("hi",hi)

hi = bong_abbang()
hi.printhi()    

# class bread:
#%%
hi
# %%
print(hi)