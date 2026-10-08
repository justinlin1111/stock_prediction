# In[0]
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 下載數據
# ticker為股名 
ticker = '00878.TW'
# start、end表示開始日期與結束日期
start = '2022-01-01'
end = '2024-06-01'

data = yf.download(ticker, start, end)

# 選擇OHLCV列
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# In[1]
# 添加 wantToBuy 列
data['wantToBuy'] = 0
# 從第8筆資料後才有前7天資料
for i in range(8, len(data)):
    # 利用過去8天(不包含當天)的資料去判斷下一天是不是想要買的(1.05表示漲了5%)
    if data['Close'].iloc[i-1] > data['Close'].iloc[i-8] * 1.05:
        data.at[data.index[i], 'wantToBuy'] = 1

# 計算歷史波動性
data['volatility'] = data['Close'].rolling(window=7).std()
data.fillna(0, inplace=True)

print(data.head(15))

# In[2]
# 把數據標準化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 創建訓練集與測試集的函數
def create_dataset(dataset, time_step=30):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :-1])  
        Y.append(dataset[i + time_step, -1])   
    return np.array(X), np.array(Y)

# 設定時間間隔
time_step = int(len(scaled_data)/10)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print(X_test)

# 把模型變成LSTM所期望的格式，即:(樣本數、時間間隔、特徵數量)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# In[3]
# 建立LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1, activation='sigmoid'))  # 使用sigmoid activate函數預測0/1

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型，
# batch_size表示一次迭代利用幾個數據，batch_size越大，訓練速度越快
# epochs表示訓練的次數
# validation_split表示拿出20%的資料當作驗證集，避免overfitting
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# In[4]
# 預測準確率
# 將資料依照0.5為界線切割成0或1
y_train_pred = (y_train > 0.5).astype(int)
y_test_pred = (y_test > 0.5).astype(int)

# 注意訓練集跟測試集的長度不同
train_accuracy = accuracy_score(data['wantToBuy'][:len(y_train_pred)], y_train_pred)
test_accuracy = accuracy_score(data['wantToBuy'][-len(y_test_pred):], y_test_pred)
print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# In[5]
# 預測整個數據集的 wantToBuy
full_data = scaled_data
X_full, _ = create_dataset(full_data, time_step)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], X_full.shape[2])

full_pred = (model.predict(X_full) > 0.5).astype("int32")

# 輸出最後一次的 wantToBuy 结果
last_want_to_buy = full_pred[-1]
print(f'今天的 wantToBuy 结果: {last_want_to_buy[0]}')
print("="*40)

# 計算過去90天的波動率做比較
historical_volatility = (data['Close'].rolling(window=90).std() / np.sqrt(90)).iloc[-1]
print(f"過去90天的波動率為: {historical_volatility}")
print(f"波動率為: {data['volatility'].iloc[-1]}")
print("="*40)
if historical_volatility > data['volatility'].iloc[-1]:
    print("今天的波動率比90天以來的平均要低")
else:
    print("今天的波動率比90天以來的平均要高")

# 繪製收盤價
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price')
plt.title(ticker + ' Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# %%