import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
import os

# ---------------------------
# 1. Đọc dữ liệu
# ---------------------------
df = pd.read_csv('data.csv')
df = df.drop(columns=["KL", "% Thay đổi"])
df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')

# Chuyển đổi các cột giá sang float
for col in ['Lần cuối','Mở','Cao','Thấp']:
    df[col] = df[col].str.replace(',', '').astype(float)

# Sắp xếp theo ngày
df = df.sort_values(by='Ngày')
df1 = df[['Ngày','Lần cuối']].copy()
df1.index = df1['Ngày']
df1.drop('Ngày', axis=1, inplace=True)

# ---------------------------
# 2. Chuẩn hóa dữ liệu
# ---------------------------
data = df1.values
sc = MinMaxScaler(feature_range=(0,1))
sc_data = sc.fit_transform(data)

# ---------------------------
# 3. Chia train/test
# ---------------------------
train_size = int(len(data) * 0.8)
train_data = sc_data[:train_size]
test_data = sc_data[train_size-50:]  # lấy thêm 50 giá để làm window

# Tạo dữ liệu cho LSTM
def create_dataset(dataset, look_back=50):
    x, y = [], []
    for i in range(look_back, len(dataset)):
        x.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0],1))
    return x, y

x_train, y_train = create_dataset(train_data)
x_test, y_test = create_dataset(test_data) if len(test_data) > 50 else (np.array([]), np.array([]))

# ---------------------------
# 4. Kiểm tra model đã có sẵn
# ---------------------------
save_model_path = "save_model.keras"
if os.path.exists(save_model_path):
    model = load_model(save_model_path)
    print("Đã load mô hình từ save_model.keras")
else:
    # Xây dựng model mới
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1],1), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # Huấn luyện
    checkpoint = ModelCheckpoint(save_model_path, monitor='loss', save_best_only=True, verbose=2)
    model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[checkpoint])
    model = load_model(save_model_path)

# ---------------------------
# 5. Dự đoán và đảo chuẩn hóa
# ---------------------------
y_train_pred = model.predict(x_train)
y_train_pred = sc.inverse_transform(y_train_pred)
y_train_true = sc.inverse_transform(y_train)

if x_test.size > 0:
    y_test_pred = model.predict(x_test)
    y_test_pred = sc.inverse_transform(y_test_pred)
    y_test_true = sc.inverse_transform(y_test)
else:
    y_test_pred = np.array([])
    y_test_true = np.array([])

# ---------------------------
# 6. Vẽ biểu đồ
# ---------------------------
plt.figure(figsize=(18,6))
plt.plot(df1.index, df1['Lần cuối'], label='Giá thực tế', color='red')

train_index = df1.index[50:train_size]
plt.plot(train_index, y_train_pred, label='Dự đoán train', color='green')

if len(y_test_pred) > 0:
    test_index = df1.index[train_size:]
    plt.plot(test_index, y_test_pred, label='Dự đoán test', color='blue')

plt.title('So sánh giá dự đoán và giá thực tế')
plt.xlabel('Thời gian')
plt.ylabel('Giá Lần cuối (VNĐ)')
plt.legend()
plt.show()

# ---------------------------
# 7. Đánh giá
# ---------------------------
print('---- Train ----')
print('R2:', r2_score(y_train_true, y_train_pred))
print('MAE:', mean_absolute_error(y_train_true, y_train_pred))
print('MAPE:', mean_absolute_percentage_error(y_train_true, y_train_pred))

if len(y_test_pred) > 0:
    print('---- Test ----')
    print('R2:', r2_score(y_test_true, y_test_pred))
    print('MAE:', mean_absolute_error(y_test_true, y_test_pred))
    print('MAPE:', mean_absolute_percentage_error(y_test_true, y_test_pred))

# ---------------------------
# 8. Dự đoán ngày kế tiếp
# ---------------------------
last_50 = sc_data[-50:]
x_next = np.reshape(last_50, (1, last_50.shape[0], 1))
y_next_pred = model.predict(x_next)
y_next_pred_real = sc.inverse_transform(y_next_pred)

next_date = df['Ngày'].iloc[-1] + pd.Timedelta(days=1)
print(f"Ngày tiếp theo: {next_date.date()}, Giá dự đoán: {y_next_pred_real[0][0]:.2f}")
