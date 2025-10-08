import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint

# --- Đọc và xử lý dữ liệu ---
df = pd.read_csv('data.csv')
df = df.drop(columns=["KL", "% Thay đổi"])
df["Ngày"] = pd.to_datetime(df.Ngày, format="%d/%m/%Y")
df = df.sort_values(by='Ngày')
for col in ['Lần cuối','Mở','Cao','Thấp']:
    df[col] = df[col].str.replace(',', '').astype(float)

df1 = pd.DataFrame(df, columns=['Ngày','Lần cuối'])
df1.index = df1.Ngày
df1.drop('Ngày', axis=1, inplace=True)

# --- Chia dữ liệu train/test ---
data = df1.values
train_data = data[:1500]

# --- Chuẩn hóa ---
sc = MinMaxScaler(feature_range=(0,1))
sc_train = sc.fit_transform(data)

# --- Tạo dữ liệu sequence ---
x_train, y_train = [], []
for i in range(50, len(train_data)):
    x_train.append(sc_train[i-50:i, 0])
    y_train.append(sc_train[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
y_train = y_train.reshape(-1,1)

# --- Xây dựng mô hình ---
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1],1), return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')

# --- Train và lưu mô hình ---
save_model = "save_model.keras"
checkpoint = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[checkpoint])

# --- Lưu scaler nếu muốn dùng lại ---
import joblib
joblib.dump(sc, "scaler.save")
print("Train xong, mô hình và scaler đã lưu.")
