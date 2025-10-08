import pandas as pd #đọc dữ liệu
import numpy as np #xử lý dữ liệu
from sklearn.preprocessing import MinMaxScaler #chuẩn hóa dữ liệu
from keras.models import Sequential #đầu vào
from keras.layers import LSTM, Dropout, Dense #các lớp để xây dựng mô hình
from keras.callbacks import ModelCheckpoint #lưu lại huấn luyện tốt nhất

# --- Đọc và xử lý dữ liệu ---
df = pd.read_csv('data.csv') # đọc dữ liệu từ file csv
df = df.drop(columns=["KL", "% Thay đổi"]) # Xóa hai dòng "KL" và "Thay đổi %" từ DataFrame dataSet
df["Ngày"] = pd.to_datetime(df.Ngày, format="%d/%m/%Y") #định dạng cấu trúc thời gian
df = df.sort_values(by='Ngày') # Sắp xếp lại dữ liệu theo thứ tự thời gian
for col in ['Lần cuối','Mở','Cao','Thấp']:  # Chuyển đổi định dạng các cột giá thành số thực 
    df[col] = df[col].str.replace(',', '').astype(float)

df1 = pd.DataFrame(df, columns=['Ngày','Lần cuối']) # Lấy thông tin năm từ cột "Ngày"
df1.index = df1.Ngày # Đặt cột "Ngày" làm chỉ mục
df1.drop('Ngày', axis=1, inplace=True) # Xóa cột "Ngày" khỏi DataFrame

# --- Chia dữ liệu train/test ---
data = df1.values # chuyển DataFrame thành mảng numpy
train_data = data[:1500] # lấy 1500 dòng đầu làm dữ liệu train

# --- Chuẩn hóa ---
sc = MinMaxScaler(feature_range=(0,1)) # khởi tạo hàm chuẩn hóa
sc_train = sc.fit_transform(data) # chuẩn hóa dữ liệu train

# --- Tạo dữ liệu sequence ---
x_train, y_train = [], [] # khởi tạo dữ liệu train
for i in range(50, len(train_data)):   
    x_train.append(sc_train[i-50:i, 0]) # lấy 50 giá trị trước làm dữ liệu đầu vào
    y_train.append(sc_train[i, 0]) # giá trị thứ 51 làm đầu ra
x_train, y_train = np.array(x_train), np.array(y_train) # chuyển sang mảng numpy
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # reshape lại dữ liệu đầu vào cho phù hợp với LSTM
y_train = y_train.reshape(-1,1) # reshape lại dữ liệu đầu ra

# --- Xây dựng mô hình ---
model = Sequential() # khởi tạo mô hình tuần tự
model.add(LSTM(128, input_shape=(x_train.shape[1],1), return_sequences=True)) # lớp LSTM đầu tiên với 128 đơn vị, trả về chuỗi
model.add(LSTM(64)) # lớp LSTM thứ hai với 64 đơn vị
model.add(Dropout(0.5)) # lớp Dropout để tránh overfitting
model.add(Dense(1)) # lớp Dense đầu ra với 1 đơn vị
model.compile(loss='mean_absolute_error', optimizer='adam') # biên dịch mô hình với hàm mất mát MAE và trình tối ưu Adam

# --- Train và lưu mô hình ---
save_model = "save_model.keras" # đường dẫn lưu mô hình
checkpoint = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto') # callback để lưu mô hình tốt nhất
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[checkpoint]) # huấn luyện mô hình

# --- Lưu scaler nếu muốn dùng lại ---
import joblib # thư viện để lưu mô hình
joblib.dump(sc, "scaler.save") # lưu scaler đã dùng để chuẩn hóa dữ liệu
print("Train xong, mô hình và scaler đã lưu.") # thông báo hoàn thành
