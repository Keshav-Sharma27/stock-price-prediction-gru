📈 Stock Price Prediction using GRU

This project is a deep learning–based stock price prediction system using a **GRU (Gated Recurrent Unit)** model.  
It also includes a **Streamlit web application** where users can upload stock CSV data and predict the **next day closing price**.

---

🚀 Features
- Predicts next-day stock closing price
- Uses GRU (Deep Learning – RNN)
- Automatic technical indicator calculation:
  - Moving Average (MA7, MA21)
  - RSI (14)
  - MACD
- Streamlit web interface
- Supports raw CSV files downloaded from Yahoo Finance

---

🛠️ Tech Stack
- Python
- TensorFlow / Keras
- GRU (Recurrent Neural Network)
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📂 Project Structure

stock-price-prediction-gru/
│
├── app.py # Streamlit web app
├── requirements.txt # Project dependencies
├── models/
│ └── gru_model.h5 # Trained GRU model
├── data/
│ ├── raw/ # Raw stock CSV files
│ └── processed/ # Data with indicators
├── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 02_gru_model.ipynb
│ └── 03_lstm_model.ipynb
└── README.md


---

▶️ How to Run the Project

 1️⃣ Clone the repository
git clone https://github.com/Keshav-Sharma27/stock-price-prediction-gru.git
cd stock-price-prediction-gru


2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate


3️⃣ Install dependencies

4️⃣ Run the Streamlit app

---

📊 How Prediction Works
- User uploads a raw stock CSV file
- Indicators are automatically calculated
- Last 60 days of data is used
- GRU predicts the next day closing price

---

🎯 Future Improvements
- Multi-day prediction
- Online deployment
- Better visualization

---

👤 Author
**Keshav Sharma**  
B.Tech – Computer Science (AI&ML)

