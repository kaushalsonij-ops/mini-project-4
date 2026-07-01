import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction System")

# 1. Dataset Read Kar
df = pd.read_csv("StockPrice.csv")

# 2. X aur y Alag Kar - Requirement 3
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# 3. Train-Test Split 80-20 - Requirement 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Train Kar - Requirement 5
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Accuracy Calculate Kar - Requirement 7
y_pred_test = model.predict(X_test)
accuracy = r2_score(y_test, y_pred_test)

# Streamlit UI Start - Requirement 8
st.subheader("Dataset")
with st.expander("View Dataset"):
    st.dataframe(df)

st.subheader("📊 Enter Market Parameters to Predict Close Price")

col1, col2 = st.columns(2)
with col1:
    open_price = st.number_input("Open Price:", min_value=0.0, value=170.0, step=1.0)
    high_price = st.number_input("High Price:", min_value=0.0, value=175.0, step=1.0)

with col2:
    low_price = st.number_input("Low Price:", min_value=0.0, value=168.0, step=1.0)
    volume = st.number_input("Volume:", min_value=0, value=152000, step=1000)

if st.button("Predict Closing Price 💰"):
    # 6. User Input Se Prediction - Requirement 6
    user_input = [[open_price, high_price, low_price, volume]]
    prediction = model.predict(user_input)

    st.success(f"Predicted Closing Price: ₹{prediction[0]:,.2f}")
    st.info(f"Model Accuracy (R² Score): {accuracy:.4f}")

    # Graph - Requirement Graph
    fig, ax = plt.subplots(figsize=(10,6))

    # Scatter Plot - Open vs Close
    ax.scatter(df['Open'], df['Close'], color='blue', s=60, label='Actual Data')

    # Regression Line - Simple line ke liye sirf Open pe model banaya
    temp_model = LinearRegression()
    temp_model.fit(df[['Open']], df['Close'])
    ax.plot(df['Open'], temp_model.predict(df[['Open']]), color='red', linewidth=2, label='Regression Line')

    # User ka Prediction Point
    ax.scatter(open_price, prediction[0], color='green', s=250, label='Your Prediction', marker='*')

    ax.set_title("Open Price vs Close Price", fontsize=16) # Graph Title
    ax.set_xlabel("Open Price (₹)") # X-axis Label
    ax.set_ylabel("Close Price (₹)") # Y-axis Label
    ax.grid(True) # Grid
    ax.legend() # Legend
    st.pyplot(fig)