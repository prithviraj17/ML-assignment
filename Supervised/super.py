import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Title
st.title("🛒 Customer Purchase Prediction")

st.write("Predict whether a customer will purchase a product using ML models.")

# ✅ Load dataset from CSV
data = pd.read_csv("customer_data.csv")

# Show dataset (optional)
st.subheader("📂 Dataset Preview")
st.dataframe(data)

# Features and target
X = data[['age', 'income', 'browsing_time', 'previous_purchases']]
y = data['purchased']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Accuracy
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

# ---------------- UI INPUT ----------------
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 70, 25)
income = st.sidebar.slider("Income", 10000, 100000, 30000)
browsing_time = st.sidebar.slider("Browsing Time (minutes)", 1, 60, 10)
previous_purchases = st.sidebar.slider("Previous Purchases", 0, 10, 1)

# Prediction
input_data = np.array([[age, income, browsing_time, previous_purchases]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    lr_pred = lr_model.predict(input_scaled)[0]
    knn_pred = knn_model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Results")

    st.write(f"Logistic Regression: {'Yes' if lr_pred==1 else 'No'}")
    st.write(f"KNN: {'Yes' if knn_pred==1 else 'No'}")

    if lr_pred == knn_pred:
        st.success(f"Final Decision: {'Customer WILL Purchase' if lr_pred==1 else 'Customer will NOT Purchase'}")
    else:
        st.warning("Models disagree! Try more data for better accuracy.")

# ---------------- GRAPH 1 ----------------
st.subheader("📈 Graph 1: Income vs Browsing Time")

fig1, ax1 = plt.subplots()
ax1.scatter(data['income'], data['browsing_time'], c=data['purchased'])
ax1.set_xlabel("Income")
ax1.set_ylabel("Browsing Time")
ax1.set_title("Income vs Browsing Time (Color = Purchased)")
st.pyplot(fig1)

# ---------------- GRAPH 2 ----------------
st.subheader("📊 Graph 2: Model Accuracy Comparison")

fig2, ax2 = plt.subplots()
models = ['Logistic Regression', 'KNN']
accuracy = [lr_acc, knn_acc]

ax2.bar(models, accuracy)
ax2.set_ylabel("Accuracy")
ax2.set_title("Model Comparison")

st.pyplot(fig2)