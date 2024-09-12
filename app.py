import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv(r"C:\Users\KIIT\Desktop\credit\creditcard.csv")

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Keep only the first 30 relevant features
features_columns = data.columns[:-1]  # Assuming the last column is 'Class'
X = data[features_columns]  # Use only the first 30 features
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Train and evaluate models
accuracy_scores = {}
roc_curves = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy_scores[model_name] = accuracy_score(y_test, y_pred)
    
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[model_name] = (fpr, tpr)

color_map = {
    "Logistic Regression": "green",
    "Random Forest": "blue",
    "Decision Tree": "red"
}

# Streamlit app
st.title("Credit Card Fraud Detection with Different Algorithms")
st.subheader("Model Comparison")

# Display accuracy scores
st.write("Model Accuracy Scores:")
st.write(accuracy_scores)

# Plot bar chart for accuracy comparison
fig_bar, ax_bar = plt.subplots()
bars = ax_bar.bar(accuracy_scores.keys(), accuracy_scores.values(), color=[color_map[model] for model in accuracy_scores.keys()])
ax_bar.set_ylabel('Accuracy')
ax_bar.set_title('Model Accuracy Comparison')
st.pyplot(fig_bar)

# ROC Curve plotting
st.write("ROC Curves:")
fig_roc, ax_roc = plt.subplots()
for model_name, (fpr, tpr) in roc_curves.items():
    ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {auc(fpr, tpr):.2f})", color=color_map[model_name])
ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves")
ax_roc.legend(loc='best')
st.pyplot(fig_roc)

# Allow user to select the model
st.write("Select Model for Prediction:")
model_choice = st.selectbox("Choose a model:", list(models.keys()))
selected_model = models[model_choice]

# User input for prediction
st.title("Predict Transaction Type")
input_df = st.text_input('Input all 30 features, separated by commas:')
input_df_lst = input_df.split(',')
submit = st.button("Submit")

if submit:
    try:
        features = np.array(input_df_lst, dtype=np.float64)
        if len(features) != 30:
            st.error("Please enter exactly 30 features.")
        else:
            # Scale the features
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = selected_model.predict(features_scaled)
            
            if prediction[0] == 0:
                st.title("Legitimate transaction")
            else:
                st.title("Fraudulent transaction")
    except ValueError:
        st.error("Please ensure all input features are numeric.")

# Display classification report
st.write("Classification Report:")
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(f"Classification Report for {model_name}:")
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))
