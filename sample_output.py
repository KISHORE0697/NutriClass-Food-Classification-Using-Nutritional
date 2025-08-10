import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\user\Desktop\NutriClass Food Classification Using Nutritional Data\synthetic_food_dataset_imbalanced.csv")


# Simulated models and results for demonstration (replace this with your actual model loop)
models_results = {
    "Logistic Regression": {
        "accuracy": 0.85,
        "conf_matrix": [[50, 5], [10, 35]],
    },
    "Decision Tree": {
        "accuracy": 0.80,
        "conf_matrix": [[48, 7], [12, 33]],
    },
    "Random Forest": {
        "accuracy": 0.90,
        "conf_matrix": [[52, 3], [7, 38]],
    }
}

# Create folder for output
os.makedirs("ml_plots", exist_ok=True)

# Create and save plots
html_images = []
for name, result in models_results.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = result["conf_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} - Accuracy: {result['accuracy']:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plot_path = f"ml_plots/{name.replace(' ', '_')}_conf_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    html_images.append(f'<h3>{name}</h3><img src="{plot_path}" width="500"><br>')

# Create HTML report
html_content = f"""
<html>
<head><title>Model Evaluation Report</title></head>
<body>
    <h1>Machine Learning Model Evaluation</h1>
    {''.join(html_images)}
</body>
</html>
"""
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}


html_file_path = "ml_plots/ml_results.html"
with open(html_file_path, "w") as f:
    f.write(html_content)

X = df.drop("Meal_Type", axis=1)
y = df["Meal_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


html_file_path  # Return the path to the generated HTML file
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Accuracy: {acc:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save the plot
    plot_path = f"ml_plots/{name.replace(' ', '_')}_conf_matrix.png"
    plt.savefig(plot_path)
    plt.close()
models = ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors", 
          "SVM", "Gradient Boosting", "XGBoost"]

html_content = "<html><head><title>Model Evaluation</title></head><body><h1>Model Evaluation Report</h1>"

for model in models:
    img_file = f"{model.replace(' ', '_')}_conf_matrix.png"
    html_content += f"<h3>{model}</h3><img src='ml_plots/{img_file}' width='600'><br><br>"

html_content += "</body></html>"

with open("ml_plots/ml_results.html", "w") as f:
    f.write(html_content)
from selenium import webdriver
import os
import time

file_path = os.path.abspath("ml_plots/ml_results.html")
url = f"file:///{file_path.replace(os.sep, '/')}"

driver = webdriver.Chrome()
driver.get(url)

time.sleep(60)
driver.quit()
