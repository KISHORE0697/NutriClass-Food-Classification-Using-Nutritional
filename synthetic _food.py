import pandas as pd


df = pd.read_csv(r"C:\Users\user\Desktop\NutriClass Food Classification Using Nutritional Data\synthetic_food_dataset_imbalanced.csv")

total_missing = df.isnull().sum().sum()
df.isnull().sum().sort_values(ascending=False)
before_drop = df.shape[0]
df = df.dropna()
after_drop = df.shape[0]
before_dup = df.shape[0]
df = df.drop_duplicates()
after_dup = df.shape[0]

categorires_column = ['Meal_Type','Preparation_Method','Food_Name']

for column in categorires_column:
    df[column] = df[column].astype('category')

### TO FIND OUTLIERS ###

numerical_column = ['Calories','Protein','Fat','Carbs','Sugar','Fiber','Sodium','Cholesterol','Glycemic_Index','Water_Content','Serving_Size']

for column in numerical_column:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)  
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR    

outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0] 
print(f"{column}: {outliers} outliers")

# result found Serving_size has 84 outliers hence this is trained through IQR method with high and low bound values

column = 'Serving_Size'
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75) 
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR    

df[column] = df[column].clip(lower_bound, upper_bound) 

outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
print(f"After capping, number of outliers in '{column}': {len(outliers)}")


##############  Normalization ##############
## through Mean 0 and Standard Deviation 1

from sklearn.preprocessing import StandardScaler

numerical_column = ['Calories','Protein','Fat','Carbs','Sugar','Fiber','Sodium','Cholesterol','Glycemic_Index','Water_Content','Serving_Size']
StandardScaler = StandardScaler()
df[numerical_column] = StandardScaler.fit_transform(df[numerical_column])
print(df[numerical_column].describe())

############# feature engineering ############# to reduce dimensionality  Categorical columns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns   


label_encoder = LabelEncoder()
df['Target'] = label_encoder.fit_transform(df['Meal_Type'])

# One-hot encode categorical features (excluding target)
df = pd.get_dummies(df, columns=['Preparation_Method', 'Food_Name'])

# Prepare features and target
X = df.drop(['Target', 'Meal_Type'], axis=1)
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier()
}

# Train and evaluate models
for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Accuracy: {acc:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Feature importance (from Random Forest)
importances = models["Random Forest"].feature_importances_
feat_names = X.columns
feat_importance = pd.Series(importances, index=feat_names).sort_values(ascending=False)
feat_importance.head(20).plot(kind='barh', title='Top 20 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Example values (replace with your actual y_test and y_pred)
y_test = [0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# --- Save Confusion Matrix as Image ---
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Save image
plt.close()


html_content = f"""
<html>
<head><title>ML Model Evaluation</title></head>
<body>
    <h1>Model Evaluation Report</h1>
    <h2>Accuracy: {accuracy:.4f}</h2>

    <h2>Classification Report:</h2>
    <pre>{classification_report(y_test, y_pred)}</pre>

    <h2>Confusion Matrix:</h2>
    <img src="confusion_matrix.png" alt="Confusion Matrix">

</body>
</html>
"""

# --- Save to HTML file ---
with open("ml_results.html", "w") as f:
    f.write(html_content)