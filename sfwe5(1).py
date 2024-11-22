import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_data(filename):
    return pd.read_csv(filename)

def check_pass_fail(grades):
    return int(all(grade >= 50 for grade in grades))

def prepare_data(df):
    X = df[[f'Course{i}' for i in range(1, 7)]].values
    y = df.apply(lambda row: check_pass_fail([row[f'Course{i}'] for i in range(1, 7)]), axis=1).values
    return X, y

filename = 'students_grades.csv'
df = load_data(filename)
X, y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Updated model parameters
model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=500, learning_rate_init=0.01, alpha=0.01, early_stopping=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy_percentage = accuracy * 100
print(f"Model accuracy: {accuracy_percentage:.2f}%")
