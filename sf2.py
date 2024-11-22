import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress warnings about convergence
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def check_pass_fail(grades):
    """Check if the student has failed based on their grades."""
    return int(all(grade >= 50 for grade in grades))


def train_model(filename):
    # Load the dataset
    df = pd.read_csv(filename)

    # Prepare features and labels
    X = df[[f'Course{i}' for i in range(1, 7)]]
    y = df.apply(lambda row: check_pass_fail([row[f'Course{i}'] for i in range(1, 7)]), axis=1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the Single Layer Perceptron model
    model = MLPClassifier(hidden_layer_sizes=(), max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    return model, scaler


def predict_pass_fail(model, scaler):
    # Get user input for marks
    print("Enter the marks for the 6 courses:")
    marks = [float(input(f"Course{i}: ")) for i in range(1, 7)]

    # Convert marks to DataFrame and standardize the user input
    marks_df = pd.DataFrame([marks], columns=[f'Course{i}' for i in range(1, 7)])
    marks_scaled = scaler.transform(marks_df)

    # Predict pass/fail
    prediction = model.predict(marks_scaled)
    result = "Pass" if prediction[0] == 1 else "Fail"

    print(f"Predicted probability of passing: {model.predict_proba(marks_scaled)[0][1]:.2f}")
    print(f"The student is predicted to: {result}")


def main():
    filename = 'student_grades.csv'

    # Train the model
    model, scaler = train_model(filename)

    # Predict pass/fail for user input
    predict_pass_fail(model, scaler)


if __name__ == "__main__":
    main()
