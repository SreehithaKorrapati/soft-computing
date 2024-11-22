import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score

data = pd.read_csv('student_grades.csv')
data['Failed'] = (data[['Course1', 'Course2', 'Course3', 'Course4', 'Course5', 'Course6']] < 50).any(axis=1)

X = data.drop(columns=['StudentID', 'Failed'])
y = data['Failed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# classifiers initialized
decision_tree = DecisionTreeClassifier(random_state=42)
svm_classifier = SVC(random_state=42)
knn_classifier = KNeighborsClassifier()

classifiers = [decision_tree, svm_classifier, knn_classifier]
classifier_names = ['Decision Tree', 'SVM', 'KNN']

# Train
for classifier, name in zip(classifiers, classifier_names):
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
# print
    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")

print(data.head())
print(data['Failed'].value_counts())

