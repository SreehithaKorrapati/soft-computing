import pandas as pd
import numpy as np

# Define the number of students and courses
num_students = 1000
num_courses = 6

# Generate student IDs
student_ids = [f'Student{i+1}' for i in range(num_students)]

# Generate random marks between 40 and 100 for each course
data = {
    'StudentID': student_ids
}

for course in range(1, num_courses + 1):
    data[f'Course{course}'] = np.random.randint(40, 100, size=num_students)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('students_grades.csv', index=False)

print("CSV file 'student_grades.csv' created successfully with 1000 entries.")
