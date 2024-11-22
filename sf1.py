import pandas as pd

def check_pass_fail(grades):
    """Check if the student has failed based on their grades."""
    return all(grade >= 50 for grade in grades)

# processing
def process_chunk(chunk):
    for index, row in chunk.iterrows():
        student_id = row['StudentID']
        grades = [row[f'Course{i}'] for i in range(1, 7)]
        result = "Pass" if check_pass_fail(grades) else "Fail"
        print(f"{student_id}: {result}")

def main():
    chunk_size = 100

    try:
        # Read  CSV file
        for chunk in pd.read_csv('student_grades.csv', chunksize=chunk_size):
            process_chunk(chunk)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
