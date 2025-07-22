import pandas as pd

# Load the .xls file
xls_file = pd.read_excel("heart_failure_clinical_records_dataset (1).xls")

# Save as .csv
xls_file.to_csv("heart_failure_clinical_records_dataset.csv", index=False)

print("Conversion complete: XLS ➜ CSV")
