import pandas as pd

df = pd.read_csv("cptac_119_clinical_final.csv")

os_event = 'Survival status (1, dead; 0, alive)'
rfs_event = 'Recurrence status (1, yes; 0, no)'

print("--- Event Distribution Summary ---")
if os_event in df.columns:
    print("\nOverall Survival Status:")
    print(df[os_event].value_counts(dropna=False))

if rfs_event in df.columns:
    print("\nRecurrence Status:")
    print(df[rfs_event].value_counts(dropna=False))
