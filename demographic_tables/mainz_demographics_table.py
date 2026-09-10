import pandas as pd

original = pd.read_excel(r"E:\Project_DCE\Mainz\bids\MAGNIMS_T1_T2_mapping_Mainz.xlsx")
original['PatientID'] = original['Group'] + original['ID'].astype(str).str.zfill(3)

new = original[['PatientID', 'MRI_date', 'Sex', 'Age at MRI']].copy()
new.rename(columns={'MRI_date': 'AcquisitionDate', 'Sex': 'PatientSex', 'Age at MRI': 'PatientAge'}, inplace=True)

new['AcquisitionDate'] = '01'

new['PatientSex'] = new['PatientSex'].map({'Female': 'F', 'Male': 'M'})
print(new.head())

new.to_excel(r"E:\Project_DCE\Mainz\bids\bids_demographics_sequences.xlsx", index=False)
