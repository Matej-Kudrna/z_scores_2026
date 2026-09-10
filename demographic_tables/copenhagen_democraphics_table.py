import pandas as pd

original = pd.read_excel(r"E:\Project_DCE\Copenhagen\TEMPLATE_MAGNIMS 1_3_2023_v2_sent.xlsx")

new = original[['Patient_ID', 'MRI_date', 'Sex', 'Age_years']].copy()

new['MRI_date'] = new['MRI_date'].dt.strftime('%Y%m%d')
new['Patient_ID'] = new['Patient_ID'].str.replace('sub-', '')
new.rename(columns={'Patient_ID': 'PatientID', 'MRI_date': 'AcquisitionDate', 'Sex': 'PatientSex', 'Age_years': 'PatientAge'}, inplace=True)
new.to_excel(r"E:\Project_DCE\Copenhagen\bids_demographics_sequences.xlsx", index=False)
