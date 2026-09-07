import pandas as pd

results = pd.read_csv(r"E:\Project_DCE\4_Mainz\bids\derivatives\2026_06_23\results_all.csv")
clinics = pd.read_excel(r"E:\Project_DCE\4_Mainz\bids\MAGNIMS_T1_T2_mapping_Mainz.xlsx")

print(results.columns.tolist()[:10])
print(clinics.columns.tolist()[:10])

clinics.rename(columns={'ID': 'subject', 'MRI_date': 'session'}, inplace=True)
# clinics.drop(columns=['session'], inplace=True)

print(results[['subject', 'session']][:2])
print(clinics[['subject', 'session']][:2])

results.drop(columns=['session'], inplace=True)

results["subject"] = 'MAINZ-' + results["subject"].astype(str)
clinics["subject"] = 'MAINZ-' + clinics['Group'] + clinics["subject"].astype(str).str.zfill(3)

print(results['subject'][:2])
print(clinics['subject'][:2])

df_merge = pd.merge(clinics, results, on="subject", how="left")

df_merge.to_csv(r"E:\Project_DCE\4_Mainz\bids\derivatives\2026_06_23\mainz_results_clinics.csv", index=False)