import pandas as pd

results = pd.read_csv(r"E:\Project_DCE\2_Genova\bids\derivatives\2026_05_05\results_all.csv")
clinics = pd.read_excel(r"E:\Project_DCE\2_Genova\bids\GENOVA.xlsx")

print(results.columns.tolist()[:10])
print(clinics.columns.tolist()[:10])

clinics.rename(columns={'Patient_ID': 'subject', 'MRI_date': 'session'}, inplace=True)
clinics.drop(columns=['session'], inplace=True)

print(results['subject'][:2])
print(clinics['subject'][:2])

results["subject"] = 'GE' + results["subject"].astype(str).str.zfill(3)

print(results['subject'][:2])
print(clinics['subject'][:2])

df_merge = pd.merge(clinics, results, on="subject", how="left")

df_merge.to_csv(r"E:\Project_DCE\1_Copenhagen\derivatives\2026_06_15\copenhagen_results_clinics.csv", index=False)