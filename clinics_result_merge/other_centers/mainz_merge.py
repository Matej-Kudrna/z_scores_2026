import pandas as pd

results = pd.read_csv(r"E:\Project_DCE\3_Bochum\bids\derivatives\2026_05_05\results_all.csv")
clinics = pd.read_excel(r"E:\Project_DCE\3_Bochum\bids\Magnims_quant_Bochum.xlsx")

print(results.columns.tolist()[:10])
print(clinics.columns.tolist()[:10])

clinics.rename(columns={'Patient_ID': 'subject', 'MRI_date': 'session'}, inplace=True)
# clinics.drop(columns=['session'], inplace=True)

print(results[['subject', 'session']][:2])
print(clinics[['subject', 'session']][:2])

clinics.drop(columns=['session'], inplace=True)

results["subject"] = 'BOCHUM-' + results["subject"].astype(str).str.zfill(3)
clinics["subject"] = clinics["subject"].astype(str).str.replace('Subject_', 'BOCHUM-')

print(results['subject'][:2])
print(clinics['subject'][:2])

df_merge = pd.merge(clinics, results, on="subject", how="left")

df_merge.to_csv(r"E:\Project_DCE\3_Bochum\bids\derivatives\2026_05_05\bochum_results_clinics.csv", index=False)