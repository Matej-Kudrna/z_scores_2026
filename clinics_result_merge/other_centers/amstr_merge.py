import pandas as pd

results = pd.read_csv(r"E:\Project_DCE\5_Amsterdam\bids\derivatives\2026_06_23\results_all.csv")
clinics = pd.read_excel(r"E:\Project_DCE\5_Amsterdam\bids\export_2026_05_20260311.xlsx", sheet_name='export_ms_type_overview')
clinics_sex_age = pd.read_excel(r"E:\Project_DCE\5_Amsterdam\bids\export_2026_05_20260311.xlsx", sheet_name='export_pat')
clinics_edss = pd.read_excel(r"E:\Project_DCE\5_Amsterdam\bids\export_2026_05_20260311.xlsx", sheet_name='export_clinical')

clinics_short = pd.merge(clinics_edss[['pat_id', 'edss', 'visit_date']], clinics[['pat_id', 'current_ms']], on=['pat_id'], how='left')
clinics_short = pd.merge(clinics_short, clinics_sex_age[['pat_id', 'gender', 'age_march_2026']], on='pat_id', how='left')
clinics_short.dropna(inplace=True, subset='edss')

results['session_dt'] = pd.to_datetime(results['session'], format='%Y%m%d')
results['subject'] = results['subject'].str.replace('MSCA', '').astype(int)
##--------------------------------------------------------------------------
# # overit ze sjou phenotypy pred datem rezonance
#---------------------------------------------------------------------------
results_sub_ses = results[['subject', 'session']]
cli_res_merge = pd.merge(results_sub_ses, clinics, how='left', left_on=['subject'], right_on='pat_id')

print(pd.to_datetime(cli_res_merge['session'], format='%Y%m%d') > cli_res_merge['relapsing_remitting'])
clinics_short['pat_id'] = clinics_short['pat_id'].astype(int)
results_phenotype = pd.merge_asof(
    results.sort_values('session_dt'),
    clinics_short.rename(columns={'pat_id': 'subject'}).sort_values('visit_date'),
    direction='backward',
    left_on='session_dt',
    right_on='visit_date',
    by='subject'
)

print(results[['subject', 'session']])
print(clinics_short.columns.tolist()[:10])

results_phenotype.rename(columns={'pat_id': 'subject', 'current_ms': 'Phenotype',
                                  'gender': 'Sex', 'age_march_2026': 'Age_years',
                                  'edss': 'EDSS_total_score'}
                         , inplace=True
                         )


results_phenotype["subject"] = 'AMSTR-' + results_phenotype["subject"].astype(str)

print(results_phenotype['Phenotype'].value_counts())
results_phenotype.drop(columns=['session_dt'], inplace=True)

cleanup_map = {
    'male': 'M',
    'female': 'F'
}
results_phenotype['Sex'] = results_phenotype['Sex'].replace(cleanup_map)
print(results_phenotype['Sex'].value_counts())

results_phenotype.to_csv(r"E:\Project_DCE\5_Amsterdam\bids\derivatives\2026_06_23\results_clinics.csv", index=False)