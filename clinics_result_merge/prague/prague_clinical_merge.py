import pandas as pd
from patient_id_mapper import fill_from_RC

def get_ms_course_at_date(row, phenotype_date_col='Visit Date'):
    """
    Vrátí (MS course, datum přechodu na ten course).
    """
    try:
        mri_date = pd.to_datetime(row[phenotype_date_col], format='%d.%m.%Y')
    except:
        return None, None

    date1 = pd.to_datetime(row['Date MSCourse 1'], format='%d.%m.%Y', errors='coerce')
    date2 = pd.to_datetime(row['Date MSCourse 2'], format='%d.%m.%Y', errors='coerce')
    date3 = pd.to_datetime(row['Date MSCourse 3'], format='%d.%m.%Y', errors='coerce')

    if mri_date < date1:
        return None, None
    elif pd.isna(date2) or mri_date < date2:
        return row['MSCourse1'], row['Date MSCourse 1']
    elif pd.isna(date3) or mri_date < date3:
        return row['MSCourse2'], row['Date MSCourse 2']
    else:
        return row['MSCourse3'], row['Date MSCourse 3']


# read all the files
sestry = pd.read_pickle(r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\magnims_project_39\sestry_all.pkl")
imed_vi_path = r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\iMed_export\IMED export 26_1_2026\imed_export 26_1_2026_VI.txt"
imed_id_path = r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\iMed_export\IMED export 26_1_2026\imed_export 26_1_2026_ID.txt"
results = pd.read_csv(r"E:\Project_DCE\DCE_PRG_NEW\bids\derivatives\2026_07_02\results_all.csv")

# get IDs to the sestry table
cohorts = sestry[['Rodne_cislo', 'cohort']]
sestry_ids = fill_from_RC(cohorts, 'Rodne_cislo')
sestry_ids.rename(columns={'cohort': 'cohort_sestry'}, inplace=True)

print(sestry_ids[:][:3])
# ananymize, keep only paciid, and imed id
sestry_imed_paciid = sestry_ids.drop(columns=['Rodne_cislo', 'RC'])
sestry_imed_paciid.rename(columns={'paciid': 'subject', 'iMed_ID': 'Patient ID'}, inplace=True)

# inner merge with merge_pipeline_results from lausanne pipeline from merge_pipeline_results\results_all.py on dce_prg_new bids structure
prg_results_merge = pd.merge(sestry_imed_paciid, results, how='inner', on='subject', indicator=True)

# sessions to datetime for merge with edss and phenotype
prg_results_merge['session_dt'] = pd.to_datetime(prg_results_merge['session'], format='%Y%m%d')

print(sestry_ids.head(10))
## ------- get phenotype from ID table to every visit row in VI table, than merge on closest date with merge_pipeline_results
# get EDSS from imed_export
imed_visits = pd.read_csv(
    imed_vi_path, sep="\t", header=0, encoding="latin-1",
    dtype={'Patient Code': "string", "Patient ID": "string"}
)
imed_edss = imed_visits[["Patient ID", 'Visit Date', 'EDSS']].dropna().drop_duplicates()

#get fenotype from imed_export
imed_id = pd.read_csv(
    imed_id_path, sep="\t", header=0, encoding="latin-1",
    dtype={'Patient Code': "string", "Patient ID": "string"}
)

# redukce sloupcu
imed_edss = imed_id[["Patient ID", 'MSCourse1', 'Date MSCourse 1', 'MSCourse2',
       'Date MSCourse 2', 'MSCourse3', 'Date MSCourse 3', 'Birth Date', 'Gender']]

# sex age from mr database
mr_path = r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA\MrExams\ValuesWithHeader_T1_FLAIR_20260304.TXT"

database = pd.read_csv(mr_path,
                    sep="\t", header=0, encoding="cp1250", dtype=str
                )
database['session'] = pd.to_datetime(database['DatumVys'], format="%d.%m.%Y").dt.strftime('%Y%m%d').astype(int)
database['subject'] = database['paciid'].astype(int)
prg_results_merge_mr = pd.merge(prg_results_merge, database[['subject', 'VekPriVys', 'PatSex', 'session']], on=["subject", 'session'])
prg_results_merge_mr.rename(columns={
    'PatSex': 'Sex',
    'VekPriVys': 'Age_years',
}, inplace=True)

# Sloučit oba dataframy
imed_visits['Patient ID'] = imed_visits['Patient ID'].astype(str)
imed_edss['Patient ID'] = imed_edss['Patient ID'].astype(str)

imed_edss_phenotype = imed_visits.merge(imed_edss, on='Patient ID', how='left', indicator=True)
imed_edss_phenotype = imed_edss_phenotype[imed_edss_phenotype['Patient ID'].astype(str).isin(prg_results_merge_mr['Patient ID'].astype(str))]
# Přidat sloupec s aktivním kurzem v den navstevy
imed_edss_phenotype[['phenotype_imed', 'phenotype_imed_date']] = imed_edss_phenotype.apply(
    lambda row: pd.Series(get_ms_course_at_date(row, 'Visit Date')),
    axis=1
)



##----- merge to get edss and phenotype in the time of MRI
imed_edss_phenotype['session_dt'] = pd.to_datetime(imed_edss_phenotype['Visit Date'], format='%d.%m.%Y')
imed_edss_phenotype['Birth Date'] = pd.to_datetime(imed_edss_phenotype['Birth Date'], format='%d.%m.%Y')

prg_results_merge_mr['Patient ID'] = prg_results_merge_mr['Patient ID'].astype(str)
imed_edss_phenotype['Patient ID'] = imed_edss_phenotype['Patient ID'].astype(str)

## redukce klinickych dat
# clinics_clean = clinics[['subject', 'Sex', 'Date_of_birth', 'EDSS_total_score', 'Phenotype', 'Age_years']]
imed_edss_phenotype_reduced = imed_edss_phenotype[['Patient ID', 'session_dt', 'EDSS', 'phenotype_imed', 'phenotype_imed_date', 'Birth Date']]
imed_edss_phenotype_reduced.rename(columns={
    'Birth Date': 'Date_of_birth',
    'EDSS': 'EDSS_total_score',
    'phenotype_imed': 'Phenotype'
}, inplace=True)

print(imed_edss_phenotype_reduced['Phenotype'].value_counts())

cleanup_map = {
    'RR': 'RRMS',
    'PR': 'RRMS',   # tohle nevim co je
    'SP': 'SPMS',
    'PP': 'PPMS',
    "CIS": 'RRMS'
}
imed_edss_phenotype_reduced['Phenotype'] = imed_edss_phenotype_reduced['Phenotype'].replace(cleanup_map)



prg_results_merge_edss = pd.merge_asof(
    prg_results_merge_mr.sort_values('session_dt'),  # seřazeno
    imed_edss_phenotype_reduced.sort_values('session_dt'),  # seřazeno
    on='session_dt',  # merguj na datum!
    by='Patient ID',  # pro každého pacienta zvlášť
    direction='backward'  # vezmi poslední EDSS před datem
)

prg_results_merge_edss['subject'] = 'PRG-' + prg_results_merge_edss['subject'].astype(str)

prg_results_merge_edss.drop(columns=['cohort_sestry', 'Patient ID', 'phenotype_imed_date', 'session_dt', '_merge'], inplace=True)

prg_results_merge_edss.to_csv(r"E:\Project_DCE\DCE_PRG_NEW\bids\derivatives\2026_07_02\\prague_results_clinics.csv", index=False)

print('')