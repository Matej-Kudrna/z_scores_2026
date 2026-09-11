import pandas as pd


results = pd.read_csv(r"E:\Project_DCE\DCE_PRG_NEW\bids_test_hc\derivatives\2026_09_06\results_all.csv")

results["subject"] = 'PRG-' + results["subject"].astype(str)

results['Phenotype'] = 'HC'

results.to_csv(r"E:\Project_DCE\DCE_PRG_NEW\bids_test_hc\derivatives\2026_09_06\results_clinics.csv")