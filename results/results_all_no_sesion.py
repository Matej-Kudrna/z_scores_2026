"""
================================================================================
ANALYSIS RESULTS AGGREGATOR
================================================================================

PURPOSE:
This script collects, processes, and consolidates quantitative analysis results
from multiple neuroimaging analysis pipelines into unified CSV/Excel outputs.

WORKFLOW:
1. Selects a derivatives directory containing analysis results
2. Processes results from 7 different analysis pipelines:
   - morphobox: volumetry data (vol_)
   - wmh: white matter hyperintensity segmentation (les_)
   - quant_periles_*: perilesional quantification (map_lez_)
   - quant_perivent_*: periventricular quantification (map_per_)
   - quant_regions_*: regional quantification (map_reg_)
   - quant_wmtracts_*: white matter tract quantification (map_tra_)
   - concord: concordance metrics (con_)
3. Merges all results into a single comprehensive results table
4. Exports to both CSV and Excel formats

OUTPUT:
- Individual CSVs for each analysis type
- results_all.csv/xlsx: complete merged dataset with all measurements
- A QA checker showing which analyses were run for each subject/session

REQUIREMENTS:
- pandas, glob, os, json, tkinter (standard libraries)
- Directory structure: derivatives/[analysis_folder]/sub-*/ses-*/
- CSV files within session folders

================================================================================
"""

import glob, os, json
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def select_directory():
    """
    Opens a GUI dialog to select the derivatives directory.

    Returns:
        str: Full path to selected directory, or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    directory_path = filedialog.askdirectory()
    if directory_path:
        print(f"Selected directory: {directory_path}")
        return directory_path
    else:
        print("No directory selected")

def prefix_to_columns(table, prefix):
    """
    Rearranges columns so subject/session are first, then adds prefix to other columns.

    Args:
        table (DataFrame): Input data
        prefix (str): Prefix to add to column names (e.g., 'vol_', 'map_lez_')

    Returns:
        DataFrame: Reorganized table with prefix applied
    """
    columns_to_front = ['subject', 'session'] + [col for col in table.columns
                                                   if col not in ['subject', 'session']]
    table = table[columns_to_front]
    table.columns = [
        f"{prefix}{col}" if col not in ['subject', 'session'] else col
        for col in table.columns
    ]
    return table

def analysis_checker(checker, data, analysis):
    """
    Updates the QA checker table tracking which analyses ran for each subject/session.

    Args:
        checker (DataFrame): Existing checker table
        data (DataFrame): Data from current analysis
        analysis (str): Name of the analysis

    Returns:
        DataFrame: Updated checker table
    """
    if ~data.empty:
        check = pd.DataFrame()
        check['subject'] = data['subject']
        check['session'] = data['session']
        check['analysis'] = analysis
        check['value'] = True
        checker = pd.concat([checker, check], ignore_index=True)
    else:
        check = pd.DataFrame()
        check['subject'] = data['subject']
        check['session'] = data['session']
        check['analysis'] = False
        check['value'] = False
        checker = pd.concat([checker, check], ignore_index=True)
    return checker

# ================================================================================
# ANALYSIS PROCESSING FUNCTIONS
# ================================================================================

def run_perilesional_results(derivatives_path, checker):
    """
    Process perilesional quantification data (quant_periles_t1 and t2).
    Outputs separate results for t1 and t2 weighted imaging.
    Prefix: 'map_lez_'
    Skips if CSV file not found in session folder.
    """
    prefix = 'map_lesions'

    # Process T1 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_periles_t1'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]
        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')

            table = pd.concat([table, data], ignore_index=True)
            checker = analysis_checker(checker, data, analysis)

    if not table.empty:
        table = prefix_to_columns(table, prefix)
        for col in ['map_lesion_id', 'map_relaxometry_measure', 'map_Unnamed: 0']:
            if col in table.columns:
                table.drop(col, axis=1, inplace=True)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    # Process T2 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_periles_t2'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]
        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')
            data.columns = [col.replace('_z_', '_t2_z_') for col in data.columns]
            table = pd.concat([table, data], ignore_index=True)
            checker = analysis_checker(checker, data, analysis)

    if not table.empty:
        table = prefix_to_columns(table, prefix)
        for col in ['map_lesion_id', 'map_relaxometry_measure', 'map_Unnamed: 0']:
            if col in table.columns:
                table.drop(col, axis=1, inplace=True)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    # Process T2/T1 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_periles_t2overt1'
    found = False
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            found = True
            break

    if found:
        subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                    and item.startswith('sub-')]

        for subject in subjects:
            sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                        if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                        and not item.endswith('.nii.gz')]
            for session in sessions:
                table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
                if not table_path:  # Skip if no CSV found
                    continue
                data = pd.read_csv(table_path[0])
                data['subject'] = subject.strip('sub-')
                data['session'] = session.strip('ses-')
                data.columns = [col.replace('_z_', '_t2overt1_z_') for col in data.columns]
                table = pd.concat([table, data], ignore_index=True)
                checker = analysis_checker(checker, data, analysis)

        if not table.empty:
            table = prefix_to_columns(table, prefix)
            for col in ['map_lesion_id', 'map_relaxometry_measure', 'map_Unnamed: 0']:
                if col in table.columns:
                    table.drop(col, axis=1, inplace=True)
            table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    return checker

def run_perivent_results(derivatives_path, checker):
    """
    Process periventricular quantification data (quant_perivent_t1 and t2).
    Outputs separate results for t1 and t2 weighted imaging.
    Aggregates by band (one row per band) - keeps only whole_band ROI.
    Each column becomes: map_per_{metric}_{band_index}
    Prefix: 'map_per_'
    Skips if CSV file not found in session folder.
    """
    prefix = 'map_perivent_'

    # Process T1 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_perivent_t1'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]
        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')


            # Aggregate by band and ROI: pivot to one row per subject/session
            if 'band' in data.columns and 'ROI' in data.columns:
                id_cols = ['subject', 'session']
                band_col = 'band'
                roi_col = 'ROI'
                metric_cols = [col for col in data.columns
                              if col not in id_cols + [band_col, roi_col]]

                # Create band_roi_id column for pivoting
                data['band_roi_id'] = 't1_' + 'band_' + data[band_col].astype(int).astype(str) + '_' + data[roi_col].astype(str)

                # Find all unique combinations
                all_bands = sorted([int(b) for b in data[band_col].dropna().unique()])
                all_rois = sorted(data[roi_col].dropna().unique())
                max_band = max(all_bands) if all_bands else 0

                # Pivot each metric separately
                pivoted = data[id_cols].drop_duplicates().copy()

                for metric in metric_cols:
                    pivot_data = data[id_cols + ['band_roi_id', metric]].copy()
                    pivot_wide = pivot_data.pivot_table(index=id_cols, columns='band_roi_id', values=metric, aggfunc='first')
                    pivot_wide.columns = [f'{metric}_{col}' for col in pivot_wide.columns]
                    pivoted = pivoted.merge(pivot_wide.reset_index(), on=id_cols, how='left')

                # Add missing columns for all band+ROI combinations
                for band_id in range(int(max_band) + 1):
                    for roi in all_rois:
                        for metric in metric_cols:
                            col_name = f'{metric}_band_{band_id}_{roi}'
                            if col_name not in pivoted.columns:
                                pivoted[col_name] = None

                data = pivoted

            table = pd.concat([table, data], ignore_index=True)
            checker = analysis_checker(checker, data, analysis)

    if not table.empty:
        table = prefix_to_columns(table, prefix)
        for col in list(table.columns):
            if 'Unnamed' in col or 'lesion_id' in col or 'relaxometry_measure' in col:
                table.drop(col, axis=1, inplace=True)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    # Process T2 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_perivent_t2'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]
        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')
            data.columns = [col.replace('_z_', '_t2_z_') for col in data.columns]

            # Aggregate by band and ROI: pivot to one row per subject/session
            if 'band' in data.columns and 'ROI' in data.columns:
                id_cols = ['subject', 'session']
                band_col = 'band'
                roi_col = 'ROI'
                metric_cols = [col for col in data.columns
                              if col not in id_cols + [band_col, roi_col]]

                # Create band_roi_id column for pivoting
                data['band_roi_id'] = 't2_' + 'band_' + data[band_col].astype(int).astype(str) + '_' + data[roi_col].astype(str)

                # Find all unique combinations
                all_bands = sorted([int(b) for b in data[band_col].dropna().unique()])
                all_rois = sorted(data[roi_col].dropna().unique())
                max_band = max(all_bands) if all_bands else 0

                # Pivot each metric separately
                pivoted = data[id_cols].drop_duplicates().copy()

                for metric in metric_cols:
                    pivot_data = data[id_cols + ['band_roi_id', metric]].copy()
                    pivot_wide = pivot_data.pivot_table(index=id_cols, columns='band_roi_id', values=metric, aggfunc='first')
                    pivot_wide.columns = [f'{metric}_{col}' for col in pivot_wide.columns]
                    pivoted = pivoted.merge(pivot_wide.reset_index(), on=id_cols, how='left')

                # Add missing columns for all band+ROI combinations
                for band_id in range(int(max_band) + 1):
                    for roi in all_rois:
                        for metric in metric_cols:
                            col_name = f'{metric}_band_{band_id}_{roi}'
                            if col_name not in pivoted.columns:
                                pivoted[col_name] = None

                data = pivoted

            table = pd.concat([table, data], ignore_index=True)
            checker = analysis_checker(checker, data, analysis)

    if not table.empty:
        table = prefix_to_columns(table, prefix)
        for col in list(table.columns):
            if 'Unnamed' in col or 'lesion_id' in col or 'relaxometry_measure' in col:
                table.drop(col, axis=1, inplace=True)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    # Process T2/T1 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_perivent_t2overt1'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]
        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')
            data.columns = [col.replace('_z_', '_t2overt1_z_') for col in data.columns]

            # Aggregate by band and ROI: pivot to one row per subject/session
            if 'band' in data.columns and 'ROI' in data.columns:
                id_cols = ['subject', 'session']
                band_col = 'band'
                roi_col = 'ROI'
                metric_cols = [col for col in data.columns
                              if col not in id_cols + [band_col, roi_col]]

                # Create band_roi_id column for pivoting
                data['band_roi_id'] = 't2overt1_' + 'band_' + data[band_col].astype(int).astype(str) + '_' + data[roi_col].astype(str)

                # Find all unique combinations
                all_bands = sorted([int(b) for b in data[band_col].dropna().unique()])
                all_rois = sorted(data[roi_col].dropna().unique())
                max_band = max(all_bands) if all_bands else 0

                # Pivot each metric separately
                pivoted = data[id_cols].drop_duplicates().copy()

                for metric in metric_cols:
                    pivot_data = data[id_cols + ['band_roi_id', metric]].copy()
                    pivot_wide = pivot_data.pivot_table(index=id_cols, columns='band_roi_id', values=metric, aggfunc='first')
                    pivot_wide.columns = [f'{metric}_{col}' for col in pivot_wide.columns]
                    pivoted = pivoted.merge(pivot_wide.reset_index(), on=id_cols, how='left')

                # Add missing columns for all band+ROI combinations
                for band_id in range(int(max_band) + 1):
                    for roi in all_rois:
                        for metric in metric_cols:
                            col_name = f'{metric}_band_{band_id}_{roi}'
                            if col_name not in pivoted.columns:
                                pivoted[col_name] = None

                data = pivoted

            table = pd.concat([table, data], ignore_index=True)
            checker = analysis_checker(checker, data, analysis)

    if not table.empty:
        table = prefix_to_columns(table, prefix)
        for col in list(table.columns):
            if 'Unnamed' in col or 'lesion_id' in col or 'relaxometry_measure' in col:
                table.drop(col, axis=1, inplace=True)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    return checker

def run_regions_results(derivatives_path, checker):
    """
    Process regional quantification data (quant_regions_t1 and t2).
    Restructures data from long format (ROI_NAWM column) to wide format.
    Prefix: 'map_reg_'
    Skips if CSV file not found in session folder.
    """
    table = pd.DataFrame()
    prefix = 'map_reg_'

    # Process T1 data
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_regions_t1'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = os.listdir(os.path.join(derivatives_path, analysis))

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            original_data = pd.read_csv(table_path[0])
            new_line = pd.DataFrame()
            df = original_data
            df_columns = [item for item in df.columns if item != 'ROI_NAWM']

            # Pivot: create columns for each ROI-measure combination
            for column in df_columns:
                for item in df["ROI_NAWM"]:
                    new_line[f'{item}_lobe_t1_zscore_{column}'] = df[column][df["ROI_NAWM"] == item].values

            new_line['subject'] = subject.strip('sub-')
            new_line['session'] = session.strip('ses-')
            if 'Unnamed: 0' in new_line.columns:
                new_line.drop('Unnamed: 0', axis=1, inplace=True)
            table = pd.concat([table, new_line], ignore_index=True)
            checker = analysis_checker(checker, new_line, analysis)

    if not table.empty:
        table = table.sort_index(axis=1)
        table = prefix_to_columns(table, prefix)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))

    # Process T2 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_regions_t2'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = os.listdir(os.path.join(derivatives_path, analysis))

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            original_data = pd.read_csv(table_path[0])
            new_line = pd.DataFrame()
            df = original_data
            df_columns = [item for item in df.columns if item != 'ROI_NAWM']

            for column in df_columns:
                for item in df["ROI_NAWM"]:
                    new_line[f'{item}_lobe_t2_zscore_{column}'] = df[column][df["ROI_NAWM"] == item].values

            new_line['subject'] = subject.strip('sub-')
            new_line['session'] = session.strip('ses-')
            if 'Unnamed: 0' in new_line.columns:
                new_line.drop('Unnamed: 0', axis=1, inplace=True)
            table = pd.concat([table, new_line], ignore_index=True)
            checker = analysis_checker(checker, new_line, analysis)

    if not table.empty:
        table = table.sort_index(axis=1)
        table = prefix_to_columns(table, prefix)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
    return checker

def run_tracts_results(derivatives_path, checker):
    """
    Process white matter tract quantification data (quant_wmtracts_t1 and t2).
    Combines lesion statistics and q-values into a single output.
    Prefix: 'map_tra_'
    Skips if CSV files not found in session folder.
    """
    table = pd.DataFrame()
    prefix = 'map_tra_'

    # Process T1 data
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_wmtracts_t1'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = os.listdir(os.path.join(derivatives_path, analysis))

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            # Read lesion statistics
            table_les_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                                    'lesion_stat_in_wm_tracts.csv'))
            if not table_les_path:  # Skip if file not found
                continue

            original_data = pd.read_csv(table_les_path[0])
            new_line_t1 = pd.DataFrame()
            df = original_data
            df_columns = [item for item in df.columns if item != 'tract']

            for item in df["tract"]:
                for column in df_columns:
                    new_line_t1[f'{item}_{column}'] = df[column][df["tract"] == item].values

            new_line_t1['subject'] = subject.strip('sub-')
            new_line_t1['session'] = session.strip('ses-')
            if 'Unnamed: 0' in new_line_t1.columns:
                new_line_t1.drop('Unnamed: 0', axis=1, inplace=True)

            # Read q-values
            table_les_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                                    'qvalues_in_wm_tracts.csv'))
            if not table_les_path:  # Skip if file not found
                continue

            original_data = pd.read_csv(table_les_path[0])
            new_line_t2 = pd.DataFrame()
            df = original_data
            df_columns = [item for item in df.columns if (item != 'tract') & (item != 'ROI')]

            new_line_t2_list = []
            for index, row in df.iterrows():
                new_col_names = [f"{row['tract']}_{row['ROI']}_t1_{col}" for col in df.columns[2:]]
                new_df = pd.DataFrame([row[2:].values], columns=new_col_names)
                new_line_t2_list.append(new_df)

            new_line_t2 = pd.concat(new_line_t2_list, axis=1) if new_line_t2_list else pd.DataFrame()
            new_line_t2['subject'] = subject.strip('sub-')
            new_line_t2['session'] = session.strip('ses-')
            if 'Unnamed: 0' in new_line_t2.columns:
                new_line_t2.drop('Unnamed: 0', axis=1, inplace=True)

            new_line = pd.merge(new_line_t1, new_line_t2, on=['subject', 'session'], how='left')
            table = pd.concat([table, new_line], ignore_index=True)

    if not table.empty:
        table = prefix_to_columns(table, prefix)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
        # Update checker once per analysis (avoid duplicates)
        checker = analysis_checker(checker, table, analysis)

    # Process T2 data (same structure as T1)
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_wmtracts_t2'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = os.listdir(os.path.join(derivatives_path, analysis))

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            table_les_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                                    'lesion_stat_in_wm_tracts.csv'))
            if not table_les_path:  # Skip if file not found
                continue

            original_data = pd.read_csv(table_les_path[0])
            new_line_t1 = pd.DataFrame()
            df = original_data
            df_columns = [item for item in df.columns if item != 'tract']

            for item in df["tract"]:
                for column in df_columns:
                    new_line_t1[f'{item}_{column}'] = df[column][df["tract"] == item].values

            new_line_t1['subject'] = subject.strip('sub-')
            new_line_t1['session'] = session.strip('ses-')
            if 'Unnamed: 0' in new_line_t1.columns:
                new_line_t1.drop('Unnamed: 0', axis=1, inplace=True)

            table_les_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                                    'qvalues_in_wm_tracts.csv'))
            if not table_les_path:  # Skip if file not found
                continue

            original_data = pd.read_csv(table_les_path[0])
            new_line_t2 = pd.DataFrame()
            df = original_data
            df_columns = [item for item in df.columns if (item != 'tract') & (item != 'ROI')]

            new_line_t2_list = []
            for index, row in df.iterrows():
                new_col_names = [f"{row['tract']}_{row['ROI']}_t2_{col}" for col in df.columns[2:]]
                new_df = pd.DataFrame([row[2:].values], columns=new_col_names)
                new_line_t2_list.append(new_df)

            new_line_t2 = pd.concat(new_line_t2_list, axis=1) if new_line_t2_list else pd.DataFrame()
            new_line_t2['subject'] = subject.strip('sub-')
            new_line_t2['session'] = session.strip('ses-')
            if 'Unnamed: 0' in new_line_t2.columns:
                new_line_t2.drop('Unnamed: 0', axis=1, inplace=True)

            new_line = pd.merge(new_line_t1, new_line_t2, on=['subject', 'session'], how='left')
            table = pd.concat([table, new_line], ignore_index=True)

    if not table.empty:
        table = table.sort_index(axis=1)
        table = prefix_to_columns(table, prefix)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
        # Update checker once per analysis (avoid duplicates)
        checker = analysis_checker(checker, table, analysis)

    # Process T2/T1 data
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    analysis_substring = 'quant_wmtracts_t2overt1'
    found = False
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            found = True
            break

    if found:
        subjects = os.listdir(os.path.join(derivatives_path, analysis))

        for subject in subjects:
            sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                        if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                        and not item.endswith('.nii.gz')]

            for session in sessions:
                table_les_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                                        'lesion_stat_in_wm_tracts.csv'))
                if not table_les_path:  # Skip if file not found
                    continue

                original_data = pd.read_csv(table_les_path[0])
                new_line_t1 = pd.DataFrame()
                df = original_data
                df_columns = [item for item in df.columns if item != 'tract']

                for item in df["tract"]:
                    for column in df_columns:
                        new_line_t1[f'{item}_{column}'] = df[column][df["tract"] == item].values

                new_line_t1['subject'] = subject.strip('sub-')
                new_line_t1['session'] = session.strip('ses-')
                if 'Unnamed: 0' in new_line_t1.columns:
                    new_line_t1.drop('Unnamed: 0', axis=1, inplace=True)

                table_les_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                                        'qvalues_in_wm_tracts.csv'))
                if not table_les_path:  # Skip if file not found
                    continue

                original_data = pd.read_csv(table_les_path[0])
                new_line_t2_list = []
                df = original_data
                df_columns = [item for item in df.columns if (item != 'tract') & (item != 'ROI')]

                new_line_t2 = pd.DataFrame()
                for index, row in df.iterrows():
                    new_col_names = [f"{row['tract']}_{row['ROI']}_t2overt1_{col}" for col in df.columns[2:]]
                    new_df = pd.DataFrame([row[2:].values], columns=new_col_names)
                    new_line_t2_list.append(new_df)

                new_line_t2 = pd.concat(new_line_t2_list, axis=1) if new_line_t2_list else pd.DataFrame()
                new_line_t2['subject'] = subject.strip('sub-')
                new_line_t2['session'] = session.strip('ses-')
                if 'Unnamed: 0' in new_line_t2.columns:
                    new_line_t2.drop('Unnamed: 0', axis=1, inplace=True)

                new_line = pd.merge(new_line_t1, new_line_t2, on=['subject', 'session'], how='left')
                table = pd.concat([table, new_line], ignore_index=True)

        if not table.empty:
            table = prefix_to_columns(table, prefix)
            table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
            checker = analysis_checker(checker, table, analysis)

    return checker

def run_mb_results(derivatives_path, checker):
    """
    Process morphobox volumetry data.
    Extracts only columns with 'ml' (volume) or '%' (percentage) measurements.
    Removes columns with all-zero values.
    Prefix: 'vol_'
    Skips if CSV file not found in session folder.
    """
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    prefix = 'vol_'
    analysis_substring = 'morphobox'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session), '*.csv'))
            if not table_path:  # Skip if no CSV found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')
            checker = analysis_checker(checker, data, analysis)
            colmns_to_front = ['subject', 'session'] + [col for col in data.columns
                                                        if col not in ['subject', 'session']]
            data = data[colmns_to_front]
            if 'Unnamed: 0' in data.columns:
                data.drop('Unnamed: 0', axis=1, inplace=True)
            table = pd.concat([table, data], ignore_index=True)

    if not table.empty:
        # Keep only volume/percentage columns
        ml_columns = ['subject', 'session'] + [col for col in table.columns
                                               if 'ml' in col or '%' in col]
        table = table[ml_columns]

        # Remove all-zero columns
        zero_columns = [col for col in table.columns if table[col].eq(0).all()]
        table = table.drop(zero_columns, axis=1)
        table = prefix_to_columns(table, prefix)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
    return checker

def run_lesion_seg_results(derivatives_path, checker):
    """
    Process white matter hyperintensity (wmh) lesion segmentation results.
    Reads JSON output from lesion quantification (lesions_quant_metrics_v4.0.0.json).
    Excludes voxel-level measurements.
    Prefix: 'les_'
    Skips if JSON file not found in session folder.
    """
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    prefix = 'les_'
    analysis_substring = 'wmh'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            # Look for JSON file in cross_results subfolder
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session,
                                               'cross_results'), 'lesions_quant_metrics_v4.0.0.json'))
            if not table_path or not os.path.isfile(table_path[0]):  # Skip if file not found
                continue

            with open(table_path[0]) as file:
                j_son = json.load(file)

            data = pd.DataFrame([j_son])
            data = data.drop(columns=['version'])
            data = data.add_prefix(prefix)
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')
            colmns_to_front = ['subject', 'session'] + [col for col in data.columns
                                                        if col not in ['subject', 'session']]
            data = data[colmns_to_front]
            if 'Unnamed: 0' in data.columns:
                data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
            checker = analysis_checker(checker, data, analysis)
            table = pd.concat([table, data], ignore_index=True)

    if not table.empty:
        # Remove voxel-level columns (keep only aggregate metrics)
        ml_columns = [col for col in table.columns if not "_voxel" in col]
        table = table[ml_columns]

        # Remove all-zero columns
        zero_columns = [col for col in table.columns if table[col].eq(0).all()]
        table = table.drop(zero_columns, axis=1)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
    return checker

def run_concord_results(derivatives_path, checker):
    """
    Process concordance metrics from concord pipeline.
    Reads *_summary.csv files.
    Pivots by con_Level so each level (C2, C3, C4, C5) becomes separate columns with level prefix.
    Example: con_MUCCA_mean_C2, con_MUCCA_mean_C3, etc.
    Prefix: 'con_'
    Skips if summary CSV file not found in session folder.
    """
    table = pd.DataFrame()
    results = os.listdir(derivatives_path)
    prefix = 'con_'
    analysis_substring = 'concord'
    for result in results:
        if analysis_substring in result and os.path.isdir(os.path.join(derivatives_path, result)):
            analysis = result
            break

    subjects = [item for item in os.listdir(os.path.join(derivatives_path, analysis))
                if os.path.isdir(os.path.join(derivatives_path, analysis, item))
                and item.startswith('sub-')]

    for subject in subjects:
        sessions = [item for item in os.listdir(os.path.join(derivatives_path, analysis, subject))
                    if os.path.isdir(os.path.join(derivatives_path, analysis, subject, item))
                    and not item.endswith('.nii.gz')]

        for session in sessions:
            # Find summary CSV
            table_path = glob.glob(os.path.join(os.path.join(derivatives_path, analysis, subject, session),
                                               '*_summary.csv'))
            if not table_path:  # Skip if file not found
                continue
            data = pd.read_csv(table_path[0])
            data['subject'] = subject.strip('sub-')
            data['session'] = session.strip('ses-')

            # Pivot by con_Level: each level gets its own set of columns
            if 'Level' in data.columns:
                id_cols = ['subject', 'session']
                level_col = 'Level'
                value_cols = [col for col in data.columns if col not in id_cols + [level_col]]

                # Create one row per subject/session with all levels as columns
                pivoted_dict = {col: data[col].iloc[0] if len(data) > 0 else None for col in id_cols}

                for level in sorted(data[level_col].unique()):
                    level_data = data[data[level_col] == level]
                    if len(level_data) > 0:
                        level_row = level_data.iloc[0]
                        for col in value_cols:
                            new_col_name = f'{col}_{level}'
                            pivoted_dict[new_col_name] = level_row[col] if col in level_row.index else None

                data = pd.DataFrame([pivoted_dict])

            checker = analysis_checker(checker, data, analysis)
            table = pd.concat([table, data], ignore_index=True)

    if not table.empty:
        if 'Unnamed: 0' in table.columns:
            table.drop('Unnamed: 0', axis=1, inplace=True)
        table = prefix_to_columns(table, prefix)
        table.to_csv(os.path.join(derivatives_path, f'{analysis}.csv'))
    return checker

def missing_analysis_results(df:pd.DataFrame, output_dir:str):
    """
    Optional: Identify and export rows with missing data for each measurement.
    Creates separate text files for each measurement with NaN values.
    """
    measurement_columns = df.columns.difference(['subject', 'session'])

    for measure in measurement_columns:
        nan_rows = df[df[measure].isnull()]
        if not nan_rows.empty:
            result = nan_rows[['subject', 'analysis \n session']]
            if not result.empty:
                output_file_path = f'nan_{measure}.txt'
                result.to_csv(os.path.join(output_dir, output_file_path),
                            sep='\t', index=False, header=False)
                print(f"Processed {len(result)} rows with NaN values in {measure}. "
                      f"Output saved to '{output_file_path}'")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

# User selects derivatives directory
derivatives_path = select_directory()

# Initialize QA checker
checker = pd.DataFrame(columns=['subject', 'session', 'analysis'])

# Process all analysis types in order
print("\nProcessing morphobox results...")
checker = run_mb_results(derivatives_path, checker)

print("Processing lesion segmentation results...")
checker = run_lesion_seg_results(derivatives_path, checker)

print("Processing perilesional results...")
checker = run_perilesional_results(derivatives_path, checker)

print("Processing periventricular results...")
checker = run_perivent_results(derivatives_path, checker)

print("Processing regional results...")
checker = run_regions_results(derivatives_path, checker)

print("Processing white matter tract results...")
checker = run_tracts_results(derivatives_path, checker)

print("Processing concordance results...")
checker = run_concord_results(derivatives_path, checker)

# Create pivot table showing which analyses ran for each subject/session
# Remove any duplicate entries before pivoting
checker = checker.drop_duplicates(subset=['subject', 'session', 'analysis'])
# checker_w = checker.pivot(index=['subject', 'session'], columns='analysis', values='value')

# Optional: Find missing analyses (comment in if needed)
# missing_analysis_results(checker_w, derivatives_path)

# Merge all individual analysis CSVs into one comprehensive table
print("\nMerging all results into unified table...")

# List of analysis files to merge (created by the script)
analysis_patterns = ['morphobox_v', 'wmh_', 'quant_periles_t1_', 'quant_periles_t2_', 'quant_periles_t2overt1_',
                     'quant_perivent_t1_', 'quant_perivent_t2_', 'quant_perivent_t2overt1_',
                     'quant_regions_t1_', 'quant_regions_t2_',
                     'quant_wmtracts_t1_', 'quant_wmtracts_t2_', 'quant_wmtracts_t2overt1_',
                     'concord_v']

# Find files matching analysis patterns
results = os.listdir(derivatives_path)
csv_files = [file for file in results if file.endswith('.csv') and
             any(pattern in file for pattern in analysis_patterns)]

if csv_files:
    print(f"Found {len(csv_files)} analysis files to merge")

    # Start with first file
    table = pd.read_csv(os.path.join(derivatives_path, csv_files[0]))
    print(f"  Started with {csv_files[0]}: {len(table)} rows, {len(table.columns)} columns")

    # Merge remaining files
    for file in csv_files[1:]:
        df = pd.read_csv(os.path.join(derivatives_path, file))
        table = pd.merge(table, df, on=['subject', 'session'], how='left', suffixes=('', '_dup'))
        # Drop duplicate columns ending with _dup
        dup_cols = [col for col in table.columns if col.endswith('_dup')]
        table = table.drop(columns=dup_cols)
        print(f"  Merged {file}: {len(table)} rows, {len(table.columns)} columns")

    # Clean up column names (remove extra spaces, standardize underscores)
    table.columns = (table.columns.str.replace('_ ', '_', regex=True)
                                  .str.replace(' ', '_', regex=True)
                                  .str.replace('__', '_', regex=True))

    # Remove completely empty columns
    empty_cols = [col for col in table.columns if table[col].isna().all()]
    if empty_cols:
        table = table.drop(columns=empty_cols)
        print(f"  Removed {len(empty_cols)} empty columns")

    # Save final merged results
    print("\nSaving final merged results...")
    table.to_csv(os.path.join(derivatives_path, 'results_all.csv'), index=False)
    print(f"✓ Saved: {len(table.columns)} columns × {len(table)} rows")
else:
    print("No analysis CSV files found to merge.")

print("\n✓ Analysis complete!")