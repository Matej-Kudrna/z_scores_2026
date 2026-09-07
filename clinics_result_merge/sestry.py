import os.path

import pandas as pd
import numpy as np

def validate_and_process_rc(df, rc_column_name='RC'):
    """
    Processes the DataFrame to validate the RC numbers and separate valid and invalid ones.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    rc_column_name (str): The name of the column containing the RC numbers.

    Returns:
    pd.DataFrame, pd.DataFrame: DataFrames containing valid and invalid RC numbers.
    """
    # Validations for input DataFrame
    required_columns = [rc_column_name]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain the following columns: {required_columns}")

    # Drop rows where specified columns are NaN
    df = df.dropna(subset=required_columns)

    # Clean the RC numbers column
    df[rc_column_name] = df[rc_column_name].astype(str).str.replace(r'[/. x]', '', regex=True)

    df_final, df_final_check = pd.DataFrame(), pd.DataFrame()

    # Process RC numbers of 10 characters
    df_rc_10 = df[df[rc_column_name].str.len() == 10]
    df_final, df_final_check = update_df_finals(rc_column_name, df_rc_10, df_final, df_final_check)

    # Process RC numbers of 9 characters
    df_rc_9 = df[df[rc_column_name].str.len() == 9]

    # Check if there are any valid RC numbers before proceeding
    if not df_rc_9.empty:
        df_rc_9['modulo_check'] = df_rc_9[rc_column_name].apply(lambda x: int(x) % 11 == 0)

        # Separate valid and invalid RC numbers based on modulo check
        df_rc_9_valid = df_rc_9[df_rc_9['modulo_check']]
        df_rc_9_invalid = df_rc_9[~df_rc_9['modulo_check']]

        df_final = pd.concat([df_final, df_rc_9_invalid])

        # Further processing of valid 9 characters
        possible_months = {'01', '02', '03', '04', '05', '06', '07', '08', '09',
                           '10', '11', '12', '51', '52', '53', '54', '55',
                           '56', '57', '58', '59', '60', '61', '62'}

        df_rc_9_valid['Test_RC'] = df_rc_9_valid[rc_column_name].apply(
            lambda x: len(x) > 1 and x[2:4] in possible_months
        )

        # Processing less than 9 character RC numbers
    df_rc_less9 = df[df[rc_column_name].str.len() < 9]
    if not df_rc_less9.empty:
        df_rc_less9 = pd.concat([df_rc_9_valid[~df_rc_9_valid['Test_RC']].drop('Test_RC', axis=1), df_rc_less9])
        df_rc_less9[rc_column_name] = df_rc_less9[rc_column_name].str.zfill(10)
        df_rc_less9 = df_rc_less9[~df_rc_less9[rc_column_name].str.match(r'^000000')]

        df_final, df_final_check = update_df_finals(rc_column_name, df_rc_less9, df_final, df_final_check)
        df_final_check = pd.concat([df_rc_9_valid[df_rc_9_valid['Test_RC']], df_final_check])

    return df_final, df_final_check


def update_df_finals(rc_column_name, df_in, df_final, df_final_check):
    """A simplified version of your original function to update final DataFrames."""
    if rc_column_name not in df_in.columns:
        raise ValueError(f"Column '{rc_column_name}' is not in DataFrame.")

    df_in['modulo_check'] = df_in[rc_column_name].apply(lambda x: int('0' + x) % 11 == 0)

    df_final = pd.concat([df_in[df_in['modulo_check']].drop('modulo_check', axis=1), df_final])
    df_final_check = pd.concat([df_in[~df_in['modulo_check']].drop('modulo_check', axis=1), df_final_check])

    return df_final, df_final_check


if __name__ == '__main__':
    dir_path = r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA - Copy\magnims_project_39"
    table_path = r"C:\Users\mri-admin\Všeobecná fakultní nemocnice v Praze\DATA - Copy\magnims_project_39\SESTRY Seznam pacientu 21_2_2024.xlsx"

    one_a = pd.read_excel(table_path,
                          sheet_name=r'Ia noviCIS_', dtype={'Rodne_cislo': str})
    one_b = pd.read_excel(table_path,
                          sheet_name=r'Ib, III NMOSD_Alena', dtype={'Rodne_cislo': str})
    one_c = pd.read_excel(table_path,
                          sheet_name=r'Ic RS_ataka_ ', dtype={'Rodne_cislo': str})
    two = pd.read_excel(table_path,
                        sheet_name=r'II stariCIS_FEMS_', dtype={'Rodne_cislo': str})
    four = pd.read_excel(table_path,
                         sheet_name=r'IV Progres_Alena', dtype={'Rodne_cislo': str})
    five = pd.read_excel(table_path,
                         sheet_name=r'V Likvor_RS_Zaneta', dtype={'Rodne_cislo': str})
    six = pd.read_excel(table_path,
                        sheet_name=r'VI Likvor_Neuro_Zaneta', dtype={'Rodne_cislo': str})
    #tady nejsou vytrideni, udajne jsou ve skupinach zarazeni spravne proto pokaracujeme bez prvni tabulky likvor
    # likvor = pd.read_excel(table_path,
    #                        sheet_name=r'VSECHNY Lumbalky', dtype={'Rodne_cislo': str})

    combined_df = pd.DataFrame()
    check_rc = pd.DataFrame()
    all_cohorts = [one_a, one_b, one_c, two, four, five, six]
    all_cohorts_name = ['Ia', 'Ib', 'Ic', 'II', 'IV', 'V', 'VI']
    for i, var in enumerate(all_cohorts):
        print(f'\n {all_cohorts_name[i]} \n')
        var['cohort'] = all_cohorts_name[i]
        rc_ok, rc_err = validate_and_process_rc(var, 'Rodne_cislo')
        combined_df = pd.concat([combined_df, rc_ok])
        check_rc = pd.concat([check_rc, rc_err])

    skupiny = combined_df[['Rodne_cislo', 'cohort']]
    skupiny.to_pickle(os.path.join(dir_path, 'sestry_rc_skupiny_dce.pkl'))
    combined_df.to_pickle(os.path.join(dir_path, 'sestry_all.pkl'))
    skupiny['Rodne_cislo'].to_csv('skupiny_mrexam.txt', sep='\t', index=False, header=False)
