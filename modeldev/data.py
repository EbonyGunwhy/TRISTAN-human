import pandas as pd
import numpy as np

########## Ebony's functions (START) ##########
def store_data(data: dict,
               column_list: list) -> dict:
    """Stores info and time curves from csv sheets in dictionary."""
    dictionary = {} # create empty dict
    for col in column_list: # loop through columns (i.e., each ROI)
        # match columns containing only part of phrase (e.g., kidney = kidney-parenchima)
        if data.columns.str.contains(col, case=False, regex=False).any():
            # take matching column name
            column_name = data.columns[data.columns.str.contains(col)][0]
            # store corresponding columns in dict
            dictionary[col] = data[['time', column_name]]
    return dictionary


def twoshot_twoscan_WB(subj):
    """Extracts data and prepares in format ready for WB modelling.
    
    Please note: templated from Steven's 'twoshot_twoscan' function
    and therefore extracts corresponding data for each scan (dyn1 & dyn2).
    However, only dyn1 data is currently used and fitted in
    WB modelling scripts.
    """
    const = pd.read_excel(subj, sheet_name='const')
    const.set_index('name', inplace=True)
    dyn1 = pd.read_excel(subj, sheet_name='dyn1')
    dyn2 = pd.read_excel(subj, sheet_name='dyn2')
    molli1 = pd.read_excel(subj, sheet_name='MOLLI1')
    molli2 = pd.read_excel(subj, sheet_name='MOLLI2')
    molli3 = pd.read_excel(subj, sheet_name='MOLLI3')
    dyn1.sort_values('time', inplace=True)
    dyn2.sort_values('time', inplace=True)
    molli1.sort_values('time', inplace=True)
    molli2.sort_values('time', inplace=True)
    molli3.sort_values('time', inplace=True)
    if 'aorta_valid' not in dyn1:
        dyn1['aorta_valid'] = 1
    if 'liver_valid' not in dyn1:
        dyn1['liver_valid'] = 1
    if 'portal_valid' not in dyn1:
        dyn1['portal_valid'] = 1
    if 'aorta_valid' not in dyn2:
        dyn2['aorta_valid'] = 1
    if 'liver_valid' not in dyn2:
        dyn2['liver_valid'] = 1
    if 'portal_valid' not in dyn2:
        dyn2['portal_valid'] = 1
    t0 = dyn1.time.values[0]
    for molli in [molli1, molli2, molli3]:
        molli['portal-vein'] = molli['portal vein'] # rename to match dyn1 & dyn2
    column_list = dyn1.columns[2:] # Extract columns of interest (i.e., each ROI)
    # Store data in dictionaries
    dyn1_dict, dyn2_dict, molli1_dict, molli2_dict, molli3_dict = [store_data(dyn1, column_list),
                                                                   store_data(dyn2, column_list),
                                                                   store_data(molli1, column_list),
                                                                   store_data(molli2, column_list),
                                                                   store_data(molli3, column_list)]

    time1 = dyn1.time.values-t0
    time2 = dyn2.time.values-t0
    fa1 = dyn1.fa.values
    fa2 = dyn2.fa.values
    return (
        time1, fa1, dyn1_dict,
        time2, fa2, dyn2_dict,
        molli1, molli2, molli3,
        const.at['weight', 'value'], const.at['dose1', 'value'], const.at['dose2', 'value'],
    )
########## Ebony's functions (END) ##########


def oneshot_onescan(subj):

    const = pd.read_excel(subj, sheet_name='const')
    const.set_index('name', inplace=True)
    dyn1 = pd.read_excel(subj, sheet_name='dyn1')
    molli1 = pd.read_excel(subj, sheet_name='MOLLI1')
    molli2 = pd.read_excel(subj, sheet_name='MOLLI2')
    dyn1.sort_values('time', inplace=True)
    molli1.sort_values('time', inplace=True)
    molli2.sort_values('time', inplace=True)
    t0 = dyn1.time.values[0]
    return (
        dyn1.time.values-t0, dyn1.fa.values, dyn1.aorta.values, dyn1.liver.values,
        molli1.time.values[0]-t0, molli1.aorta.values[0], molli1.liver.values[0],
        molli2.time.values[0]-t0, molli2.aorta.values[0], molli2.liver.values[0],
        const.at['weight', 'value'], const.at['dose1', 'value'], 
    )

def twoshot_twoscan(subj):

    const = pd.read_excel(subj, sheet_name='const')
    const.set_index('name', inplace=True)
    dyn1 = pd.read_excel(subj, sheet_name='dyn1')
    dyn2 = pd.read_excel(subj, sheet_name='dyn2')
    molli1 = pd.read_excel(subj, sheet_name='MOLLI1')
    molli2 = pd.read_excel(subj, sheet_name='MOLLI2')
    molli3 = pd.read_excel(subj, sheet_name='MOLLI3')
    dyn1.sort_values('time', inplace=True)
    dyn2.sort_values('time', inplace=True)
    molli1.sort_values('time', inplace=True)
    molli2.sort_values('time', inplace=True)
    molli3.sort_values('time', inplace=True)
    if 'aorta_valid' not in dyn1:
        dyn1['aorta_valid'] = 1
    if 'liver_valid' not in dyn1:
        dyn1['liver_valid'] = 1
    if 'portal_valid' not in dyn1:
        dyn1['portal_valid'] = 1
    if 'aorta_valid' not in dyn2:
        dyn2['aorta_valid'] = 1
    if 'liver_valid' not in dyn2:
        dyn2['liver_valid'] = 1
    if 'portal_valid' not in dyn2:
        dyn2['portal_valid'] = 1
    t0 = dyn1.time.values[0]
    return (
        dyn1.time.values-t0, dyn1.fa.values, dyn1.aorta.values, dyn1.liver.values, dyn1['portal-vein'].values, 
        dyn1.aorta_valid.values, dyn1.liver_valid.values, dyn1.portal_valid.values,
        dyn2.time.values-t0, dyn2.fa.values, dyn2.aorta.values, dyn2.liver.values, dyn2['portal-vein'].values, 
        dyn2.aorta_valid.values, dyn2.liver_valid.values, dyn2.portal_valid.values,
        molli1.time.values[0]-t0, molli1.aorta.values[0], molli1.liver.values[0], molli1['portal vein'].values[0],
        molli2.time.values[0]-t0, molli2.aorta.values[0], molli2.liver.values[0], molli2['portal vein'].values[0],
        molli3.time.values[0]-t0, molli3.aorta.values[0], molli3.liver.values[0], molli3['portal vein'].values[0],
        const.at['weight', 'value'], const.at['dose1', 'value'], const.at['dose2', 'value'],
    )

def oneshot_twoscan(subj):

    (   time1, fa1, aorta1, liver1, 
        aorta_valid1, liver_valid1, 
        time2, fa2, aorta2, liver2,
        aorta_valid2, liver_valid2, 
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2) = twoshot_twoscan(subj)
    i1 = np.nonzero(time2 < time2[0] + 10*60)[0]
    return (
        time1, fa1, aorta1, liver1,
        aorta_valid1, liver_valid1, 
        time2[i1], fa2[i1], aorta2[i1], liver2[i1], 
        aorta_valid2[i1], liver_valid2[i1], 
        T1time1, T1aorta1, T1liver1,
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
         weight, dose1, 
    )

def twoshot_onescan(subj):

    (   time1, fa1, aorta1, liver1, 
        aorta_valid1, liver_valid1, 
        time2, fa2, aorta2, liver2,
        aorta_valid2, liver_valid2, 
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2) = twoshot_twoscan(subj)
    return (
        time2, fa2, aorta2, liver2, 
        aorta_valid2, liver_valid2, 
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2, 
    )

