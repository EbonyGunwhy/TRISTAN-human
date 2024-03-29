"""
Python scripts for modelling of the human TRISTAN data. 

Not optimised for speed so running very slowly now. 
This approach will too slow for pixel-by-pixel analysis.

To run this as is - download and extract the attachment and then run "main.py". 
This will automatically regenerate the "results" folder.

If you want to run it on other data - the data need to be 
in the same format as the csv files in the folder "sourcedata". 
They are read and formatted by functions in the module data.py. 

The models are defined in models_1Ktrans.py (one Ktrans value), or models_3Ktrans.py 
(allowing for 3 Ktrans values). They use some functions from dcemri.py 
and the CurveFit class in curvefit.py, which is just a convenience wrapper for scipy's curve_fit function. 
"""

import os
import pandas as pd

import data

from models_Aorta import(
    AORTAPARS, 
    AortaOneShotOneScan, 
    AortaTwoShotTwoScan,
    AortaOneShotTwoScan, 
    AortaTwoShotOneScan,  
)
from models_3Ktrans import(
    LIVERPARS,
    LiverOneShotOneScan, 
    LiverTwoShotTwoScan,
    LiverOneShotTwoScan, 
    LiverTwoShotOneScan,  
)


datafolder = 'sourcedata'
resultsfolder = 'fit_all_protocols'

filepath = os.path.dirname(__file__)
datapath = os.path.join(filepath, datafolder)
subject = [f for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]
subject = [s.split('.')[0] for s in subject]


def oneshot_onescan(subj, path, show):

    (time1, fa1, aorta1, liver1, 
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        weight, dose1) = data.oneshot_onescan(subj)

    aorta = AortaOneShotOneScan()
    aorta.weight = weight
    aorta.dose = dose1
    aorta.set_xy(time1, aorta1)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.set_R11(T1time2, 1000.0/T1aorta2)
    aorta.estimate_p()
    aorta.fit_p()
    aorta.plot_fit(save=True, show=show, path=path)
    aorta.export_p(path=path)

    liver = LiverOneShotOneScan(aorta)
    
    liver.set_xy(time1, liver1)
    liver.set_R10(T1time1, 1000.0/T1liver1)
    liver.set_R11(T1time2, 1000.0/T1liver2)
    liver.estimate_p()
    liver.fit_p()
    liver.plot_fit(save=True, show=show, path=path)
    liver.export_p(path=path)

    return aorta.p, liver.p

def twoshot_twoscan(subj, path, show):

    (   time1, fa1, aorta1, liver1, 
        time2, fa2, aorta2, liver2,
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2) = data.twoshot_twoscan(subj)

    aorta = AortaTwoShotTwoScan()
    aorta.weight = weight
    aorta.dose = [dose1, dose2]
    aorta.set_x(time1, time2)
    aorta.set_y(aorta1, aorta2)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.set_R11(T1time2, 1000.0/T1aorta2)
    aorta.set_R12(T1time3, 1000.0/T1aorta3)
    aorta.estimate_p()
    aorta.fit_p()
    aorta.plot_fit(save=True, show=show, path=path)
    aorta.export_p(path=path)

    liver = LiverTwoShotTwoScan(aorta)
    liver.set_x(time1, time2)
    liver.set_y(liver1, liver2)
    liver.set_R10(T1time1, 1000.0/T1liver1)
    liver.set_R11(T1time2, 1000.0/T1liver2)
    liver.set_R12(T1time3, 1000.0/T1liver3)
    liver.estimate_p()
    liver.fit_p()
    liver.plot_fit(save=True, show=show, path=path)
    liver.export_p(path=path)

    return aorta.p, liver.p

def oneshot_twoscan(subj, path, show):

    (   time1, fa1, aorta1, liver1, 
        time2, fa2, aorta2, liver2,
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1) = data.oneshot_twoscan(subj)

    aorta = AortaOneShotTwoScan()
    aorta.weight = weight
    aorta.dose = dose1
    aorta.set_x(time1, time2)
    aorta.set_y(aorta1, aorta2)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.set_R11(T1time2, 1000.0/T1aorta2)
    aorta.set_R12(T1time3, 1000.0/T1aorta3)
    aorta.estimate_p()
    aorta.fit_p()
    aorta.plot_fit(save=True, show=show, path=path)
    aorta.export_p(path=path)

    liver = LiverOneShotTwoScan(aorta)
    liver.set_x(time1, time2)
    liver.set_y(liver1, liver2)
    liver.set_R10(T1time1, 1000.0/T1liver1)
    liver.set_R11(T1time2, 1000.0/T1liver2)
    liver.set_R12(T1time3, 1000.0/T1liver3)
    liver.estimate_p()
    liver.fit_p()
    liver.plot_fit(save=True, show=show, path=path)
    liver.export_p(path=path)

    return aorta.p, liver.p

def twoshot_onescan(subj, path, show):

    (   time2, fa2, aorta2, liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2) = data.twoshot_onescan(subj)

    aorta = AortaTwoShotOneScan()
    aorta.weight = weight
    aorta.dose = [dose1, dose2]
    aorta.set_x(time2)
    aorta.set_y(aorta2)
    aorta.set_R12(T1time3, 1000.0/T1aorta3)
    aorta.estimate_p()
    aorta.fit_p()
    aorta.plot_fit(save=True, show=show, path=path)
    aorta.export_p(path=path)

    liver = LiverTwoShotOneScan(aorta)
    liver.set_x(time2)
    liver.set_y(liver2)
    liver.set_R12(T1time3, 1000.0/T1liver3)
    liver.estimate_p()
    liver.fit_p()
    liver.plot_fit(save=True, show=show, path=path)
    liver.export_p(path=path)

    return aorta.p, liver.p



def run_all():

    show = False

    for s in subject:

        subj = os.path.join(filepath, datafolder, s+'.xlsx')
        path = os.path.join(filepath, resultsfolder, s)

        ap11, lp11 = oneshot_onescan(subj, path, show)
        ap22, lp22 = twoshot_twoscan(subj, path, show)
        ap12, lp12 = oneshot_twoscan(subj, path, show)
        ap21, lp21 = twoshot_onescan(subj, path, show)

        cols = ['symbol', '2shot2scan', '1shot1scan', '1shot2scan', '2shot1scan2']   

        pars = AORTAPARS
        ap = []
        for p in pars:
            row = [p, ap22.value[p], ap11.value[p], ap12.value[p], ap21.value[p]]
            ap.append(row)
        ap = pd.DataFrame(ap, columns=cols)
        ap.set_index('symbol', inplace=True)
        save_file = os.path.join(path, 'summary_aorta.csv')
        try:
            ap.to_csv(save_file)
        except:
            pass

        pars = LIVERPARS
        lp = []
        for p in pars:
            row = [p, lp22.value[p], lp11.value[p], lp12.value[p], lp21.value[p]]
            lp.append(row)
        lp = pd.DataFrame(lp, columns=cols)
        lp.set_index('symbol', inplace=True)
        save_file = os.path.join(path, 'summary_liver.csv')
        try:
            lp.to_csv(save_file)
        except:
            pass

 
if __name__ == "__main__":

    run_all()

    print('DONE!!')