import os
from pathlib import Path
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data
import curvefit
from models import WB

# HELPER FUNCTIONS
def make_dir(path: Path
             ) -> None:
    """Creates directory if it does not exist."""
    if not os.path.exists(path):
        os.mkdir(path)


def plot_curves(x: np.ndarray,
                y_obs: np.ndarray,
                y_sim: np.ndarray,
                ROI: str,
                resultspath: Path,
                prefix: str) -> None:
    """Plots simulated vs. observed MRI signals."""
    g = sns.scatterplot(x=x, y=y_obs, label=f"{ROI} observed")
    g = sns.lineplot(x=x, y=y_sim, label=f"{ROI} simulated", color='r')
    #g.set(xlim=(0,90))
    #g.set(ylim=(0,None))
    g.set_title(f"SIMULATED VS. OBSERVED MRI SIGNALS \n Region of interest = {ROI}", weight='bold')
    g.set_xlabel("Time [secs]", weight='bold')
    g.set_ylabel("\u0394 $S$ [a.u.]", weight='bold')
    sns.despine(offset=10, trim=True)
    #plt.show()
    plt.savefig(fname=os.path.join(resultspath, prefix+'_curveplot.png'))
    plt.close()

# MAIN
def main(datapath: Path,
         resultspath: Path) -> None:
    """Main function - fits all subject data to WB model."""
    output_file = os.path.join(resultspath, 'parameters.csv')
    output = pd.DataFrame(columns=['subject','visit','name','value','unit'])
    useHeader = True # include dataframe header when saving to csv
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath)[:1]:
            print(f"subject {s[:3]} visit {visit}: Obtaining data")
            subj = os.path.join(visitdatapath, s)
            (time1, fa1, dyn1_dict,
             time2, fa2, dyn2_dict,
             molli1, molli2, molli3,
             weight, dosage1, dosage2) = data.twoshot_twoscan_WB(subj)
            
            # Create time curve data for 'body' by summing muscle & vertebra ROIs
            dyn1_dict['body'] = dyn1_dict['fat'] # create dictionary entry with same layout
            dyn1_dict['body']['body'] = np.sum(np.array([dyn1_dict['vertebra']['vertebra'],
                                                         dyn1_dict['muscle']['muscle']]),
                                                         axis=0)
            dyn1_dict['body'].drop(columns='fat', inplace=True) # drop irrelevant column

            # Fixed parameter inputs
            constants = {
                # Gadoxetate injection
                'conc': 0.25, # mmol/mL
                'rate': 1, # mL/sec
                # MRI scanning parameters
                'TR': 3.71/1000.0, # Repetition time (sec)
                # Biological parameters
                'Hct': 0.45, # hematocrit
                'r_bl': 6.4, # relaxivity of blood [Hz/mM]
                'rh': 9.8, # relaxivity of blood [Hz/mM]
                }
            
            dose = weight*dosage1 # mL
            R10Aorta = (0.52 * constants['Hct'] + 0.38) # precontrast relaxation [s-1]

            # Flow (F) ~ Cardiac Output (CO) [mL/sec]
            FB = 6500/60 # total blood flow (cardiac output 35-y/o male) [mL blood/sec (6500ml/min)]
            #Fb = FB/(VHeart + VLungs) # total blood flow [ml blood/sec/ml tissue]
            # Plasma Flow = Total Blood Flow × (1 - Hct)
            Fp = ((1-constants['Hct'])*FB)/1000 # [L plasma/sec] (division by 1000 for mL->L)
            # t0 = baseline ~ 1030 seconds (pre-contrast)
            t0 = 78 # secs; length of zero conc before step function
            iterations = 40 # number of blood circulations through system
            # Internal time resolution & acquisition time
            dt = 1 # sec
            # step function
            # 0, 0.02, 0
            duration_inj = dose/constants['rate'] # length of step function

            # Estimated paramter inputs
            S0Aorta, S0Liver, S0Kidneys, S0Body = [5000, 650, 2500, 850]
            R10Aorta, R10Liver, R10Kidneys, R10Body = [0.2, 6.5, 0.55, 4]
            params = {'THLu': 9,
                      'Tgut': 6,
                      'E1': 0.255,
                      'TeL': 322,
                      'Th': 1000,
                      'Eh': 0.04,
                      'E2': 0.3,
                      'TpG': 70,
                      'Tt': 10,
                      'Et': 0.04,
                      'Tp': 10,
                      'Te': 3,
                      'Erb': 0.7}

            #cinj = (conc*rate)/Fp # inj = IV injection # (mmol/ml * ml/s) / ml/s = mmol/mL = M => 1000*cinj = mM
            # Jinj = J0max = cinj*Fp = Fp*(conc*rate)/Fp = conc*rate
            J0max = constants['conc']*constants['rate'] # mmol/s = mM/s

            # Simulate signals
            print(f"subject {s[:3]} visit {visit}: Simulating signals")
            AllSignals = WB.simulate_signals(time1,
                                             J0max,
                                             t0,
                                             duration_inj,
                                             constants['TR'],
                                             fa1[0],
                                             constants['r_bl'],
                                             constants['rh'],
                                             Fp,
                                             R10Aorta,
                                             R10Liver,
                                             R10Kidneys,
                                             R10Body,
                                             S0Aorta,
                                             S0Liver,
                                             S0Kidneys,
                                             S0Body,
                                             params,
                                             iterations=40,
                                             dt=0.1)
            
            # Concatentate observed signal data into one array
            ObservedSignals = np.concatenate([dyn1_dict['aorta'].iloc[:,-1],
                                              dyn1_dict['liver'].iloc[:,-1],
                                              dyn1_dict['kidney'].iloc[:,-1],
                                              dyn1_dict['body'].iloc[:,-1]])
            
            # Fit observed data to whole body model
            print(f"subject {s[:3]} visit {visit}: Fitting data")
            variables = curvefit.fit_WBmodel(time1,
                                             ObservedSignals,
                                             J0max,
                                             t0,
                                             duration_inj,
                                             constants['TR'],
                                             fa1[0],
                                             constants['r_bl'],
                                             constants['rh'],
                                             Fp,
                                             R10Aorta,
                                             R10Liver,
                                             R10Kidneys,
                                             R10Body,
                                             S0Aorta,
                                             S0Liver,
                                             S0Kidneys,
                                             S0Body,
                                             params,
                                             iterations,
                                             dt)
            # Save fitted parameters in csv file    
            output['value'] = variables
            output['name'] = ['R10Aorta', 'R10Liver', 'R10Kidneys', 'R10Body', 'S0Aorta', 'S0Liver', 'S0Kidneys', 'S0Body'] + list(params.keys())
            output['unit'] = ['1/sec', '1/sec', '1/sec', '1/sec',
                              'a.u.', 'a.u.', 'a.u.', 'a.u.',
                              'sec', 'sec', 'unitless',
                              'sec', 'sec', 'unitless', 'unitless',
                              'sec', 'sec', 'unitless',
                              'sec', 'sec', 'unitless']
            output['subject'] = s[:3]
            output['visit'] = visit
            print(f"subject {s[:3]} visit {visit}: Saving parameter estimates")
            output.to_csv(output_file, index=False, mode="a", header = useHeader)
            useHeader = False # stop appending dataframe header when saving to csv in loop

            params_fit = params # make copy of parameter dictionary to update with fitted
            # update parameter estimates with fitted parameters
            for key, element in zip(params_fit, variables[-len(params_fit):]):
                params_fit[key] = element
            [R10Aorta_fit, R10Liver_fit, R10Kidneys_fit, R10Body_fit] = variables[:4]
            [S0Aorta_fit, S0Liver_fit, S0Kidneys_fit, S0Body_fit] = variables[4:8]

            # Re-simulate signals using fitted parameter inputs
            print(f"subject {s[:3]} visit {visit}: Re-simulating signals using fitted parameters")
            FittedSignals = WB.simulate_signals(time1,
                                                J0max,
                                                t0,
                                                duration_inj,
                                                constants['TR'],
                                                fa1[0],
                                                constants['r_bl'],
                                                constants['rh'],
                                                Fp,
                                                R10Aorta_fit,
                                                R10Liver_fit,
                                                R10Kidneys_fit,
                                                R10Body_fit,
                                                S0Aorta_fit,
                                                S0Liver_fit,
                                                S0Kidneys_fit,
                                                S0Body_fit,
                                                params_fit,
                                                iterations=40,
                                                dt=0.1)
            
            # Split concatenated fitted signals by ROI using index notation
            series_length = int(len(FittedSignals)/4) # length of one ROI series
            Saorta = FittedSignals[:series_length] # fitted aorta
            Sliver = FittedSignals[series_length:2*series_length] # fitted liver
            Skidney = FittedSignals[2*series_length:3*series_length] # fitted kidney
            Sbody = FittedSignals[3*series_length:4*series_length] # fitted rest of body

            # Plot observed vs. fitted MRI signals for each ROI & save
            print(f"subject {s[:3]} visit {visit}: Plotting time curves")
            for ROI in ['aorta', 'liver', 'kidney', 'body']:
                prefix=f"{s[:3]}_{visit}_{ROI}"
                plot_curves(time1,
                            dyn1_dict[ROI].iloc[:,-1],
                            locals()['S' + ROI],
                            ROI,
                            resultspath,
                            prefix)
    print("Done!")

if __name__ == '__main__':

    start = time.time()

    filepath = os.path.dirname(__file__)
    datapath = os.path.join(filepath, 'data')
    resultspath = os.path.join(filepath, 'results_WB')
    make_dir(resultspath)

    main(datapath, resultspath)
    
    print('Calculation time (mins): ', (time.time()-start)/60)
