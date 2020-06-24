"""
getsample.py contains functions to get a subsample of the APOGEE data.

Natalie Price-Jones, UofT, 2020
"""
import numpy as np
import numpy.lib.recfunctions as rfn
import apogee.tools.read as apread
from apogee.tools.path import change_dr

def get_apogee(dr='16',use_astroNN=True):
    """

    dr:         dr to select.
    astronn:    boolean flag for whether to use astroNN abundances, defaults
                to ASPCAP abundances or not.

    Returns APOGEE allStar file without duplicates.
    """
    dr = str(dr)
    # Change to relevant data release
    change_dr(dr)

    # Only use astroNN values if they are available for this data release.
    if use_astroNN:
        if int(dr)<14:
            use_astroNN=False
        elif int(dr)>=14:
            allStar=apread.allStar(rmcommissioning=True,
                                   rmdups=False,
                                   use_astroNN=True)
    if not use_astroNN:
        allStar=apread.allStar(rmcommissioning=True,
                               rmdups=False,
                               use_astroNN=False)
    # Manually remove duplicates
    apids,inds=np.unique(allStar['APOGEE_ID'],return_index=True)
    return allStar[inds]


def gen_keys_condition(allStar,elems,dr='16',uplim=0.05,downlim=-1000,
                       snrlim=50,gerrlim=0.2,terrlim=100,tuplim=5000,
                       tlolim=3500,guplim=4,glolim=0):
    """

    allStar:        numpy structured array or pandas dataframe with the
                    following keys:
                    'SNR', 'TEFF', 'LOGG', 'TEFF_ERR', 'LOGG_ERR', as well as
                    abundance ratios and their errors, e.g. ['C_H',C_H_ERR']
    elems:          list of elements on which to execute cuts, e.g. ['C','N']
    dr:             data release of the dataset
    uplim:          upper limit on element uncertainties. can be dictionary
                    with different uppers for each element.
    downlim:        lower limit on element uncertainties (used to cut -9999
                    mask used to mark bad data in APOGEE)
    snrlim:         lower limit on signal to noise ratio
    gerrlim:        upper limit on logg uncertainties
    terrlim:        upper limit on teff uncertainties
    tuplim:         upper limit on teff
    tlolim:         lower limit on teff
    guplim:         upper limit on logg
    glolim:         lower limit on logg

    Returns original data, indexes of 'good' stars, and keys for the
    elements listed.
    """
    elems = [elem.upper() for elem in elems]
    # Do prelimiary cut on surface gravity uncertainty, temperature uncertainty,
    # signal to noise ratio, temperature, and surface gravity.
    good = (allStar['LOGG_ERR']<gerrlim) & (allStar['TEFF_ERR']<terrlim)\
           & (allStar['SNR']>snrlim) & (allStar['TEFF']>tlolim)\
           & (allStar['TEFF']<tuplim) & (allStar['LOGG']>glolim)\
           & (allStar['LOGG']<guplim)
    # Create element keys, and convert abundance ratios to be with respect to
    # iron (FE) instead of hydrogen (H), except for iron in data releases
    # earlier than DR 13.
    keys = []
    for elem in elems:
        if elem != 'FE':
            suff = 'FE'
            if int(dr) < 13:
                allStar[f'{elem}_H']-=allStar['FE_H']
                newerr = np.sqrt(allStar[f'{elem}_H_ERR']**2\
                                 +allStar['FE_H_ERR']**2)
                allStar[f'{elem}_H_ERR'] = newerr
                newnames = {f'{elem}_H':f'{elem}_FE',
                            f'{elem}_H_ERR':f'{elem}_FE_ERR'}
                if isinstance(allStar,np.ndarray):
                    allStar=rfn.rename_fields(allStar,newnames)
                elif isinstance(allStar,pd.DataFrame):
                    allStar.rename(columns=newnames)
        elif elem == 'FE':
            suff = 'H'

        # Add to list of element keys
        keys.append(f'{elem}_{suff}')
        #
        if isinstance(uplim,float):
            good = good & \
                   (allStar[f'{elem}_{suff}_ERR'] < uplim) & \
                   (allStar[f'{elem}_{suff}_ERR'] > downlim)
        elif isinstance(uplim,dict):
            good = good & \
                   (allStar[f'{elem}_{suff}_ERR'] < uplim[elem]) & \
                   (allStar[f'{elem}_{suff}_ERR'] > downlim)
    # Return initial data file, boolean array with True where stars satisfy
    # our conditions, and the keys for the elements in this data release.
    return allStar,good,keys
