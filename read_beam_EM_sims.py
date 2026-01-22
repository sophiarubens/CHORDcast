import numpy as np
import pandas as pd

"""
Aditya: The CST output files have the theta values ranging from -180 to 180 and phi 
values from -90 to 90 which is not ideal for plotting. This code converts them to  
the usual ranges (0 to 180 theta, 0 to 360 phi) like a 2D grid.
"""

freq = np.array([0.4, 0.6, 0.8])
beam_sim_directory="/Users/sophiarubens/Downloads/research/code/pipeline/fiducial_beams_from_Aditya/"

def gen_power_pattern_for_box(xxx,yyy,zzz):
    # need to know more about the shape of what you import directly from CST
    pass

def read_sim_beam(filename):
    
    df = pd.read_table(filename, skiprows=[0, 1,], sep='\s+', names=['theta', 'phi', 'AbsE', 'AbsCr', 'PhCr', 'AbsCo', 'PhCo', 'AxRat'])
    ntheta = df.theta.abs()
    nphi = df.phi.copy()
    nphi[df.theta < 0] += 180
    ntheta[df.theta == -180] = 0.
    nphi[nphi < 0] += 360

    ndf = df.copy()
    ndf['ntheta'] = ntheta
    ndf['nphi'] = nphi
    ndf = ndf.query('phi != -90')
    ndf = ndf.sort_values(by=['ntheta', 'nphi'], ignore_index=True)
    ndf = ndf.query('ntheta < 90')   # Read theta values within ranges 0 to 90  (or) 0 to 180
    
    ndf.loc[ndf.ntheta == 0] = ndf.query('theta == 0 and phi == 0').values
    ndf.loc[ndf.ntheta == 0, 'nphi'] = ndf.loc[ndf.ntheta == 1]['nphi'].values

    Nth = len(ndf.ntheta.unique())
    abs_E = ndf.AbsE.values.reshape((Nth, -1)) 
    non_log = (10**(abs_E/10))  # Converting dB to watts
    theta = ndf.ntheta.values.reshape((Nth, -1))
    phi = ndf.nphi.values.reshape((Nth, -1))
    
    abs_E_1=abs_E/abs_E.max()  # Normalise the power pattern
        
    return np.array(abs_E_1)  # A 3D array with theta, phi and frequency

fname="farfield (f=0.3) [1]_efield.txt"
test_300_MHz_beam=read_sim_beam(beam_sim_directory+"CHORD_fiducial_farfield/"+fname)

print("test_300_MHz_beam.shape=",test_300_MHz_beam.shape)