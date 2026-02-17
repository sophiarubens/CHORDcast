import numpy as np
import pandas as pd
from scipy.interpolate import griddata as gd
from forecasting_pipeline import per_antenna
import time

"""
CST output has
    * theta in [-180,180]
    * phi in [-90, 90].
translate before continuing to simulate!
"""

pi=np.pi
tol=5 # same as in forecasting_pipeline for now
# freq = np.array([0.4, 0.6, 0.8])
beam_sim_directory="/Users/sophiarubens/Downloads/research/code/pipeline/fiducial_beams_from_Aditya/CHORD_Fiducial_Farfield/"


def translate_sim_beam_slice(filename):
    # read in both polarizations
    df = pd.read_table(filename, skiprows=[0, 1,], sep='\s+', 
                       names=['theta', 'phi', 'AbsE', 'AbsCr', 'PhCr', 'AbsCo', 'PhCo', 'AxRat'])
    
    # translate coordinates
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
    ndf = ndf.query('ntheta < 90') # get all the theta values. might be too slow and overkill (realistically the flat-sky appx breaks down way before theta=90 deg)
    
    ndf.loc[ndf.ntheta == 0] = ndf.query('theta == 0 and phi == 0').values
    ndf.loc[ndf.ntheta == 0, 'nphi'] = ndf.loc[ndf.ntheta == 1]['nphi'].values

    # establish non-log values
    power=10**(ndf.AbsE.values/10)
    theta_deg=ndf.ntheta.values
    theta=theta_deg*pi/180
    phi_deg=ndf.nphi.values
    phi=phi_deg*pi/180
    sky_angle_x=theta*np.cos(phi)
    sky_angle_y=theta*np.sin(phi)
    sky_angle_points=np.array([sky_angle_x,sky_angle_y]).T
    return sky_angle_points,power

Npix=256
freqs=np.arange(0.32,0.66,0.02) # could go up to 0.98 at this resolution
nu_ref=freqs[0]
uv_manager=per_antenna(mode="pathfinder",nu_ctr=nu_ref)
uv_synth_freq_agnostic=uv_manager.uv_synth*nu_ref
all_ungridded_u=uv_synth_freq_agnostic[:,0,:]
all_ungridded_v=uv_synth_freq_agnostic[:,1,:]
uvmagmax_freq_agnostic=tol*np.max([np.max(np.abs(all_ungridded_u)),
                                   np.max(np.abs(all_ungridded_v))])
uvmagmin_freq_agnostic=2*uvmagmax_freq_agnostic/Npix
# the sky angle I'm used to calling theta here I'll call alpha to avoid confusion with the Ludwig-III (spherical) theta
alphamax_freq_agnostic=1/uvmagmin_freq_agnostic # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
    
def box_from_simulated_beams(freqs,
                             f_n_head,pol1_identifier,pol2_identifier,f_n_tail,
                             custom_outname):
    N_LoS=len(freqs)
    ti=time.time()
    t=np.zeros(N_LoS)
    for i,freq in enumerate(freqs):
        sky_angle_points,uninterp_slice_pol1=translate_sim_beam_slice(f_n_head+str(np.round(freq,2))+
                                                                      str(pol1_identifier)+f_n_tail) # we know both polarizations will be sampled at the same (theta,phi)
        _,               uninterp_slice_pol2=translate_sim_beam_slice(f_n_head+str(np.round(freq,2))+
                                                                      str(pol2_identifier)+f_n_tail)

        # tie the purely angular beam values to the diffraction-limited domain
        alphamax=alphamax_freq_agnostic*(freq/nu_ref)
        alpha_vec=np.linspace(-alphamax,alphamax,Npix)
        alpha_grid_x,alpha_grid_y=np.meshgrid(alpha_vec,alpha_vec,indexing="ij")

        if i==0:
            alpha_grid_points=np.array([alpha_grid_x,alpha_grid_y]).T
            box=np.zeros((Npix,Npix,N_LoS)) # hold interpolated beam slices

        pol1_interpolated=gd(sky_angle_points,uninterp_slice_pol1,
                             alpha_grid_points,method="linear")
        pol2_interpolated=gd(sky_angle_points,uninterp_slice_pol2,
                             alpha_grid_points,method="linear")
        product=pol1_interpolated*pol2_interpolated
        power=product/np.max(product)
        box[:,:,i]=power
        ti=time.time()
        t[i]=ti
    np.save("CST_box_"+custom_outname,box)
    print(N_LoS,"slices managed in",np.sum(t),"s")
    print("mean=",np.mean(t),"s")
    print("std=",np.std(t),"s")
    return box

fname="farfield_(f=0.3)_[1]_efield.txt"
sky_angles_irreg,power_irreg=translate_sim_beam_slice(beam_sim_directory+fname)
print("sky_angles_irreg.shape=",sky_angles_irreg.shape)
print("power_irreg.shape=",power_irreg.shape)

p1id=")_[1]"
p2id=")_[1]"
box_test=box_from_simulated_beams(freqs,
                                  f_n_head=beam_sim_directory+"farfield_(f=",
                                  pol1_identifier=p1id,pol2_identifier=p2id,f_n_tail="_efield.txt",
                                  custom_outname="test_320_660_box")