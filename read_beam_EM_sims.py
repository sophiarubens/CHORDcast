import numpy as np
import pandas as pd
from scipy.interpolate import griddata as gd
from forecasting_pipeline import per_antenna
from matplotlib import pyplot as plt
import time

save_CST=True
pi=np.pi
tol=5 # same as in forecasting_pipeline for now
beam_sim_directory="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams_Aditya/CHORD_Fiducial_200kHz/"

def translate_sim_beam_slice(filename):
    # read in both polarizations
    df = pd.read_table(filename, skiprows=[0, 1,], sep='\s+', 
                       names=['theta', 'phi', 'AbsE', 'AbsCr', 'PhCr', 'AbsCo', 'PhCo', 'AxRat'])
    
    # establish non-log values
    power=10**(df.AbsE.values/10)
    theta_deg=df.theta.values
    theta=theta_deg*pi/180
    phi_deg=df.phi.values
    phi=phi_deg*pi/180
    sky_angle_x=theta*np.cos(phi)
    sky_angle_y=theta*np.sin(phi)
    sky_angle_points=np.array([sky_angle_x,sky_angle_y]).T

    if save_CST:
        np.save("CST_power",power)
        np.save("CST_theta",theta)
        np.save("CST_phi",phi)

    # print("slice: np.any(np.isnan(power)): ",np.any(np.isnan(power)))
    # print("slice: np.any(np.isinf(power)): ",np.any(np.isinf(power)))
    return sky_angle_points,power

Npix=256
freq_lo=0.58 # GHz for file names
freq_hi=0.62 
freqs=np.arange(freq_lo,freq_hi,0.0002)
nu_ref=freqs[0]
uv_manager=per_antenna(mode="pathfinder",nu_ctr=0.5*(freq_hi-freq_lo),Delta_nu=0.0002)
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
        # alphamax=alphamax_freq_agnostic*(freq/nu_ref)
        alphamax=1
        print("alphamax=",alphamax)
        alpha_vec=np.linspace(-alphamax,alphamax,Npix)
        alpha_grid_x,alpha_grid_y=np.meshgrid(alpha_vec,alpha_vec,indexing="ij")

        if i==0:
            alpha_grid_points=np.array([alpha_grid_x,alpha_grid_y]).T
            box=np.zeros((Npix,Npix,N_LoS)) # hold interpolated beam slices

        pol1_interpolated=gd(sky_angle_points,uninterp_slice_pol1,
                             alpha_grid_points,method="nearest") # linear applies nans when extrapolation would be necessary
        pol2_interpolated=gd(sky_angle_points,uninterp_slice_pol2,
                             alpha_grid_points,method="nearest")
        product=pol1_interpolated*pol2_interpolated
        power=product/np.max(product)
        box[:,:,i]=power
        ti1=ti
        ti=time.time()
        t[i]=ti-ti1
    np.save("CST_box_"+custom_outname,box)
    np.save("alpha_vec_"+custom_outname,box)
    return box,alpha_vec

def get_freq_names(freqs):
    freq_names=np.zeros(dtype=str)
    for i,freq in enumerate(freqs):
        freq_name=str(np.round(freq,4)) # round to four decimal places and convert to string
        freq_names[i]=freq_name.rstrip("0") # strip trailing zeros
    return freq_names

p1id=")_[1]"
p2id=")_[2]"

boxname="test_"+str(np.round(1000*freq_lo,0))+"_"+str(np.round(1000*freq_hi,0))+"_box"
gen_box=True
if gen_box:
    box_test,alpha_vec=box_from_simulated_beams(freqs,
                                                f_n_head=beam_sim_directory+"farfield_(f=",
                                                pol1_identifier=p1id,pol2_identifier=p2id,f_n_tail="_efield.txt",
                                                custom_outname=boxname)
else:
    box_test=np.load("CST_box_"+boxname+".npy")
    alpha_vec=np.load("alpha_vec_"+boxname+".npy")

print("box_test.shape=",box_test.shape)

# examine some slices to see if mínimamente estoy barking up the right tree
fig,axs=plt.subplots(4,3,figsize=(12,12),layout="constrained")
Nfreqs=len(freqs)
for j in range(4):
    jcut=j*Nfreqs//4

    yz_slice=box_test[jcut,:,:]
    print("yz_slice.shape=",yz_slice.shape)
    axs[j,0].imshow(yz_slice,origin="lower") #,extent=[alpha_vec[0],alpha_vec[-1],0,Nfreqs-1])
    axs[j,0].set_xlabel("y index ")
    axs[j,0].set_ylabel("z index")
    axs[j,0].set_title(str(jcut)+"/"+str(Nfreqs)+" yz")

    xz_slice=box_test[:,jcut,:]
    print("xz_slice.shape=",xz_slice.shape)
    axs[j,1].imshow(xz_slice,origin="lower") #,extent=[alpha_vec[0],alpha_vec[-1],0,Nfreqs-1])
    axs[j,1].set_xlabel("x index")
    axs[j,1].set_ylabel("z index")
    axs[j,1].set_title(str(jcut)+"/"+str(Nfreqs)+" xz")

    xy_slice=box_test[:,:,jcut]
    print("xy_slice.shape=",xy_slice.shape)
    axs[j,2].imshow(xy_slice,origin="lower") #,extent=[alpha_vec[0],alpha_vec[-1],alpha_vec[0],alpha_vec[-1]])
    axs[j,2].set_xlabel("x index")
    axs[j,2].set_ylabel("y index")
    axs[j,2].set_title(str(jcut)+"/"+str(Nfreqs)+" xy")
    plt.savefig("CST_beam_slices.png")