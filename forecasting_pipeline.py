import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm,TwoSlopeNorm,SymLogNorm #,CenteredNorm 

from scipy.fft import fftshift,ifftshift,fftfreq, fftn, irfftn, set_workers
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata as gd
from scipy.signal import convolve
from scipy.stats import binned_statistic_dd

import camb

from astropy.cosmology import Planck18
from astropy.cosmology.units import littleh
from astropy import constants as const
from astropy.units import Quantity
from astropy import units as u
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum

import cmasher
import inspect
import json
import pandas as pd
from pathlib import Path
import pygtc
import time

set_workers(6)

# cosmological. all are Planck18, whether they come from astropy or not
H0=Planck18.H0
h=H0/100
Omegam=Planck18.Om0
Omegamh2=Omegam*h**2
Omegab=Planck18.Ob0
Omegabh2=Omegab*h**2
Omegach2=0.12011
OmegaLambda=0.6842
ln1010AS=3.0448
AS=np.exp(ln1010AS)/10**10
ns=0.96605
w=-1
Omegamh2=Omegam*h**2
pars_fidu=    [ H0,    Omegabh2,      Omegamh2,      AS,           ns,    w] # suitable for getting matter power spec
parnames_fidu=["H_0", "Omega_b h^2", "Omega_c h^2", "10^9 * A_S", "n_s", "w"]

pars_forecast=    [ H0,    Omegabh2,      Omegach2,      w  ] # expect a 21-cm experiment to provide insight into these
parnames_forecast=["H_0", "Omega_b h^2", "Omega_c h^2", "w"]

dpar_default=1e-3*np.ones(len(pars_fidu))
dpar_default[3]*=1e-9

# physical
nu_HI_z0=1420.405751768*u.MHz
c=const.c
dif_lim_prefac=1.029

# mathematical
pi=np.pi
twopi=2.*pi
ln2=np.log(2)

# numerical
maxint=   np.iinfo(np.int64  ).max
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # intentionally preposterous number of sig figs from Mathematica
eps=1e-15
dpi_to_use=250

# CHORD
N_NS_full=24
N_EW_full=22
b_NS=8.5*u.m
b_EW=6.3*u.m
b_max_CHORD=np.sqrt((N_NS_full*b_NS)**2+(N_EW_full*b_EW)**2)*u.m
DRAO_lat=49.320791*pi/180.*u.rad # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6.*u.m
CHORD_channel_widco_MHz=0.1953125*u.MHz
def_observing_dec=pi/60.
def_offset=1.75*pi/180. # for this placeholder state where I build up the CHORD layout using rotation matrices instead of actual measurements. probably add Hans' mask at some point to punch the corners and receiver hut holes out...
def_pbw_pert_frac=1e-2
def_evol_restriction_threshold=1./30. # HERA 1/15 was made up. turn this down for a computationally less intense substitute
img_bin_tol=5 # ringing is remarkably insensitive to turning this down; you get really bad scale mismatch by turning it up... the real solution was the "need good resolution in both Fourier and configuration space" thing
def_PA_N_grid_pix=256 # doesn't change the deltaxy; a lower number of pixels per side means eval will be faster
N_fid_beam_types=1
integration_s=10*u.s # seconds
hrs_per_night=8*u.hr # borrowed from Debanjan / 21cmSense
# N_nights=100 # also borrowed from Debanjan / 21cmSense
N_nights=1
# def_N_timesteps=int(N_nights*hrs_per_night//integration_s)
def_N_timesteps=1 # for local tests
print("def_N_timesteps=",def_N_timesteps)

# side calculations
def get_padding(n): # avoid edge effects in a convolution
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi
def synthesized_beam_crossing_time(nu,bmax,dec=30.*u.deg): # to accumulate rotation synthesis
    synthesized_beam_widco_rad=dif_lim_prefac*(c/nu)/bmax
    beam_widco_deg=synthesized_beam_widco_rad*180/pi
    crossing_time_hrs_no_dec=beam_widco_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec.to(u.rad))
    return crossing_time_hrs
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
def comoving_dist_arg(z,Omegam=Omegam,OmegaLambda=OmegaLambda): # this is 1/ E(z)
    return 1/np.sqrt(Omegam*(1+z)**3+OmegaLambda)
def comoving_distance(z=0.5,H0=H0,Omegam=Omegam,OmegaLambda=OmegaLambda):
    integral,_=quad(comoving_dist_arg,0,z,args=(Omegam,OmegaLambda,))
    return (c.value*integral)/(H0.value*1000)*u.Mpc

# typical trivial conversions
def freq2z(nu_rest,nu_obs):
    nu_obs=nu_obs.to(nu_rest.unit)
    return nu_rest.value/nu_obs.value-1.
def z2freq(nu_rest=600.*u.MHz,z=nu_HI_z0/(600*u.MHz)-1.):
    return nu_rest/(z+1)

# Fourier space
def kpar(nu_ctr=600*u.MHz,chan_width=0.1953125*u.MHz,N_chan=300,H0=H0): # not pure cosmo. relies on LoS details of survey
    prefac=1e3*twopi*H0.value*nu_HI_z0.value/c.value # 1e3 to account for units of H0/c ... assumes nu_HI_z0 and chan_width have the same units
    z_ctr=freq2z(nu_HI_z0,nu_ctr)
    Ez=1/comoving_dist_arg(z_ctr)
    zterm=Ez/((1+z_ctr)**2*chan_width.value)
    kparmax=prefac*zterm
    kparmin=kparmax/N_chan
    Delta_kpar=kparmin
    kpar_bins=np.arange(kparmin,kparmax+Delta_kpar,Delta_kpar)/u.Mpc # units by construction
    return kpar_bins # evaluating at the z of the central freq of the survey (trusting slow variation...)
def kperp(nu_ctr=600.*u.MHz,bmin=6.*u.m,bmax=500.*u.m): # not pure cosmo. relies on sky plane details of survey
    Dc=comoving_distance(freq2z(nu_HI_z0,nu_ctr)) # evaluating at the z of the central freq of the survey (rely on slow variation = not worth reevaluating at each freq, as usual)
    prefac=twopi*nu_HI_z0.value*1e6/(c.value*Dc.value)
    kperpmin=prefac*bmin.value
    kperpmax=prefac*bmax.value
    Delta_kperp=kperpmin
    kperp_bins=np.arange(kperpmin,kperpmax+Delta_kperp,Delta_kperp)/u.Mpc # units by construction
    return kperp_bins
def wedge_kpar(nu_ctr,kperp,H0=H0,nu_rest=nu_HI_z0): # for some kperps of interest, which kparallels will the interferometer smear the wedge up to?
    nu_rest=nu_rest.to(nu_ctr.unit)
    z=freq2z(nu_rest,nu_ctr)
    E=1/comoving_dist_arg(z)
    Dc=comoving_distance(z)
    prefac=(H0*Dc*E).decompose()/(c*(1+z))
    return prefac*kperp
def PA_Gaussian(u,v,ctr,fwhm):
    u0,v0=ctr
    fwhmx,fwhmy=fwhm
    evaled=np.exp(-pi**2*((u-u0)**2*fwhmx**2+(v-v0)**2*fwhmy**2)/ln2) # prefactor ((pi*ln2)/(fwhmx*fwhmy)) will be overwritten during normalization anyway
    return evaled
def calc_b_HI(z):
    return 1.489 +0.460*(z-1) -0.118*(z-1)**2 +0.0678*(z-1)**3 -0.0128*(z-1)**4 +0.0009*(z-1)**5 # https://arxiv.org/abs/1804.09180 # Villaescusa-Navarro 2018. Widely accepted, but CHIME disagrees. CHIME is just one data point, but CHORD will probably be doing early science at similarly nonlinear scales
def Blackman_Harris_safe_for_FFT(N):
    a0,a1,a2,a3=0.35875,0.48829,0.14128,0.01168 # from the MATLAB (!) docs https://www.mathworks.com/help/signal/ref/blackmanharris.html
    n=N*fftfreq(N)
    w= a0 \
        -a1*np.cos(twopi*n/N) \
        +a2*np.cos(4.*pi*n/N) \
        -a3*np.cos(6.*pi*n/N)
    return w

# main computations
"""
this class helps compute contaminant power and cosmological parameter biases
using a Fisher-based formalism and numerical windowing for power beams with  
assorted properties and systematics.
"""

class beam_effects(object):
    def __init__(self,
                 # SCIENCE
                 # the observation
                 bmin:float=b_EW,bmax:float=b_max_CHORD,                          # max and min baselines of the array
                 nu_ctr:float=600.*u.MHz,                                         # central freq of survey
                 delta_nu:float=CHORD_channel_widco_MHz,                          # channel width
                 evol_restriction_threshold:float=def_evol_restriction_threshold, # how close to coeval is close enough? \Delta z/z
                 
                 # beam generalities
                 primary_beam_categ:str="CST",               # per-antenna Gaussian or uniform-across-array CST primary beams? >>>>>> COMING SOON: PER-ANTENNA CST <<<<<
                 primary_beam_type:str="Gaussian",           # beam type. right now an artifact of older eval modes, but will be useful in the future to specify Gaussian or per-antenna CST beams
                 primary_beam_aux:np.ndarray=None,           # NOW ONLY NECESSARY AS A QUANTITY THAT DOESN'T ALREADY EXIST UNDER ANOTHER NAME FOR BASE PA mode: beam FWHMs for both polarization cuts
                 primary_beam_unc:float=None,                # uncertainty in the primary beam width
                 primary_beam_modes:np.ndarray=None,  # config space pts at which a pre–discretely sampled primary beam is known

                 # additional considerations for per-antenna beams
                 PA_N_pert_types:int=0,                         # number of types of perturbed beam
                 N_pbws_pert:int=0,                          # number of beams to perturb
                 PA_N_fidu_types:int=N_fid_beam_types,          # number of types of fiducial beam
                 PA_fidu_types_prefactors=None,                 # multiplicative prefactors by which the different types of fiducial beam widths differ from those of the the diffraction-limited fiducial value
                 ioname:str="placeholder",                   # unique identifier for saving files and figures related to the uv coverage of this scenario
                 PA_distribution:str="random",                  # random, column, or corner distribution of fiducial beam types?
                 mode:str="full",                               # full or pathfinder CHORD?
                 per_channel_systematic=None,                   # which chunks of the survey band are afflicted by wrong beam widths?
                 per_chan_syst_facs:np.ndarray=[1.05,0.9,1.25], # multiplicative prefracs by which chunks of survey band have the wrong beam width

                 # additional considerations for CST beams
                 CST_lo=None,CST_hi=None,               # low and high frequencies of the CST simulation band (GHz !!!!!!!!!!not MHz)
                 CST_deltanu=None,                      # frequency spacing of CST simulations (MHz)
                 beam_sim_directory=None,               # directory to import CST simulations from 
                 f_mid1:str=")_[1]",f_mid2:str=")_[2]", # middle part of CST file names... should include something distinguish the two polarizations (not enforced)
                 f_tail:str="_efield.txt",              # trailing part of CST file names 
                 CST_f_head_fidu:str="farfield_(f=",CST_f_head_syst:str="farfield_(f=", # start of CST file names for different beam types (see Memo I for terminology description)

                 # additional^2 considerations for per-antenna CST beams (distinguish different systematics with different pointing errors era)
                 pointing_errors=[0.,0.,0.], # subject the real and thgt beams to pointing errors 

                 # FORECASTING
                 pars_set_cosmo:np.ndarray=pars_fidu,          # cosmo params to condition CAMB calls
                 pars_forecast:np.ndarray=pars_fidu,           # cosmo params of interest for a forecast
                 pars_forecast_names:np.ndarray=parnames_fidu, # >>>>> coming soon: support for derived parameters <<<<<
                 P_fid_for_cont_pwr=None,                      # fiducial power spectrum to use in Monte Carlo... typical choice for forecasting is CAMB (enforced default); some analyses may favour, for example, a flat spectrum
                 k_idx_for_window:int=0,                       # examine contaminant power or window functions?
                 interp_to_survey_modes:bool=False,            # don't bother turning down the k-space resolution to literal instrument-accessible modes
                 wedge_cut:bool=False,                         # excise info from voxels inside the foreground wedge?
                 layer_foregrounds:bool=True,                  # add synchrotron foregrounds on top of cosmo + beam data?

                 # NUMERICAL 
                 n_sph_modes:int=4096,                          # how many points in the cosmo power spectrum?
                 dpar=None,                                    # initial guess for numerical partial derivative step size
                 init_and_box_tol:float=0.05,                  # how much wider to make the config space extent of the brightness temp boxes compared to the survey box (numerical insurance factor...)
                 CAMB_tol:float=0.05,                          # same thing but for the CAMB call (if you make a sensible choice here, you will never have to extrapolate the cosmo spectrum to get info about a part of k-space you're interested in)
                 frac_tol_conv:float=0.1,                      # fraction (not percent) convergence for Monte Carlo ensemble -> used to determine the number of necessary realizations
                 seed=None,                                    # specify a seed if you want replicable RNG behaviour
                 ftol_deriv:float=1e-16,                       # this numerical tolerance factor * the function you are trying to differentiate gives a pointwise comparison for whether the derivative computation is accurate enough with the current step size
                 maxiter:int=5,                                # maximum number of times the partial derivative computation can recurse with an updated step size estimate
                 PA_N_grid_pix:int=def_PA_N_grid_pix,          # number of pixels per side of gridded uv plane
                 PA_img_bin_tol:int=img_bin_tol,               # how tightly to zoom into the gridded uv plane. there's a tradeoff here for a given number of voxels per side: if you zoom in really closely, you get to resolve more small differences in uv coordinates, but you'll probably incur some prominent ringing at the edges when you try to IFT a fairly sharp feature that extends to the edge of the box. If you don't zoom in much, the observed ringing won't be as stark, but there will be more wasted pixels / more of the uv coverage will be shuffled into central bins
                 radial_taper=None,image_taper=None,           # apply apodization along the line of sight or transverse directions?

                 # CONVENIENCE
                 heavy_beam_recalc:bool=True,                   # save time by not repeating per-antenna calculations?
                 already_imported_CST=False,
                 ):   
                
        # forecasting considerations
        self.seed=seed
        self.pars_set_cosmo=pars_set_cosmo
        self.N_pars_set_cosmo=len(pars_set_cosmo)
        self.pars_forecast=pars_forecast
        self.N_pars_forecast=len(pars_forecast)
        self.n_sph_modes=n_sph_modes
        self.dpar=dpar
        self.wedge_cut=wedge_cut
        nu_ctr=nu_ctr.to(u.MHz)
        self.nu_ctr=nu_ctr
        self.Deltanu=delta_nu
        self.bw=nu_ctr*evol_restriction_threshold
        self.Nchan=int(self.bw/self.Deltanu)
        self.z_ctr=freq2z(nu_HI_z0,nu_ctr)
        self.nu_lo=self.nu_ctr-self.bw/2.
        self.z_hi=freq2z(nu_HI_z0,self.nu_lo)
        self.Dc_hi=comoving_distance(self.z_hi)
        self.nu_hi=self.nu_ctr+self.bw/2.
        self.z_lo=freq2z(nu_HI_z0,self.nu_hi)
        self.Dc_lo=comoving_distance(self.z_lo)
        self.deltaz=self.z_hi-self.z_lo
        self.surv_channels=np.arange(self.nu_lo.value,self.nu_hi.value,self.Deltanu.value)*self.Deltanu.unit
        self.r0=comoving_distance(self.z_ctr)
        self.layer_foregrounds=layer_foregrounds
        self.b_NS=b_NS
        self.b_EW=b_EW
        if mode=="full":
            N_ant=512
            self.N_NS=N_NS_full
            self.N_EW=N_EW_full
        elif mode=="pathfinder":
            N_ant=64
            self.N_NS=N_NS_full//2
            self.N_EW=N_EW_full//2
        else:
            raise ValueError("unknown array mode")
        N_ant=N_ant
        
        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        self.kpar_surv=kpar_surv
        kparmin_surv=kpar_surv[0]
        kparmax_surv=kpar_surv[-1]
        self.kpar_surv=kpar_surv
        self.kparmin_surv=kparmin_surv
        self.kparmax_surv=kparmax_surv
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        self.bmax=bmax
        kperp_surv=kperp(self.nu_ctr,self.bmin,self.bmax)
        kperpmin_surv=kperp_surv[0]
        kperpmax_surv=kperp_surv[-1]
        self.kperp_surv=kperp_surv
        self.kperpmin_surv=kperpmin_surv
        self.kperpmax_surv=kperpmax_surv
        self.Nkperp_surv=len(self.kperp_surv)

        self.kmin_surv=np.sqrt(kperpmin_surv**2+kparmin_surv**2)
        self.kmax_surv=np.sqrt(kperpmax_surv**2+kparmax_surv**2)

        self.Lsurv_box_xy=twopi/kperpmin_surv
        self.Nvox_box_xy=int(self.Lsurv_box_xy*kperpmax_surv/pi)
        self.Lsurv_box_z=twopi/kparmin_surv
        self.Nvox_box_z=int(self.Lsurv_box_z*kparmax_surv/pi)

        self.fgfreqs=np.asarray([self.nu_lo.value,self.nu_hi.value])*self.nu_ctr.unit

        # primary beam considerations
        self.primary_beam_categ=primary_beam_categ
        self.primary_beam_unc= primary_beam_unc

        self.N_timesteps=           def_N_timesteps
        if (primary_beam_categ.lower()=="pa"):
            self.fwhm_x,self.fwhm_y=primary_beam_aux
            self.per_chan_syst_facs=per_chan_syst_facs

            self.PA_N_pert_types=          PA_N_pert_types
            self.N_pbws_pert=           N_pbws_pert
            if (self.N_pbws_pert>N_ant):
                print("WARNING: as called, more antennas would be perturbed than present in this array configuration")
                print("resetting with merely all antennas perturbed...")
                N_pbws_pert=N_ant
                self.N_pbws_pert=N_pbws_pert

            self.PA_N_grid_pix=            PA_N_grid_pix
            self.img_bin_tol=              PA_img_bin_tol
            self.PA_distribution=          PA_distribution
            self.PA_N_fidu_types= PA_N_fidu_types
            self.PA_fidu_types_prefactors= PA_fidu_types_prefactors
            fwhm=primary_beam_aux 

            fidu=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                pbw_pert_frac=0.,
                                N_timesteps=self.N_timesteps,
                                N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                N_fiducial_beam_types=1)
            real=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                pbw_pert_frac=0.,
                                N_timesteps=self.N_timesteps,
                                N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                distribution=self.PA_distribution,
                                N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
            thgt=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=self.PA_N_pert_types,
                                pbw_pert_frac=self.primary_beam_unc,
                                N_timesteps=self.N_timesteps,
                                N_pbws_pert=N_pbws_pert,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                distribution=self.PA_distribution,
                                N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
            per_chan_syst_name=thgt.per_chan_syst_name
            self.per_chan_syst_name=per_chan_syst_name
            self.surv_channels_MHz_from_PA=thgt.surv_channels_MHz
            self.surv_beam_widths_from_PA=thgt.surv_beam_widths

            if heavy_beam_recalc:
                fidu.stack_to_box()
                print("constructed fiducially-beamed box")
                fidu_box=fidu.box

                syst.stack_to_box()
                print("constructed systematics-beamed box")
                syst_box=syst.box
                xy_vec=real.xy_vec
                z_vec=real.z_vec

                np.save("fidu_box_"+ioname+".npy",fidu_box)
                np.save("syst_box_"+ioname+".npy",syst_box)
                np.save("xy_vec_"+  ioname+".npy",xy_vec.value)
                np.save("z_vec_"+   ioname+".npy",z_vec.value)
            else:
                fidu_box=np.load("fidu_box_"+ioname+".npy")
                syst_box=np.load("syst_box_"+ioname+".npy")
                xy_vec=  np.load("xy_vec_"+  ioname+".npy")*u.Mpc # not brittle. if imported from something saved from the pipeline, it actually will be in Mpc
                z_vec=   np.load("z_vec_"+   ioname+".npy")*u.Mpc

            primary_beam_modes=(xy_vec.value,xy_vec.value,z_vec.value) # might need to re-unit-ify this more robustly later, but for now the main use is interpolation and I don't want to jam up scipy by putting units where they have no business being  
            self.primary_beam_modes=primary_beam_modes
            self.primary_fidu=fidu_box

            assert(type(pointing_errors[0]==float)), "multiple pointing errors not supported in Gaussian beam mode. Use PA or PA-CST mode instead!"
            if pointing_errors!=[0.,0.,0.]: # mathematically nothing wrong with applying a 0º-in-every-direction rotation, but it's a waste of compute. definitely still wasting compute here constructing the same rotation matrix twice, but I'll sort that out later. 
                syst_box=repoint_beam(primary_beam_modes,syst_box,pointing_errors)

            self.primary_real=fidu_box
            self.primary_thgt=syst_box
            
        elif (primary_beam_categ.lower()=="cst"):
            pter0=pointing_errors[0]
            if type(pter0) is float:
                if pter0==0.:
                    N_pointing_errors=0
                else:
                    N_pointing_errors=1
            else: 
                N_pointing_errors=len(pointing_errors)

            precalculated_xy_vec=self.Lsurv_box_xy*fftshift(fftfreq(self.Nvox_box_xy))
            already_imported_fidu_CST=Path("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy").is_file()
            if heavy_beam_recalc and not already_imported_fidu_CST:
                fidu=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                          beam_sim_directory=beam_sim_directory,f_head=CST_f_head_fidu,
                                          f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="fidu_box_"+ioname)
                fidu.gen_box_from_simulated_beams()
                print("generated fidu beam box\n")
                fidu_box=fidu.box
                CST_z_vec=np.asarray(fidu.CST_z_vec)*u.Mpc # by construction = not brittle

            if heavy_beam_recalc and not already_imported_CST:
                syst=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                          beam_sim_directory=beam_sim_directory,f_head=CST_f_head_syst,
                                          f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="syst_box_"+ioname)
                syst.gen_box_from_simulated_beams()
                print("generated syst beam box\n")
                syst_box=syst.box

                np.save("fidu_box_"+ioname+".npy",fidu_box)
                np.save("syst_box_"+ioname+".npy",syst_box)
                already_imported_CST=True
                np.save("z_vec"+ioname+".npy",CST_z_vec.value)
            else:
                if already_imported_CST:
                    ioname_to_use=ioname.replace("Ntype_"+str(N_pointing_errors),"Ntype_1")
                else:
                    ioname_to_use=ioname
                fidu_box=np.load("fidu_box_"+ioname_to_use+".npy")
                syst_box=np.load("syst_box_"+ioname_to_use+".npy")
                CST_z_vec=np.load("z_vec"+ioname_to_use+".npy")*u.Mpc # by construction = not brittle
            primary_beam_modes=(precalculated_xy_vec.value,precalculated_xy_vec.value,CST_z_vec.value)
            self.primary_beam_modes=primary_beam_modes
            print("reconfigured/imported CST primary beams")

            N_CST_z=len(CST_z_vec)
            CST_syst_ensemble=np.zeros((1,N_pointing_errors+1,self.Nvox_box_xy,self.Nvox_box_xy,N_CST_z)) # shape of CST_syst_ensemble is ///////////
            CST_syst_ensemble[0,1,:,:,:]=syst_box

            if N_pointing_errors>0:
                for i,pointing_error in enumerate(pointing_errors):
                    CST_syst_ensemble[i+1,:,:,:]=repoint_beam(primary_beam_modes,syst_box,pointing_error)
                print("finished repointing beams for this complexity case")

            CST_freqs=np.arange(CST_lo.value,CST_hi.value,CST_deltanu.value)*CST_deltanu.unit
            if heavy_beam_recalc: # recalc the per-antenna part of CST
                fidu_per_antenna_ified=per_antenna(mode=mode,N_timesteps=self.N_timesteps,
                                                    N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                                    distribution="random",
                                                    sub_ensemble_of_CST_beams=fidu_box,CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs)
                fidu_per_antenna_ified.stack_to_box()
                print("finished per-antenna calculation for fidu CST beam")
                fidu_box_per_antenna_ified=fidu_per_antenna_ified.box
                PA_ified_xy_vec=fidu_per_antenna_ified.xy_vec
                PA_ified_z_vec=fidu_per_antenna_ified.z_vec
                syst_per_antenna_ified=per_antenna(mode=mode,N_timesteps=self.N_timesteps,
                                                    N_pbws_pert=N_pbws_pert,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                                    distribution=PA_distribution,
                                                    sub_ensemble_of_CST_beams=[fidu_box,CST_syst_ensemble],CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs)
                syst_per_antenna_ified.stack_to_box()
                print("finished per-antenna calculation for syst CST beam")
                syst_box_per_antenna_ified=syst_per_antenna_ified.box
                
                np.save("fidu_box_PA_ified_"+ioname+".npy",fidu_box_per_antenna_ified)
                np.save("syst_box_PA_ified_"+ioname+".npy",syst_box_per_antenna_ified)
                np.save("xy_vec_PA_ified_"+ioname+".npy",PA_ified_xy_vec.value)
                np.save("z_vec_PA_ified_"+ioname+".npy",PA_ified_z_vec.value)# xy_vec=real.xy_vec, z_vec=real.z_vec
                print("saved PA-ification")
            else: 
                fidu_box_per_antenna_ified=np.load("fidu_box_PA_ified_"+ioname+".npy")
                syst_box_per_antenna_ified=np.load("syst_box_PA_ified_"+ioname+".npy")
                PA_ified_xy_vec=np.load("xy_vec_PA_ified_"+ioname+".npy")*u.Mpc # by construction = not brittle
                PA_ified_z_vec=np.load("z_vec_PA_ified_"+ioname+".npy")*u.Mpc
                print("loaded PA-ification")
            print("finished importing/constructing per-antenna–ified CST beams")
            
            PA_ified_pbm=(PA_ified_xy_vec.value,PA_ified_xy_vec.value,PA_ified_z_vec.value) # might need to re-unit-ify this more robustly later, but for now the main use is interpolation and I don't want to jam up scipy by putting units where they have no business being

            self.primary_fidu=fidu_box_per_antenna_ified
            self.primary_real=fidu_box_per_antenna_ified
            self.primary_thgt=syst_box_per_antenna_ified

        elif primary_beam_categ.lower()=="pa-cst": # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            precalculated_xy_vec=self.Lsurv_box_xy*fftshift(fftfreq(self.Nvox_box_xy))
            N_CST_types=len(CST_f_head_syst)

            if np.all(pointing_errors==[[0.,0.,0.]]):
                N_pointing_errors=[0]
            else:
                N_pointing_errors=np.arange(0,len(pointing_errors)+1)
            N_pointing_errors_max=np.max(N_pointing_errors)
             
            already_imported_fidu_CST=Path("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy").is_file()
            p1="pol1/f_"
            p2="pol2/f_"
            if heavy_beam_recalc and not already_imported_fidu_CST:
                fidu=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                          beam_sim_directory=beam_sim_directory,f_head="fiducial/",
                                          f_mid1=p1,f_mid2=p2,f_tail="_GHz.txt",box_outname="fidu_box_"+ioname)
                fidu.gen_box_from_simulated_beams()
                print("generated fidu beam box\n")
                fidu_box=fidu.box
                CST_z_vec=np.asarray(fidu.CST_z_vec)*u.Mpc # by construction = not brittle
                np.save("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy",fidu_box)
                np.save("z_vec"+ioname+".npy",CST_z_vec.value)
            else:
                fidu_box=  np.load("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy")

                ioname_base_case=ioname.replace("N_CST_types_"+str(N_CST_types),"N_CST_types_1")
                ioname_base_case=ioname_base_case.replace("N_ptg_err_"+str(N_pointing_errors_max),"N_ptg_err_0")
                CST_z_vec=np.load("z_vec"+ioname_base_case+".npy")*u.Mpc # by construction = not brittle
            N_CST_z=len(CST_z_vec)

            if heavy_beam_recalc and not already_imported_CST: # only import the fiducial beam once
                syst_boxes=np.zeros((N_CST_types,self.Nvox_box_xy,self.Nvox_box_xy,N_CST_z)) # this needs to be 4D to be forward-compatible with the new iteration strategy in per_antenna
                for i,CST_f_head_syst_i in enumerate(CST_f_head_syst):
                    syst=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                              beam_sim_directory=beam_sim_directory,f_head=CST_f_head_syst_i,
                                              f_mid1=p1,f_mid2=p2,f_tail="_GHz.txt",box_outname="syst_box_"+ioname)
                    syst.gen_box_from_simulated_beams()
                    print("generated syst beam box\n")
                    syst_boxes[i,:,:,:]=syst.box
                
                np.save("syst_boxes_"+ioname+".npy",syst_boxes)
            else:
                syst_boxes=np.load("syst_boxes_"+ioname+".npy")
            
            N_CST_z=len(CST_z_vec)
            primary_beam_modes=(precalculated_xy_vec.value,precalculated_xy_vec.value,CST_z_vec.value)
            self.primary_beam_modes=primary_beam_modes

            CST_syst_ensemble=np.zeros((N_CST_types,N_pointing_errors_max+1,self.Nvox_box_xy,self.Nvox_box_xy,N_CST_z)) # shape of CST_syst_ensemble is (N_CST_types,self.Nvox_box_xy,self.Nvox_box_xy,N_CST_z) but the sub-ensembles passed to per_antenna have shapes  ////////replace
            CST_syst_ensemble[:,0,:,:,:]=syst_boxes # situate the pointing error–free versions

            if type(pointing_errors[0])==float:
                pointing_errors_to_loop_over=[pointing_errors]
            elif pointing_errors is not None:
                pointing_errors_to_loop_over=pointing_errors
            else:
                pointing_errors_to_loop_over=[[0.,0.,0.]]
            for i,syst_box in enumerate(syst_boxes):
                if N_pointing_errors_max>0:
                    for j,pointing_error in enumerate(pointing_errors_to_loop_over):
                        repointed=repoint_beam(primary_beam_modes,syst_box,pointing_error)
                        CST_syst_ensemble[i,j+1,:,:,:]=repointed
            print("finished repointing beams for this complexity case")
            
            CST_freqs=np.arange(CST_lo.value,CST_hi.value,CST_deltanu.value)*CST_deltanu.unit
            if heavy_beam_recalc: # recalc the per-antenna part of CST
                fidu_per_antenna_ified=per_antenna(mode=mode,N_timesteps=self.N_timesteps,
                                                    N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                                    distribution="random",
                                                    sub_ensemble_of_CST_beams=fidu_box,CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs)
                fidu_per_antenna_ified.stack_to_box()
                print("finished per-antenna calculation for fidu CST beam")
                fidu_box_per_antenna_ified=fidu_per_antenna_ified.box
                PA_ified_xy_vec=fidu_per_antenna_ified.xy_vec
                PA_ified_z_vec=fidu_per_antenna_ified.z_vec
                syst_per_antenna_ified=per_antenna(mode=mode,N_timesteps=self.N_timesteps,
                                                    N_pbws_pert=N_pbws_pert,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                                    distribution=PA_distribution,
                                                    sub_ensemble_of_CST_beams=[fidu_box,CST_syst_ensemble],CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs)
                syst_per_antenna_ified.stack_to_box()
                print("finished per-antenna calculation for syst CST beam")
                syst_box_per_antenna_ified=syst_per_antenna_ified.box
                
                np.save("fidu_box_PA_ified_"+ioname+".npy",fidu_box_per_antenna_ified)
                np.save("syst_box_PA_ified_"+ioname+".npy",syst_box_per_antenna_ified)
                np.save("xy_vec_PA_ified_"+ioname+".npy",PA_ified_xy_vec.value)
                np.save("z_vec_PA_ified_"+ioname+".npy",PA_ified_z_vec.value)# xy_vec=real.xy_vec, z_vec=real.z_vec
                print("saved PA-ification")
            else: 
                fidu_box_per_antenna_ified=np.load("fidu_box_PA_ified_"+ioname+".npy")
                syst_box_per_antenna_ified=np.load("syst_box_PA_ified_"+ioname+".npy")
                PA_ified_xy_vec=np.load("xy_vec_PA_ified_"+ioname+".npy")*u.Mpc # by construction = not brittle
                PA_ified_z_vec=np.load("z_vec_PA_ified_"+ioname+".npy")*u.Mpc
                print("loaded PA-ification")
            print("finished importing/constructing per-antenna–ified CST beams")
            
            PA_ified_pbm=(PA_ified_xy_vec.value,PA_ified_xy_vec.value,PA_ified_z_vec.value) # might need to re-unit-ify this more robustly later, but for now the main use is interpolation and I don't want to jam up scipy by putting units where they have no business being

            self.primary_fidu=fidu_box_per_antenna_ified
            self.primary_real=fidu_box_per_antenna_ified
            self.primary_thgt=syst_box_per_antenna_ified
        else:
            raise ValueError("unknown primary_beam_categ") # as far as primary power beam perturbations go, they can all pretty much be described as being applied PA, or in some externally-implemented custom way

        if primary_beam_categ.lower()=="pa-cst":
            self.pbm_for_cs=PA_ified_pbm
        else: 
            self.pbm_for_cs=primary_beam_modes
        self.primary_beam_type=primary_beam_type
        self.primary_beam_aux=primary_beam_aux
        self.primary_beam_unc=primary_beam_unc

        # groundwork-informed forecasting considerations
        if N_CST_types==0:
            if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
                self.perturbed_primary_beam_aux=(self.fwhm_x*(1-self.primary_beam_unc),self.fwhm_y*(1-self.primary_beam_unc))
                self.primary_beam_aux=np.array([self.fwhm_x,self.fwhm_y,self.r0.value]) 
                self.perturbed_primary_beam_aux=np.append(self.perturbed_primary_beam_aux,self.r0.value)
            else:
                raise ValueError("unknown primary_beam_type")
        self.P_fid_for_cont_pwr=P_fid_for_cont_pwr
        self.k_idx_for_window=k_idx_for_window

        # numerical protections for assorted k-ranges
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init*np.sqrt(3) # factor of sqrt(3) from pythag theorem for box to make extrapolation less likely to be necessary
        ksph,self.Ptruesph=self.get_21cm_power_spec(self.pars_set_cosmo,0.1*kmin_CAMB,10*kmax_CAMB)
        self.ksph=ksph/u.Mpc # by construction
        self.Deltabox_xy=self.Lsurv_box_xy/self.Nvox_box_xy
        self.Deltabox_z= self.Lsurv_box_z/ self.Nvox_box_z
        self.radial_taper=radial_taper
        self.image_taper=image_taper
        self.interp_to_surv_modes=interp_to_survey_modes

        # precision control for numerical derivatives
        self.ftol_deriv=ftol_deriv
        self.eps=eps
        self.maxiter=maxiter

        self.frac_tol_conv=frac_tol_conv
        
        # considerations for printing the calculated bias results
        self.pars_forecast_names=pars_forecast_names
        assert (len(pars_forecast)==len(pars_forecast_names))

        # placeholders for forecasting-relevant matrices
        self.del_P_del_pars=np.zeros((self.N_pars_forecast,self.Nkpar_surv,self.Nkperp_surv))
        self.F=None
        self.B=None

    def get_21cm_power_spec(self,pars_use:np.ndarray,minkh:float=1e-4/u.Mpc,maxkh:float=1./u.Mpc): # get matter power spec from CAMB
        N_zs=5
        z_ctr_idx=int(np.median(np.arange(1,N_zs+1)))
        z=[self.z_ctr+i for i in np.linspace(self.z_ctr/2,-self.z_ctr/2,N_zs,endpoint=True)] # matter power interpolator does better with more redshifts
        H0,ombh2,ommh2,As,ns,_=pars_use
        omch2=ommh2-ombh2
        h=H0/100.

        pars_use_internal=camb.set_params(H0=H0.value, ombh2=ombh2.value, omch2=omch2.value, ns=ns, mnu=0.06,omk=0)
        pars_use_internal.Transfer.transfer_type = camb.model.Transfer_b
        pars_use_internal.InitPower.set_params(As=As,ns=ns,r=0)
        maxkh=maxkh.to(1/u.Mpc)
        minkh=minkh.to(1/u.Mpc)
        pars_use_internal.set_matter_power(redshifts=z, kmax=maxkh.value*h.value)
        results = camb.get_results(pars_use_internal)
        hub_un=False
        matter_power_interpolator=results.get_matter_power_interpolator(nonlinear=True, hubble_units=hub_un, k_hunit=False)

        if hub_un:
            lengco_units=u.Mpc/littleh
        else:
            lengco_units=u.Mpc

        k_CAMB=np.linspace(minkh.value,maxkh.value,self.n_sph_modes)/lengco_units
        P_density_only=matter_power_interpolator.P(self.z_ctr,k_CAMB.value)*lengco_units**3

        Hz= results.hubble_parameter(z[z_ctr_idx])*u.km/u.s/u.Mpc
        Hz=Hz.to(H0.unit)
        h=h.to(u.km/u.s/u.Mpc)
        T_bar = 4.0*u.mK *(1+z[z_ctr_idx])**2 *ombh2.value/0.02 *0.7/h.value *H0.value/Hz.value # https://arxiv.org/pdf/astro-ph/0406676 eq. 5
        # b_HI =calc_b_HI(z[z_ctr_idx])
        # matter_to_21_cm=T_bar*b_HI
        # P_21=P_density_only*matter_to_21_cm**2 # CAMB handles the \eta_{\rm HI}
        P_21=P_density_only*T_bar**2

        return k_CAMB,P_21
    
    def unbin_to_Pcyl(self,pars_to_use:np.ndarray,kperp_to_use:np.ndarray=None,kpar_to_use:np.ndarray=None): # interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        if kperp_to_use is None:
            kperp_to_use=self.kperp_surv
        if kpar_to_use is None:
            kpar_to_use=self.kpar_surv
        k,Psph_use=self.get_21cm_power_spec(pars_to_use,minkh=0.1*self.kmin_surv,maxkh=10*self.kmax_surv)
        CAMBlength=len(Psph_use)
        k=k.reshape((CAMBlength,))
        Psph_use=Psph_use.reshape((CAMBlength,))
        k_unique, unique_idx = np.unique(k, return_index=True)
        Psph_use = Psph_use[unique_idx]
        k = k_unique

        self.Psph=Psph_use
        kperp_grid,kpar_grid=np.meshgrid(kperp_to_use,kpar_to_use, indexing="ij")
        kmag_grid=np.sqrt(kpar_grid**2+kperp_grid**2)
        Nkperp_use=len(kperp_to_use)
        Nkpar_use=len(kpar_to_use)
        Nk=Nkperp_use*Nkpar_use
        kmag_grid_flat=np.reshape(kmag_grid,(Nk,),order="C")
        sort_array=np.argsort(kmag_grid_flat)
        kmag_grid_flat_sorted=kmag_grid_flat[sort_array]

        Pcyl=np.zeros(Nk)
        interpolator=RGI((k.value,),Psph_use,
                         bounds_error=False,fill_value=None)
        Pcyl[sort_array]=interpolator(kmag_grid_flat_sorted[:, None])
        Pcyl=np.reshape(Pcyl,(Nkperp_use,Nkpar_use),order="C")*Psph_use.unit

        return kpar_grid,kperp_grid,Pcyl
    
    def get_pwr_law_FG_ingredient(self,Tref,nuref,alpha,sigma_alpha,rngseed=438): # change args to get different kinds of FGs described well enough as a power law (like synchrotron or free-free)
        # initialize a cosmo_stats object with a flat input power spec (P)
        fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Deltabox_z,
                       P_fid=self.P_flat,k_fid=self.k_for_flat,
                       Nvox=self.Nvox_box_xy,Nvoxz=1,
                       seed=self.seed) 

        # generate a slice from the flat temp spec
        fg.generate_GRF()
        white_noise_slice=fg.T_pristine 
        
        # apply LoS power law renormalization to each slice  
        freqs_in_ref_unit=self.freqs_for_fg.to(nuref.unit)   
        fg_box_this_ingredient=np.zeros((self.Nvox_box_xy,self.Nvox_box_xy,self.Nvox_box_z))
        rng=np.random.default_rng(rngseed)
        freq_ratios=freqs_in_ref_unit/nuref
        power_law_factor_means=np.zeros(self.Nvox_box_z)
        slice_of_alphas=rng.normal(loc=alpha, scale=sigma_alpha, size=(self.Nvox_box_xy,self.Nvox_box_xy))
        for i,freq_ratio_i in enumerate(freq_ratios):
            power_law_factor=Tref.value*freq_ratio_i**slice_of_alphas
            power_law_factor_means[i]=np.mean(power_law_factor)
            fg_slice=white_noise_slice*power_law_factor
            fg_box_this_ingredient[:,:,i]=fg_slice

        fg_box_this_ingredient*=Tref.unit
        fg_box_this_ingredient=fg_box_this_ingredient.to(u.mK)

        plt.figure()
        plt.plot(power_law_factor_means,label="power law factor means")
        plt.savefig("power_law_factor_means.png")
        plt.close()

        return fg_box_this_ingredient # centre-origin

    def calc_power_contamination(self, isolated:bool=False): # Monte Carlo numerical windowing of beam-aware brightness temp boxes to yield several cylindrically power spectra of interest for forecasting and diagnostics. various states of beam knowledge and fiducial spectrum as appropriate (see Memos I-II)
        if self.P_fid_for_cont_pwr is None:
            P_cosmo=np.reshape(self.Ptruesph,(self.n_sph_modes),order="C")
        elif self.P_fid_for_cont_pwr=="window": # make the fiducial power spectrum a numerical top hat
            P_cosmo=np.zeros(self.n_sph_modes)
            P_cosmo[self.k_idx_for_window]=1.
            P_cosmo*=u.mK**2*u.Mpc**3 # by design, not brittle
        else:
            raise ValueError("unknown P_fid_for_cont_pwr")

        power_unit=u.mK**2*self.Deltabox_xy.unit**3
        N_flat=10*self.Nkpar_surv
        P_flat=np.ones(N_flat) *power_unit
        # P_flat=np.ones(N_flat)/N_flat**2 *power_unit
        self.P_flat=P_flat
        self.k_for_flat=np.linspace(self.kparmin_surv,self.kparmax_surv,10*self.Nkpar_surv)
        if self.layer_foregrounds:
            # fg prep
            box_z_freqs= np.linspace(self.nu_hi.value,self.nu_lo.value,self.Nvox_box_z,endpoint=True)*self.Deltanu.unit
            self.freqs_for_fg=box_z_freqs.to(u.MHz) # descending in frequency to match the iteration over increasing redshift

            fg_box=np.zeros((self.Nvox_box_xy,self.Nvox_box_xy,self.Nvox_box_z))*u.mK
            # Adrian's 2011 foreground model. save point sources for later because I'll need to generalize my approach
            # format of params for each case: Tref, nuref, alpha, sigma_alpha
            fg_info_cases=[[335.4*u.K, 150*u.MHz, -2.8,  0.1],   # synchrotron
                           [33.5 *u.K, 150*u.MHz, -2.15, 0.01] ] # free-free
            # fg_info_cases=[[335.4*u.K, 150*u.MHz, -2.8,  0.1]]
            for fg_info in fg_info_cases:
                Tref,nuref,alpha,sigma_alpha=fg_info
                fg_box_ingredient=self.get_pwr_law_FG_ingredient(Tref,nuref,alpha,sigma_alpha)
                fg_box+=fg_box_ingredient
            self.fg_box=fg_box # centre-origin

            # bonus step: compute power spec to facilitate comparisons to power spectra with all the other cosmo + fidu beam + syst + fg ingredient permutations
            fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           T_pristine=fg_box,
                           frac_tol=self.frac_tol_conv,seed=self.seed)
            fg.power_Monte_Carlo()

            self.P_xx_xx_xx_fg=fg.P_binned_MC_complete
            print("                           fg MC complete")

        co_fi_xx_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_cosmo,k_fid=self.ksph, 
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_fidu,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_fidu,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,    
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=fg_box)
        self.kperpbins_internal=co_fi_xx_fg.kperpbins
        self.kparbins_internal=co_fi_xx_fg.kparbins
        co_fi_sy_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_cosmo,k_fid=self.ksph,
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_real,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_thgt,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=fg_box)
        xx_fi_sy_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_flat,k_fid=self.k_for_flat,
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_real,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_thgt,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=fg_box)
        xx_fi_xx_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_flat,k_fid=self.k_for_flat,
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_real,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_real,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=fg_box)
        co_fi_xx_xx=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_cosmo,k_fid=self.ksph, 
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_fidu,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_fidu,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,    
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=None)
        co_fi_sy_xx=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_cosmo,k_fid=self.ksph, 
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_fidu,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_real,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,    
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=None)
        co_xx_xx_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_cosmo,k_fid=self.ksph, 
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       frac_tol=self.frac_tol_conv,seed=self.seed,    
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=fg_box)
        xx_fi_xx_xx=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_flat,k_fid=self.k_for_flat, 
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_fidu,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_fidu,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,    
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=None)
        xx_fi_sy_xx=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                       P_fid=P_flat,k_fid=self.k_for_flat, 
                       Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                       primary_beam_num=self.primary_fidu,primary_beam_type_num="manual",
                       primary_beam_den=self.primary_real,primary_beam_type_den="manual",
                       frac_tol=self.frac_tol_conv,seed=self.seed,    
                       primary_beam_modes=self.pbm_for_cs,
                       radial_taper=self.radial_taper,image_taper=self.image_taper,
                       wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,fg_box=None)

        recalc_co_fi_xx_fg=False
        recalc_co_fi_sy_fg=False
        recalc_xx_fi_sy_fg=False
        recalc_xx_fi_xx_fg=False
        recalc_co_fi_xx_xx=False
        recalc_co_fi_sy_xx=False
        recalc_co_xx_xx_fg=False
        recalc_xx_fi_xx_xx=False
        recalc_xx_fi_sy_xx=False
        if isolated==False:
            recalc_co_fi_xx_fg=True
            recalc_co_fi_sy_fg=True
            recalc_xx_fi_sy_fg=True
            recalc_xx_fi_xx_fg=True
            recalc_co_fi_xx_xx=True
            recalc_co_fi_sy_xx=True
            recalc_co_xx_xx_fg=True
            recalc_xx_fi_xx_xx=True
            recalc_xx_fi_sy_xx=True
        
        elif isolated=="co_fi_xx_fg":
            recalc_co_fi_xx_fg=True
        elif isolated=="co_fi_sy_fg":
            recalc_co_fi_sy_fg=True
        elif isolated=="xx_fi_sy_fg":
            recalc_xx_fi_sy_fg=True
        elif isolated=="xx_fi_xx_fg":
            recalc_xx_fi_xx_fg=True
        elif isolated=="co_fi_xx_xx":
            recalc_co_fi_xx_xx=True
        elif isolated=="co_fi_sy_xx":
            recalc_co_fi_sy_xx=True
        elif isolated=="co_xx_xx_fg":
            recalc_co_xx_xx_fg=True
        elif isolated=="xx_fi_xx_xx":
            recalc_xx_fi_xx_xx=True
        elif isolated=="xx_fi_sy_xx":
            recalc_xx_fi_sy_xx=True

        if recalc_co_fi_xx_fg:
            co_fi_xx_fg.power_Monte_Carlo(interfix="fi")
            self.N_per_realization= co_fi_xx_fg.N_per_realization
            self.P_co_fi_xx_fg=     co_fi_xx_fg.P_binned_MC_complete
            self.kperp_for_cosmo=  co_fi_xx_fg.kperpbins
            self.kpar_for_cosmo=   co_fi_xx_fg.kparbins
            print("cosmo + fidu beam +        fg MC complete")
        if recalc_co_fi_sy_fg:
            co_fi_sy_fg.power_Monte_Carlo(interfix="co_fi_sy_fg")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_fi_sy_fg.N_per_realization
                self.kperp_for_cosmo=  co_fi_sy_fg.kperpbins
                self.kpar_for_cosmo=   co_fi_sy_fg.kparbins
            self.P_co_fi_sy_fg=         co_fi_sy_fg.P_binned_MC_complete
            print("cosmo + fidu beam + syst + fg MC complete")
        if recalc_xx_fi_sy_fg:
            xx_fi_sy_fg.power_Monte_Carlo(interfix="xx_fi_sy_fg")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= xx_fi_sy_fg.N_per_realization
                self.kperp_for_cosmo=  xx_fi_sy_fg.kperpbins
                self.kpar_for_cosmo=   xx_fi_sy_fg.kparbins
            self.P_xx_fi_sy_fg=         xx_fi_sy_fg.P_binned_MC_complete
            print("        fidu beam + syst + fg MC complete")
        if recalc_xx_fi_xx_fg:
            xx_fi_xx_fg.power_Monte_Carlo(interfix="nn")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= xx_fi_xx_fg.N_per_realization
                self.kperp_for_cosmo=  xx_fi_xx_fg.kperpbins
                self.kpar_for_cosmo=   xx_fi_xx_fg.kparbins
            self.P_xx_fi_xx_fg=         xx_fi_xx_fg.P_binned_MC_complete
            print("        fidu beam +      + fg MC complete")
        if recalc_co_fi_xx_xx:
            co_fi_xx_xx.power_Monte_Carlo(interfix="co_fi_xx_xx")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_fi_xx_xx.N_per_realization
                self.kperp_for_cosmo=  co_fi_xx_xx.kperpbins
                self.kpar_for_cosmo=   co_fi_xx_xx.kparbins
            self.P_co_fi_xx_xx=         co_fi_xx_xx.P_binned_MC_complete
            print("cosmo + fidu beam             MC complete")
        if recalc_co_fi_sy_xx:
            co_fi_sy_xx.power_Monte_Carlo(interfix="co_fi_sy_xx")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_fi_sy_xx.N_per_realization
                self.kperp_for_cosmo=  co_fi_sy_xx.kperpbins
                self.kpar_for_cosmo=   co_fi_sy_xx.kparbins
            self.P_co_fi_sy_xx=         co_fi_sy_xx.P_binned_MC_complete
            print("cosmo + fidu beam + syst      MC complete")
        if recalc_co_xx_xx_fg:
            co_xx_xx_fg.power_Monte_Carlo(interfix="co_xx_xx_fg")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_xx_xx_fg.N_per_realization
                self.kperp_for_cosmo=  co_xx_xx_fg.kperpbins
                self.kpar_for_cosmo=   co_xx_xx_fg.kparbins
            self.P_co_xx_xx_fg= co_xx_xx_fg.P_binned_MC_complete
            print("cosmo +                    fg MC complete")
        if recalc_xx_fi_xx_xx:
            xx_fi_xx_xx.power_Monte_Carlo(interfix="xx_fi_xx_xx")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= xx_fi_xx_xx.N_per_realization
                self.kperp_for_cosmo=  xx_fi_xx_xx.kperpbins
                self.kpar_for_cosmo=   xx_fi_xx_xx.kparbins
            self.P_xx_fi_xx_xx = xx_fi_xx_xx.P_binned_MC_complete
            print("        fidu beam             MC complete")
        if recalc_xx_fi_sy_xx:
            xx_fi_sy_xx.power_Monte_Carlo(interfix="xx_fi_sy_xx")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= xx_fi_sy_xx.N_per_realization
                self.kperp_for_cosmo=  xx_fi_sy_xx.kperpbins
                self.kpar_for_cosmo=   xx_fi_sy_xx.kparbins
            self.P_xx_fi_sy_xx = xx_fi_sy_xx.P_binned_MC_complete
            print("        fidu beam + syst      MC complete")
        TEST=True
        if TEST:
            COSMOTEST=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                    P_fid=P_cosmo,k_fid=self.ksph, 
                                    Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                    radial_taper=self.radial_taper,image_taper=self.image_taper,
                                    frac_tol=self.frac_tol_conv,seed=self.seed)
            COSMOTEST.power_Monte_Carlo(interfix="CO_XX_XX_XX_") # extra underscore is because numpy is fine with case-sensitive file names but MacOS is not :(
            self.P_CO_XX_XX_XX=COSMOTEST.P_binned_MC_complete

            print("COSMO                         MC COMPLETE")

        _,_,P_co_xx_xx_xx=self.unbin_to_Pcyl(self.pars_set_cosmo, 
                                             kperp_to_use=self.kperp_for_cosmo[:-1]+0.5*(self.kperp_for_cosmo[1]-self.kperp_for_cosmo[0]), 
                                             kpar_to_use=self.kpar_for_cosmo[:-1]+0.5*(self.kpar_for_cosmo[1]-self.kpar_for_cosmo[0]))
        self.P_co_xx_xx_xx=P_co_xx_xx_xx

        if isolated==False:
            self.Pcont_cyl=self.P_co_fi_sy_fg-self.P_co_fi_xx_fg

    def cyl_partial(self,n:int): # cylindrically binned matter power spectrum partial WRT one cosmo parameter
        dparn=self.dpar[n]
        pcopy=self.pars_set_cosmo.copy()
        pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

        _,_,Pcyl=self.unbin_to_Pcyl(pcopy)
        P0=np.mean(np.abs(Pcyl))+self.eps
        tol=self.ftol_deriv*P0 # generalizes tol=ftol*f0 from PHYS512

        pcopy[n]=pcopy[n]+2*dparn 
        _,_,Pcyl_2plus=self.unbin_to_Pcyl(pcopy)
        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-2*dparn
        _,_,Pcyl_2minu=self.unbin_to_Pcyl(pcopy)
        deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*self.dpar[n])

        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]+dparn
        _,_,Pcyl_plus=self.unbin_to_Pcyl(pcopy)
        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-dparn
        _,_,Pcyl_minu=self.unbin_to_Pcyl(pcopy)
        deriv2=(Pcyl_plus-Pcyl_minu)/(2*self.dpar[n])

        Pcyl_dif=Pcyl_plus-Pcyl_minu
        if (np.mean(Pcyl_dif)<tol): # might be too strict or loose a condition
            estimate=(4*deriv2-deriv1)/3
            self.iter=0 # reset for next time
            self.del_P_del_pars[n,:,:]=estimate
        else:
            pnmean=np.mean(np.abs(pndispersed)) # the np.abs part should be redundant because, by this point, all the k-mode values and their corresponding dpns and Ps should be nonnegative, but anyway... numerical stability or something idk
            Psecond=np.abs(np.mean(2*self.Pcyl-Pcyl_minu-Pcyl_plus))/self.dpar[n]**2 # an estimate!! break out of the vicious cycle of not having enough info
            dparn=np.sqrt(self.eps*pnmean*P0/Psecond)
            self.dpar[n]=dparn # send along knowledge of the updated step size
            self.iter+=1
            self.cyl_partial(n) # recurse
            if self.iter==self.maxiter:
                print("failed to converge in {:d} iterations".format(self.maxiter))
                fallback=(4*deriv2-deriv1)/3
                print("RETURNING fallback")
                self.iter=0 # still need to reset for next time
                self.del_P_del_pars[n,:,:]=fallback

    def compute_del_P_del_pars(self): # builds a (N_pars_forecast,Nkperp,Nkpar) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
        for n in range(self.N_pars_set_cosmo):
            self.iter=0 # b/c starting a new partial deriv calc.
            self.cyl_partial(n)

    def compute_noise(self):
        assert self.N_per_realization is not None, "try calling the compute_noise() method again after running calc_power_contamination()"
        self.sample_variance=np.sqrt(2/self.N_per_realization)*self.P_co_fi_sy_fg # rescale according to the number of realizations 

        sen=CHORD_sense(spacing=[self.b_EW,self.b_NS], n_side=[self.N_EW,self.N_NS], orientation=def_offset, center=None, dish_diameter=D, # array layout
                        freq_cen=self.nu_ctr, integration_time=integration_s*u.s, time_per_day=hrs_per_night, n_days=100, bandwidth=self.bw, # obs config
                        Trcv=35*u.K, latitude=DRAO_lat, tsky_ref_freq=400.*u.MHz, tsky_amplitude=25*u.K, # what's going on with the sky?
                        coherent=False, horizon_buffer=0.1*littleh/u.Mpc, foreground_model="optimistic") # processing details
        sen.sense2d()
        kperp_from_21cmSense=sen.sense2d_kperp
        kpar_from_21cmSense=sen.sense2d_kpar
        thnoise_21cmSense=sen.sense2d_P
        kperp_surv_grid,kpar_surv_grid=np.meshgrid(self.kperp_surv,self.kpar_surv, indexing="ij")
        thnoise_surv=RGI((kperp_from_21cmSense.value,kpar_from_21cmSense.value),thnoise_21cmSense,
                          bounds_error=False,fill_value=None)(np.array([kperp_surv_grid.value,kpar_surv_grid.value]).T).T
        self.thermal_noise=thnoise_surv
        self.all_sigmasuncs=self.thermal_noise+self.sample_variance # ensemble stats + 21cmSense

    def compute_F(self):
        if np.all(self.del_P_del_pars==0):
            self.compute_del_P_del_pars()
        if self.uncs is None:
            self.compute_noise()

        V=0.*self.del_P_del_pars
        for i in range(self.N_pars_forecast):
            V[i,:,:]=self.del_P_del_pars[i,:,:]/self.uncs # elementwise division for an nkpar x nkperp slice
        self.V=V
        V_completely_transposed=np.transpose(V,axes=(2,1,0))
        self.V_completely_transposed=V_completely_transposed
        self.F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("computed F")

    def compute_B(self):
        if self.del_P_del_pars is None:
            self.compute_del_P_del_pars()
        if self.uncs is None:
            self.compute_noise()
        
        self.Pcont_div_sigma=self.Pcont_cyl/self.uncs
        self.B=np.einsum("jk,ijk->i",self.Pcont_div_sigma,self.V)
        print("computed B")  
        
    def bias(self): # collect the ingredients of the parameter bias calculation
        self.compute_del_P_del_pars()
        print("built partials")
        self.calc_power_contamination()
        print("computed Pcont")

        self.compute_noise()
        print("computed uncertainties at each k-mode")

        if self.F is None:
            self.compute_F()
        if self.B is None:
            self.compute_B()
        self.biases=(np.linalg.inv(self.F)@self.B).reshape((self.N_pars_forecast,))
        print("computed b")

    def forecast_corner_plot(self,N_Fisher_samples:int=10000):
        if self.F is None:
            self.compute_F()

        C=np.linalg.inv(self.F)
        if np.any(C==np.nan):
            C=np.linalg.pinv(self.F)
        rng=np.random.default_rng
        samples=rng.multivariate_normal(np.zeros(self.N_pars_forecast),C,size=N_Fisher_samples)
        pygtc.plotGTC(chains=samples, 
                      paramNames=self.pars_forecast_names,
                      truths=self.pars_forecast,
                      plot_name="forecast_corner_plot.png")

    def print_results(self):
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(self.pars_forecast):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(self.pars_forecast_names[p], par, self.biases[p], self.biases[p]/par))
        return None
####################################################################################################################################################################################################################################

def repoint_beam(domain,beam,rot_angles=[0.,0.,0.,]):
    rot_x,rot_y,rot_z=np.asarray(rot_angles)*np.pi/180.*u.rad
    RX=np.asarray([[np.cos(rot_x),-np.sin(rot_x), 0.],
                   [np.sin(rot_x), np.cos(rot_x), 0.],
                   [0.,            0.,            1.]])
    RY=np.asarray([[ np.cos(rot_y),  0., np.sin(rot_y)],
                   [ 0.,             1., 0.],
                   [-np.sin(rot_y),  0., np.cos(rot_y)]])
    RZ=np.asarray([[1., 0.,             0.,],
                   [0., np.cos(rot_z), -np.sin(rot_z)],
                   [0., np.sin(rot_z),  np.cos(rot_z)]])
    R=RX@RY@RZ
    xvec,yvec,zvec=domain
    Nx=len(xvec)
    Ny=len(yvec)
    Nz=len(zvec)
    N=Nx*Ny*Nz
    x_grid,y_grid,z_grid=np.meshgrid(xvec,yvec,zvec, indexing="ij")
    x_flat=np.reshape(x_grid,(N,),order="C")
    y_flat=np.reshape(y_grid,(N,),order="C")
    z_flat=np.reshape(z_grid,(N,),order="C")
    xyz_flat=np.asarray([x_flat,y_flat,z_flat]).T # 3xN

    # philosophy here: need 3xN for R@ compatibility, but can't just use R@xyz_flat because RGI needs something with shape ((Nx,),(Ny,),(Nz,)), not (3,N)
    x_prime_vec,_,_=R@[x_grid[:,0,0],y_grid[:,0,0],z_grid[:,0,0]] # this is probably going to take some reslicing, re-transposing, and reassembling
    _,y_prime_vec,_=R@[x_grid[0,:,0],y_grid[0,:,0],z_grid[0,:,0]]
    _,_,z_prime_vec=R@[x_grid[0,0,:],y_grid[0,0,:],z_grid[0,0,:]]

    interpolator=RGI((x_prime_vec,y_prime_vec,z_prime_vec),beam,
                     bounds_error=False,fill_value=None)
    rotated_beam_sampled_at_original_domain=interpolator(xyz_flat)
    unflattened_output=np.reshape(rotated_beam_sampled_at_original_domain,beam.shape,order="C")
    renormalized_rotated=unflattened_output/np.max(unflattened_output)
    renormalized_rotated[unflattened_output<0.]=0. # too hacky for real science

    return renormalized_rotated

"""
this class helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box (temp field)
2. generate GRF realizations of a cosmo box consistent with a chosen power spectrum
3. Monte Carlo effective windowing of a power spectrum by a primary beam
4. interpolate a power spectrum (sph, cyl, or sph->grid)
"""

class cosmo_stats(object):
    def __init__(self,
                 Lxy:float=600.*u.Mpc,Lz:float=None,                                         # physical box length (Mpc). one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
                 T_pristine:np.ndarray=None,T_primary:np.ndarray=None,                       # brightness temperature box realizations without ("_pristine") or with ("_primary") the primary beam multiplied in
                 P_fid:np.ndarray=None,                                                      # power spectrum you want to window. probably comes from cosmo (like CAMB) or is flat (for a reference calculation)
                 k_fid:np.ndarray=None,                                                      # Fourier space points where the fiducial power spectrum is sampled
                 Nvox:int=None,Nvoxz:int=None,                                               # number of voxels in the x/y or z directions
                 primary_beam_num:np.ndarray=None,     primary_beam_den:np.ndarray=None,     # numerator/denominator (of power spectrum estimator) version of the primary beam (box of values evaluated in config space)
                 primary_beam_aux_num:np.ndarray=None, primary_beam_aux_den:np.ndarray=None, # numerator/denominator version of helpful quantities that go along with the primary beam (characteristic widths for a per-antenna Gaussian beam; x/y and z vectors for a CST beam)
                 primary_beam_type_num:str="Gaussian", primary_beam_type_den:str="Gaussian", # USED TO BE Airy/Gaussian for achromatic uniform-across-array beams. CURRENTLY can only be Gaussian, but SOON will be generalized to admit per-antenna CST beams
                 Nkperp:int=0,Nkpar:int=0,                                                  # number of k-bins in the sky plane and line of sight directions
                 binning_mode:str="lin",                                                     # bin linearly or logarithmically
                 bin_each_realization:bool=False,                                            # bin each realization of the Monte Carlo? (with the current implementation there's no typical use case where this would be necessary, but the option is there)
                 frac_tol:float=0.1,                                                         # fractional tolerance in cosmic variance of the Monte Carlo ensemble -> used to calculate the number of realizations
                 kperpbins_interp:np.ndarray=None,kparbins_interp:np.ndarray=None,           # bins where you want to know about the power spectrum (if you're interested in interpolating to some binning scheme other than what you get from chopping up the box)
                 P_MC_complete:np.ndarray=None,                                                # converged Monte Carlo power spectrum
                 kind:str="cubic",avoid_extrapolation:bool=False,                            # conditioning choices for interpolation: degree of interpolation; whether or not to avoid extrapolation
                 seed=None,                                            # Monte Carlo realization logistics: whether or not to subtract the monopole moment when you generate boxes (the option is mostly there if you're interested in off-label uses of this code to compute power spectra from fields that are not cosmological overdensity fields); RNG seed for predictable ensemble behaviour
                 primary_beam_modes:np.ndarray=None,                                         # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 radial_taper=None,image_taper=None,                                         # apodize along the sky plane or line-of-sight directions to suppress ringing originating from features that cut off sharply?
                 wedge_cut:bool=False,nu_ctr_for_wedge=None,                                 # throw away info from k-modes inside the foreground wedge?; when using synchrotron foregrounds AND performing a wedge cut, the calling routine should specify the central frequency of the survey in question to have a physical anchor for the foregrounds
                 fg_box:np.ndarray=None):                                # layer synchrotron foregrounds on top of brightness temp realizations?; fg fields and modes computed by a calling routine
        
        # spectrum and box
        if (Lz is None): # cubic box
            self.Lz=Lxy
            self.Lxy=Lxy
        else:            # rectangular prism box
            self.Lz=Lz
            self.Lxy=Lxy
        physical_volume=self.Lxy**2*self.Lz
        self.physical_volume=physical_volume
        self.fg_box=fg_box
        self.P_fid=P_fid
        self.P_fid_box=None
        self.T_primary=T_primary
        self.T_pristine=T_pristine
        if ((T_primary is None) and (T_pristine is None) and (P_fid is None)): # require either a box or a fiducial power spec (il faut some way of determining #voxels/side; passing just Nvox is not good enough)
            raise ValueError("not enough info")
        else:                                                                  # there is possibly enough info to proceed, but still need to check for conflicts and gaps
            if ((T_pristine is not None) and (T_primary is not None)):
                print("WARNING: T_pristine and T_primary both passed; T_primary will be temporarily ignored and then internally overwritten to ensure consistency with primary_beam")
                if (T_pristine.shape!=T_primary.shape):
                    raise ValueError("conflicting info")
                else:                                                          # use box shape to set cubic/ rectangular prism box attributes
                    Nvox, _, Nvoxz=T_primary.shape
            if ((Nvox is not None) and (T_pristine is not None)):              # possible conflict: if both Nvox and a box are passed, 
                T_pristine_shape0,_,T_pristine_shape2=T_pristine.shape
                if (Nvox!=T_pristine.shape[0]):                                # but Nvox and the box shape disagree,
                    raise ValueError("conflicting info")                       # estamos en problemas
                else:
                    Nvox= T_pristine_shape0                               # otherwise, initialize the Nvox attributes
                    Nvoxz=T_pristine_shape2
            elif (Nvox is not None and Nvoxz is None):                                           # if Nvox was passed but T was not, use Nvox to initialize the Nvox attributes
                Nvoxz=np.copy(Nvox)
            elif T_pristine is not None:                                                              # remaining case: T was passed but Nvox was not, so use the shape of T to initialize the Nvox attributes
                Nvox, _, Nvoxz = T_pristine.shape
            self.Nvox=Nvox
            self.Nvoxz=Nvoxz

            if (P_fid is not None): # no hi fa res si the fiducial power spectrum has a different dimensionality or bin width than the realizations you plan to generate (boxes will be generated from a grid-interpolated P_fid, anyway)
                Pfidshape=P_fid.shape
                Pfiddims=len(Pfidshape)
                if (Pfiddims==2):
                    if primary_beam_num is None: # trying to do a minimalistic instantiation where I merely provide a fiducial power spectrum and interpolate it
                        self.fid_Nkperp,self.fid_Nkpar=Pfidshape
                        if primary_beam_den is not None: 
                            raise ValueError("conflicting info") # numerator primary beam needs to be the fiducial one; doesn't make sense to claim you have a perturbed but not fiducial pb
                    else:
                        try: # see if the power spec is a CAMB-esque (1,npts) array
                            self.P_fid=np.reshape(P_fid,(Pfidshape[-1],),order="C") # make the CAMB MPS shape amenable to the calcs internal to this class
                        except: # barring that...
                            pass # treat the power spectrum as being truly cylindrically binned
                elif (Pfiddims==1):
                    self.fid_Nkperp=Pfidshape[0] # already checked that P_fid is 1d, so no info is lost by extracting the int in this one-element tuple, and fid_Nkperp being an integer makes things work the way they should down the line
                    self.fid_Nkpar=0
                else:
                    raise ValueError("unsupported binning mode")
        
        # config space
        self.box_shape=(self.Nvox,self.Nvox,self.Nvoxz) if self.Nvoxz>1 else (self.Nvox,self.Nvox)
        self.Deltaxy=self.Lxy/self.Nvox                           # sky plane: voxel side length
        self.xy_vec_for_box=self.Lxy*fftshift(fftfreq(self.Nvox)) # sky plane Cartesian config space coordinate axis
        self.Deltaz= self.Lz/self.Nvoxz                           # line of sight voxel side length
        self.z_vec_for_box= self.Lz*fftshift(fftfreq(self.Nvoxz)) # line of sight Cartesian config space coordinate axis
        self.d3r=self.Deltaz*self.Deltaxy**2                      # volume element = voxel volume

        self.xx_grid,self.yy_grid,self.zz_grid=np.meshgrid(self.xy_vec_for_box,
                                                           self.xy_vec_for_box,
                                                           self.z_vec_for_box, indexing="ij")      # box-shaped Cartesian coords CENTRE-ORIGIN

        # Fourier space
        self.Deltakxy=twopi/self.Lxy                                        # voxel side length
        self.Deltakz= twopi/self.Lz
        d3k=self.Deltakxy**2*self.Deltakz                              # volume element / voxel volume
        self.kxy_vec_for_box_corner=twopi*fftfreq(self.Nvox,d=self.Deltaxy) # one Cartesian coordinate axis - non-fftshifted/ corner origin
        self.kz_vec_for_box_corner= twopi*fftfreq(self.Nvoxz,d=self.Deltaz)
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.kxy_vec_for_box_corner,
                                                                                self.kxy_vec_for_box_corner,
                                                                                self.kz_vec_for_box_corner, indexing="ij")               # box-shaped Cartesian coords
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the box generation direction)
        self.kmag_grid_centre=fftshift(self.kmag_grid_corner)
        self.kmag_grid_centre_flat=np.reshape(self.kmag_grid_centre,(self.Nvox**2*self.Nvoxz),order="C")
        self.kmag_grid_corner_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**2*self.Nvoxz,),order="C")
        self.kmag_grid_for_comparison= self.kmag_grid_corner if self.Nvoxz>1 else self.kmag_grid_corner[:,:,0]
              
        self.kpar_column_centre= np.abs(fftshift(self.kz_vec_for_box_corner))                                      # magnitudes of kpar for a representative column along the line of sight (z-like)
        self.kperp_slice_centre= np.sqrt(fftshift(self.kx_grid_corner)**2+fftshift(self.ky_grid_corner)**2)[:,:,0] # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
        kperpgrid3,kpargrid3=np.meshgrid(self.kperp_slice_centre,self.kpar_column_centre, indexing="ij")
        self.kperpgrid3_flat=np.reshape(kperpgrid3,(self.Nvox**2*self.Nvoxz,),order="C")
        self.kpargrid3_flat= np.reshape(kpargrid3, (self.Nvox**2*self.Nvoxz,),order="C")

        if self.Nvoxz>1:
            self.transform_axes=(0,1,2)
            self.d3k=d3k
            self.iftnorm=twopi**3
        else:
            self.transform_axes=(0,1)
            self.d3k=self.Deltakxy**2
            self.iftnorm=twopi**2
        
        # foreground groundwork
        self.wedge_cut=wedge_cut
        if wedge_cut:
            assert(nu_ctr_for_wedge is not None), "an arbitrary box <-> power spectrum translation doesn't require frequency\n"+\
                                                  "info. But, when you opt into the wedge cut, you must override the None\n"+\
                                                  "default in the nu_ctr_for_wedge keyword."
            self.kperp_corner=np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2)
            wedge_kpar_threshold_corner=wedge_kpar(nu_ctr_for_wedge,self.kperp_corner)
            self.voxels_in_wedge_corner=self.kz_grid_corner<=wedge_kpar_threshold_corner
        
        # rng management
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        else:
            self.rng=np.random.default_rng()

        # if P_fid was passed, establish its values on the k grid (helpful when generating a box)
        self.k_fid=k_fid
        self.kind=kind
        self.avoid_extrapolation=avoid_extrapolation
        if (self.P_fid is not None and self.k_fid is not None):
            if (len(self.P_fid.shape)==1): # truly 1d fiducial power spec (by this point, even CAMB-like shapes have been reshuffled)
                self.resample_P_fid_on_grid()
            else:
                assert 1==0, "not yet implemented"
        
        # binning considerations
        self.bin_each_realization=bin_each_realization
        self.binning_mode=binning_mode

        bin_denom=2.2
        if Nkperp==0:
            Nkperp=int(Nvox/bin_denom)
        if Nkpar==0:
            Nkpar=int(Nvoxz/bin_denom)
        self.Nkperp=Nkperp # the number of bins to put in power spec realizations you construct
        self.Nkpar=Nkpar
        self.kmax_box_xy= pi/self.Deltaxy
        self.kmax_box_z=  pi/self.Deltaz
        self.kmin_box_xy= twopi/self.Lxy
        self.kmin_box_z=  twopi/self.Lz
        
        # voxel grids for cyl binning
        if (self.Nkpar is not None and self.Nkpar!=0):
            kperpbins=np.linspace(0,self.kmax_box_xy, self.Nkperp+1)
            bw=kperpbins[1]-kperpbins[0]
            self.kperpbins=kperpbins +0.5*bw
            
            kparbins=np.linspace(0,self.kmax_box_z-self.kmin_box_z, self.Nkpar+1)
            bw=kparbins[1]-kparbins[0]
            self.kparbins=kparbins +0.5*bw
            
            self.kperpbins_grid,self.kparbins_grid=np.meshgrid(self.kperpbins,self.kparbins, indexing="ij")
        
            self.bins_to_use=[self.kperpbins.value,self.kparbins.value]
            self.coords_to_use=[self.kperpgrid3_flat.value,self.kpargrid3_flat.value]

        else: # calling them perp bins for class reasons but they are just sph
            kmax_box=np.max([self.kmax_box_xy.value,self.kmax_box_z.value])*self.kmax_box_xy.unit # ignore the voxels outside the sphere that is contained by the box's larger axis but probably exceeds its smaller axis
            
            kperpbins=np.linspace(0,kmax_box, self.Nkperp+1)
            bw=kperpbins[1]-kperpbins[0]
            self.kperpbins=kperpbins +0.5*bw

            self.kparbins=None
            self.Nkpar=0

            self.bins_to_use=[self.kperpbins.value]
            self.coords_to_use=self.kmag_grid_centre_flat.value
            
        # tapering/apodization
        taper_xy=np.ones(self.Nvox)
        taper_z=np.ones(self.Nvoxz)
        if radial_taper is not None:
            taper_z= Blackman_Harris_safe_for_FFT(Nvoxz) # confirmed to be centre-, not corner-origin
        if image_taper is not None:
            taper_xy=Blackman_Harris_safe_for_FFT(Nvox)
        taper_xxx,taper_yyy,taper_zzz=np.meshgrid(taper_xy,taper_xy,taper_z, indexing="ij")
        taper_xyz_product=taper_xxx*taper_yyy*taper_zzz
        self.taper_xyz_centre=taper_xyz_product
        self.taper_xyz_corner=ifftshift(taper_xyz_product)

        # primary beam
        self.primary_beam_num=primary_beam_num
        self.primary_beam_aux_num=primary_beam_aux_num
        self.primary_beam_type_num=primary_beam_type_num
        self.primary_beam_modes=primary_beam_modes # fi and sy beams assumed to be sampled at the same modes, if these are passed
        if (self.primary_beam_num is not None): # non-identity FIDUCIAL primary beam
            if (self.primary_beam_type_num=="Gaussian" or self.primary_beam_type_num=="Airy"):
                self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_aux_num
                evaled_primary_num=  self.primary_beam_num(self.xx_grid,self.yy_grid,self.fwhm_x,  self.fwhm_y,  self.r0)     
            elif (self.primary_beam_type_num=="manual"):
                assert np.all(self.primary_beam_num>=0.)
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable
                    primary_beam_num.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ValueError("conflicting info") 
                if self.primary_beam_modes is None:
                    raise ValueError("not enough info")

                x_primary,y_primary,z_primary=primary_beam_modes
                x_primary*=u.Mpc
                y_primary*=u.Mpc
                z_primary*=u.Mpc
                x_have_lo=x_primary[0]
                x_have_hi=x_primary[-1]
                y_have_lo=y_primary[0]
                y_have_hi=y_primary[-1]
                z_have_lo=z_primary[0]
                z_have_hi=z_primary[-1]
                xy_want_lo=self.xy_vec_for_box[0]
                xy_want_hi=self.xy_vec_for_box[-1]
                z_want_lo=self.z_vec_for_box[0]
                z_want_hi=self.z_vec_for_box[-1]
                if (xy_want_lo<x_have_lo):
                    extrapolation_warning("low x",   xy_want_lo,  x_have_lo)
                if (xy_want_hi>x_have_hi):
                    extrapolation_warning("high x",   xy_want_hi,  x_have_hi)
                if (xy_want_lo<y_have_lo):
                    extrapolation_warning("low y",   xy_want_lo,  y_have_lo)
                if (xy_want_hi>y_have_hi):
                    extrapolation_warning("high y",   xy_want_hi,  y_have_hi)
                if (z_want_lo<z_have_lo):
                    extrapolation_warning("low z",   z_want_lo,  z_have_lo)
                if (z_want_hi>z_have_hi):
                    extrapolation_warning("high z",   z_want_hi,  z_have_hi)
                to_eval_at=np.array([self.xx_grid.value,self.yy_grid.value,self.zz_grid.value]).T
                evaled_primary_num=RGI(primary_beam_modes,self.primary_beam_num,
                                       bounds_error=False,fill_value=None)(to_eval_at).T
                self.evaled_primary_num=evaled_primary_num
            
            else:
                raise ValueError("not yet implemented")
            evaled_primary_use_for_eff_vol=evaled_primary_num # gets overwritten (as it should) if there's a separate _den beam
        
        self.primary_beam_den=primary_beam_den
        self.primary_beam_aux_den=primary_beam_aux_den
        self.primary_beam_type_den=primary_beam_type_den
        self.primary_beam_modes=primary_beam_modes # _fi and _co_fi_sy_xx assumed to be sampled at the same modes, if this is the case
        if (self.primary_beam_den is not None): # non-identity PERTURBED primary beam
            if (self.primary_beam_type_den=="Gaussian" or self.primary_beam_type_den=="Airy"):
                self.fwhm_x,self.fwhm_y,r0=self.primary_beam_aux_den
                self.r0=r0*u.Mpc
                evaled_primary_den=  self.primary_beam_den(self.xx_grid,self.yy_grid,self.fwhm_x,  self.fwhm_y,  self.r0)                
            elif (self.primary_beam_type_den=="manual"):
                assert np.all(self.primary_beam_den>=0.)
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable... so, no danger of attribute errors
                    primary_beam_den.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ValueError("conflicting info") 
                if self.primary_beam_modes is None:
                    raise ValueError("not enough info")

                evaled_primary_den=RGI(primary_beam_modes,self.primary_beam_den,
                                       bounds_error=False,fill_value=None)(np.array([self.xx_grid.value,self.yy_grid.value,self.zz_grid.value]).T).T
                evaled_primary_use_for_eff_vol=evaled_primary_den

            else:
                evaled_primary_den=None    

        else:                               # identity primary beam
            evaled_primary_num=1.
            evaled_primary_den=1.
            evaled_primary_use_for_eff_vol=np.ones(self.box_shape)
        self.evaled_primary_den=evaled_primary_den
        self.evaled_primary_num=evaled_primary_num

        self.effective_volume=np.sum((evaled_primary_use_for_eff_vol*self.taper_xyz_centre)**2*self.d3r)
        assert self.effective_volume>0
        
        if (self.T_pristine is not None):
            self.T_primary=self.T_pristine*self.evaled_primary_num
        
        # strictness control for realization averaging
        self.frac_tol=frac_tol
        self.N_realizations=int(np.round(self.frac_tol**-2))

        # P_MC_complete interpolation bins
        self.kperpbins_interp=kperpbins_interp
        self.kparbins_interp=kparbins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        self.P_unbinned_running_sum=np.zeros(self.box_shape)*u.mK**2*u.Mpc**3
        self.P_MC_complete=P_MC_complete
        self.P_interp=None
        self.MC_not_complete=None
        if self.Nkpar>0 and self.Nkpar is not None:
            self.N_cumul=np.zeros((self.Nkperp,self.Nkpar))
        else:
            self.N_cumul=np.zeros((self.Nkperp,))

    def resample_P_fid_on_grid(self): # resample a 1D power spec on a 3D grid. to break these symmetries, you can do a bit of reverse-engineering: do what you want to the box -> update the T_pristine attribute -> form a power spec using the same cosmo_stats object -> save that unbinned power spec as P_fid_grid -> continue with your Monte Carlo or whatever
        assert(len(self.k_fid)==len(self.P_fid) or len(self.k_fid)==len(self.P_fid.T))
        sort_array=np.argsort(self.kmag_grid_corner_flat)
        kmag_grid_corner_flat_sorted=self.kmag_grid_corner_flat[sort_array]
        P_fid_flattened_box=np.zeros(self.Nvox**2*self.Nvoxz)
        interpolator=RGI((self.k_fid.value,),self.P_fid.value,method="cubic",
                          bounds_error=False,fill_value=None)
        P_fid_flattened_box[sort_array]=interpolator(kmag_grid_corner_flat_sorted.value[:,None])
        P_fid_box=np.reshape(P_fid_flattened_box,self.box_shape,order="C")
        P_fid_box[P_fid_box<0]=0.
        P_fid_box[np.isnan(P_fid_box)]=0.
        self.P_fid_box=P_fid_box
            
    def generate_P(self,send_to_P_fid:bool=False,T_use=None): # from a box of temperature field values
        if (T_use is None or T_use.lower()=="primary"):
            assert self.evaled_primary_num is not None
            if self.T_primary is None:
                self.T_primary=self.T_pristine*self.evaled_primary_num
            T_use=self.T_primary
        else:
            T_use=self.T_pristine
        T_use=T_use.to(u.mK)
        
        T_tilde=fftshift( fftn( 
                                # ifftshift(T_use*self.taper_xyz_centre)*self.d3r, # shouldn't make a difference
                                ifftshift(T_use)*self.taper_xyz_corner*self.d3r,
                                s=self.box_shape, axes=self.transform_axes, norm="backward"        
                              ) 
                        ) # centre-origin
        modsq_T_tilde=np.abs(T_tilde)**2 *T_use.unit**2*self.d3r.unit**2
        P_unbinned=modsq_T_tilde/self.effective_volume # box-shaped, but calculated according to the power spectrum estimator equation

        self.P_unbinned=P_unbinned # centre-origin
        
        if self.bin_each_realization:
            self.bin_power()
        
        if send_to_P_fid: # if generate_P was called speficially to have a spec from which all future box realizations will be generated
            self.P_fid_box=ifftshift(self.P_unbinned) # P_fid_box is corner-origin
        else:             # the "normal" case where you're just accumulating a realization (any binning happens at the end of the Monte Carlo)
            self.P_unbinned_running_sum+=P_unbinned

    def bin_power(self,power_to_bin=None):
        if power_to_bin is None:
            power_to_bin=self.P_unbinned
        power_to_bin_flat=np.reshape(power_to_bin,(self.Nvox**2*self.Nvoxz,))

        P_binned=binned_statistic_dd(sample=self.coords_to_use, values=power_to_bin_flat,
                                     bins=self.bins_to_use, statistic="mean").statistic
        
        N_cumul= binned_statistic_dd(sample=self.coords_to_use, values=power_to_bin_flat,
                                     bins=self.bins_to_use, statistic="count").statistic

        P_binned[np.isnan(P_binned)]=0.
        self.P_binned=P_binned
        N_cumul[np.isnan(N_cumul)]=0.
        self.N_cumul=N_cumul
    
    def generate_GRF(self): # Gaussian random field realization consistent with a power spectrum of choice
        assert self.Nkperp<self.Nvox, "Nvox should be >= Nkperp"
        assert self.Nkpar<self.Nvoxz, "Nvoxz should be >= Nkpar"
        if (self.P_fid_box is None):
            try:
                self.generate_P(send_to_P_fid=True) # even if you start with a random realization, it'll be helpful to have a power spec summary stat to generate future realizations
            except: # something goes wrong in the P_fid calculation
                raise ValueError("not enough info")
        
        if self.P_fid_box is None:
            self.resample_P_fid_on_grid()
        sigmas=np.sqrt(self.physical_volume*self.P_fid_box/2.) # from inverting the estimator equation and turning variances into std devs
        zero_on_the_last_axis=tuple(slice(0,l) for l in self.box_shape[:-1]) + (slice(0,1),)
        sigmas[zero_on_the_last_axis]*=np.sqrt(2) # scipy irfftn puts all the variance into the real component of the half-axis slice of the last axis it transforms in the box. I need to anticipate this by giving those voxels' real components all the variance! (Nothing will be overcounted because the imag part is thrown away)
        
        sigmas[self.kmag_grid_for_comparison==0.]=0. # enforce zero-mean. This point is self-conjugate anyway!!
        T_tilde_Re,T_tilde_Im=self.rng.normal(loc=0.*sigmas,scale=sigmas,size=np.insert(sigmas.shape,0,2)) # corner-origin
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        if self.wedge_cut:
            T_tilde[self.voxels_in_wedge_corner]=0.

        T=fftshift(irfftn(T_tilde*self.d3k.value,
                          s=self.box_shape,
                          axes=self.transform_axes,
                          norm="forward"))/self.iftnorm # handle fftshiftedness, T real-valued and box-shaped, cosmology Fourier convention

        T*=u.mK # centre_origin
        if self.fg_box is not None:
            T+=self.fg_box
        
        self.T_pristine=T
        if self.primary_beam_num is not None:
            self.T_primary=T*self.evaled_primary_num

    def power_Monte_Carlo(self,interfix:str=""): # since box generation is not deterministic
        if self.P_fid_box is None:
            self.generate_P(send_to_P_fid=True)
        self.MC_not_complete=True
        if self.primary_beam_num is None:
            T_use="pristine"
        else: 
            T_use="primary"
        i=0

        t0=time.time()
        for i in range(self.N_realizations):
            self.generate_GRF()
            self.generate_P(T_use=T_use)
            ti=time.time()
            if ((ti-t0)>3600): # actually save the realizations every hour
                np.save("P_"+interfix+"_MC_incomplete.npy",self.P_unbinned_running_sum.value/i)
                t0=time.time()

        P_unbinned_MC_complete=self.P_unbinned_running_sum/self.N_realizations
        self.P_unbinned_MC_complete=P_unbinned_MC_complete
        self.bin_power(power_to_bin=P_unbinned_MC_complete)
        P_binned_MC_complete=self.P_binned
        self.P_binned_MC_complete=P_binned_MC_complete*self.P_unbinned_running_sum.unit

        self.N_per_realization=self.N_cumul/self.N_realizations

    def interpolate_P(self,use_P_fid:bool=False):
        if use_P_fid:
            self.P_MC_complete=self.P_fid
        else:
            if (self.P_MC_complete is None):
                print("WARNING: P_MC_complete DNE yet. \nAttempting to calculate it now...")
                self.power_Monte_Carlo()
            if (self.kperpbins_interp is None):
                raise ValueError("not enough info")

        if (self.kparbins_interp is not None):
            kpar_have_lo=  self.kperpbins[0]
            kpar_have_hi=  self.kperpbins[-1]
            kperp_have_lo= self.kparbins[0]
            kperp_have_hi= self.kparbins[-1]

            kpar_want_lo=  self.kperpbins_interp[0]
            kpar_want_hi=  self.kperpbins_interp[-1]
            kperp_want_lo= self.kparbins_interp[0]
            kperp_want_hi= self.kparbins_interp[-1]

            if (kpar_want_lo<kpar_have_lo):
                extrapolation_warning("low kpar",   kpar_want_lo,  kpar_have_lo)
            if (kpar_want_hi>kpar_have_hi):
                extrapolation_warning("high kpar",  kpar_want_hi,  kpar_have_hi)
            if (kperp_want_lo<kperp_have_lo):
                extrapolation_warning("low kperp",  kperp_want_lo, kperp_have_lo)
            if (kperp_want_hi>kperp_have_hi):
                extrapolation_warning("high kperp", kperp_want_hi, kperp_have_hi)
            modes_defined_at=(self.kperpbins_grid,self.kparbins_grid)
            modes_to_eval_at=(self.kperp_interp_grid,self.kpar_interp_grid).T
        else:
            k_have_lo=self.kperpbins[0]
            k_have_hi=self.kperpbins[-1]
            k_want_lo=self.kperpbins_interp[0]
            k_want_hi=self.kperpbins_interp[-1]
            if (k_want_lo<k_have_lo):
                extrapolation_warning("low k",k_want_lo,k_have_lo)
            if (k_want_hi>k_have_hi):
                extrapolation_warning("high k",k_want_hi,k_have_hi)
            modes_defined_at=(self.kperpbins.value,)
            modes_to_eval_at=(self.kperpbins_interp.value,)
        P_interpolator=RGI(modes_defined_at,self.P_MC_complete,
                           bounds_error=self.avoid_extrapolation,fill_value=None)
        P_interp=P_interpolator(modes_to_eval_at)
        if self.kparbins_interp is not None:
            P_interp=P_interp.T # anticipate the RGI behaviour
        self.P_interp=P_interp
####################################################################################################################################################################################################################################

def beam_type_distribution(N_NS,N_EW,N_types,distribution="random",frame_width=2):
    N_ant=N_NS*N_EW
    if N_types>0:
        if distribution=="random":
            rng=np.random.default_rng()
            per_antenna_types=rng.integers(0,N_types,size=(N_ant,))
        elif distribution=="corner":
            if N_types!=4:
                raise ValueError("conflicting info") # in order to use corner mode, you need four fiducial beam types
            per_antenna_types=np.zeros((N_NS,N_EW))
            half_NS=N_NS//2
            half_EW=N_EW//2
            per_antenna_types[:half_NS,half_EW:]=1
            per_antenna_types[half_NS:,:half_EW]=2
            per_antenna_types[half_NS:,half_EW:]=3 # the quarter of the array with no explicit overwriting keeps its idx=0 (as necessary)
        elif distribution=="diagonal":
            raise ValueError("not yet implemented")
        elif distribution=="column":
            per_antenna_types=np.zeros((N_NS,N_EW))
            for i in range(1,N_types):
                per_antenna_types[:,i::N_types]=i
        elif distribution=="frame":
            per_antenna_types=np.zeros((N_NS,N_EW),dtype=np.int8)
            if N_types>1: # stricter threshold for this case based on where the random numbers come from
                rng=np.random.default_rng()
                sh=per_antenna_types.shape
                sz=per_antenna_types.size
                per_antenna_types[~np.isin(np.arange(sz).reshape(sh), 
                                           np.arange(sz).reshape(sh)[frame_width:-frame_width, 
                                                                     frame_width:-frame_width])]=rng.integers(1,high=N_types,
                                                                                                              size=2*(N_NS+N_EW)-4)
        else:
            raise ValueError("beam distribution pattern not yet implemented")
        
        per_antenna_types=np.reshape(per_antenna_types,(N_ant,),order="C")
    else:
        per_antenna_types=np.zeros(N_ant)

    return per_antenna_types

"""
this class helps compute numerical windowing boxes for brightness temp boxes resulting
from primary beams that have the flexibility to differ on a per-antenna basis.
"""

class per_antenna(beam_effects): # still fairly tailored to rectangular arrays
    def __init__(self,
                 mode:str="full",                                                  # run a simulation for full or pathfinder CHORD?
                 b_NS:float=b_NS,b_EW:float=b_EW,                                  # N-S and E-W baseline lengths (m)
                 offset_rad:float=def_offset,                                      # (astropy-unitless because this class expects rad) CHORD is aligned with magnetic, not geographical north, so, when mathematically constructing the uv coverage, rotate the rectangular array grid
                 observing_dec:float=def_observing_dec,                            # declination to observe at (º)
                 N_fiducial_beam_types:int=N_fid_beam_types,N_pert_types:int=0,    # number of fiducial beam types; number of perturbed beam types
                 N_pbws_pert:int=0,                                                # number of antennas with perturbed primary beams
                 pbw_pert_frac:float=def_pbw_pert_frac,                            # ** fractional perturbation to the primary beam width
                 N_timesteps:float=def_N_timesteps,                                # number of timesteps in rotation synthesis
                 nu_ctr:float=nu_HI_z0,                                            # central frequency of the survey of interest
                 pbw_fidu:float=None,                                              # ** fiducial primary beam width (defaults to a diffraction-limited Airy beam, modulo any differences imposed by the number of fiducial beam types)
                 N_grid_pix:int=def_PA_N_grid_pix,                                 # number of pixels per side of the gridded uv plane
                 Delta_nu:float=CHORD_channel_widco_MHz,                           # channel width in frequency (MHz)
                 distribution:str="random",                                        # distribution of per-antenna systematics. the options I've encoded for now are random, column, and corner, based on where the fiducial beam types are placed within the array
                 fidu_types_prefactors=None,                                       # ** multiplicative prefactors by which the different fiducial beam types differ from the physics-informed fiducial beam width for a given frequency channel
                 per_channel_systematic=None,                                      # apply a systematic that corrupts the 1/lambda scaling of the beam width? options encoded so far are sporadic (multiply the beam widths for a contiguous chunk of frequency channels by a different multiplicative prefactor for the different fiducial beam types) and D3A-like (noise + too wide at low frequencies... inspired by early three-dish transit beam measurements)
                 per_chan_syst_facs=None,                                          # the multiplicative prefactors for the sporadic per-antenna systematic (see above)
                 evol_restriction_threshold:float=def_evol_restriction_threshold,  # max \delta z/z you will tolerate for the survey of interest and still consider the box close enough to coeval
    
                 sub_ensemble_of_CST_beams=None,                                   # array-like with shape (N_CST_types, N_pointing_errors+1, N_CST_xy, N_CST_xy, N_CST_freqs)
                 CST_xy=None,CST_freqs=None                                        # domain of each CST box in the ensemble. this domain is currently assumed to be the same for each box (not very rigorous/robust, but in practice, if you're running a simulation for a given survey frequency, it would be fairly pathological/ unintuitive/ anti–Occam's razor to get these boxes from CST slices at different frequencies. I guess the practical guidance/takeaway here is that my initial implementation will not support getting different boxes from different CST box resolutions)
                 ): 
                                                                                   # ** args unnecessary for per-antenna CST
        # array and observation geometry
        self.N_fiducial_beam_types=N_fiducial_beam_types
        self.N_pert_types=N_pert_types
        self.N_pbws_pert=N_pbws_pert
        self.pbw_pert_frac=pbw_pert_frac
        self.per_channel_systematic=per_channel_systematic
        self.N_timesteps=N_timesteps
        self.N_grid_pix=N_grid_pix
        self.distribution=distribution
        self.evol_restriction_threshold=evol_restriction_threshold
        self.Delta_nu=Delta_nu
        N_NS=N_NS_full
        N_EW=N_EW_full
        self.DRAO_lat=DRAO_lat
        if (mode=="pathfinder"):
            N_NS=N_NS//2
            N_EW=N_EW//2
        N_ant=N_NS*N_EW
        N_bl=N_ant*(N_ant-1)//2
        self.nu_ctr_MHz=nu_ctr
        self.nu_ctr_Hz=nu_ctr.to(u.Hz)
        self.Dc_ctr=comoving_distance(nu_HI_z0/nu_ctr-1)
        # bmax=np.sqrt(N_NS*b_NS**2+N_EW*b_EW**2)
        # self.N_hrs=synthesized_beam_crossing_time(self.nu_ctr_Hz,bmax=bmax,dec=observing_dec) # freq needs to be in Hz
        self.N_hrs=hrs_per_night
        self.lambda_obs=c/self.nu_ctr_Hz
        if (pbw_fidu is None):
            pbw_fidu=self.lambda_obs/D
            pbw_fidu=[pbw_fidu,pbw_fidu]
        self.pbw_fidu=np.array(pbw_fidu)
        
        # antenna positions xyz
        antennas_EN=np.zeros((N_ant,2))
        for i in range(N_NS):
            for j in range(N_EW):
                antennas_EN[i*N_EW+j,:]=[j*b_EW.value,i*b_NS.value]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        try:
            offset_rad.to(u.rad)
        except:
            offset_rad=offset_rad*u.rad
        offset_from_latlon_rotmat=np.array([[np.cos(offset_rad),-np.sin(offset_rad)],
                                            [np.sin(offset_rad), np.cos(offset_rad)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1), order="C") # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T
        
        # line-of-sight quantities
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/self.Delta_nu)
        self.N_chan=N_chan
        nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(nu_hi,nu_lo,N_chan) # decr.
        surv_channels_Hz=surv_channels_MHz.to(u.Hz)
        surv_wavelengths=c/surv_channels_Hz # incr.
        self.surv_wavelengths=surv_wavelengths.decompose()
        z_channels=nu_HI_z0/surv_channels_MHz-1.
        comoving_distances_channels=np.asarray([comoving_distance(chan).value for chan in z_channels]) # incr.
        self.comoving_distances_channels=comoving_distances_channels*u.Mpc
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]
        self.surv_channels_MHz=surv_channels_MHz

        # helper args specific to Gaussian or CST calculations
        self.CST=False if sub_ensemble_of_CST_beams is None else True
        if self.CST:
            self.CST_xy=CST_xy
            CST_Delta_xy=CST_xy[1]-CST_xy[0]
            CST_dxdy=(CST_Delta_xy)**2
            self.CST_dxdy=CST_dxdy
            self.uvbins_CST=fftshift(fftfreq(len(CST_xy),d=CST_Delta_xy))
            self.CST_freqs=CST_freqs
            self.N_CST_xy=len(CST_xy)
            self.N_CST_freqs=len(CST_freqs)

            if type(sub_ensemble_of_CST_beams) is not list: # can't use .ndim because it doesn't behave well for the inhomog arrays of the else
                fidu_box=sub_ensemble_of_CST_beams

                self.all_boxes=np.expand_dims(sub_ensemble_of_CST_beams,axis=0)

                N_total_beam_types=1
                self.N_total_beam_types=1
            else:
                fidu_box,syst_boxes=sub_ensemble_of_CST_beams # should be unpackable into two arrays:
                assert fidu_box.ndim==3 and syst_boxes.ndim==5 # one box and one "2D array of 3D boxes"
                self.N_CST_types,self.N_max_pointing_errors,Nxy,_,Nz=syst_boxes.shape

                # figure out the actual number of beam types
                N_pointing_errors_per_CST_case=np.zeros(self.N_CST_types,dtype=int)
                for i in range(self.N_CST_types):
                    for j in range(self.N_max_pointing_errors):
                        if np.all(np.isclose(syst_boxes[i,j,:,:,:],0.)):
                            N_pointing_errors_per_CST_case[i]=j # only the (j-1)st case is meaningful, but that is in zero-based indexing, not one-based counting.
                self.N_pointing_errors_per_CST_case=N_pointing_errors_per_CST_case
                N_total_beam_types=int(np.sum(N_pointing_errors_per_CST_case)+1) # +1 to include the fiducial beam
                self.N_total_beam_types=N_total_beam_types

                # store the actual beam types as a list of boxes, not 2D array of boxes + standalone box
                all_boxes=np.zeros((N_total_beam_types,Nxy,Nxy,Nz))
                all_boxes[0]=fidu_box
                k=1 # start at 1 because 0 was already taken up by the fidu box
                for i in range(self.N_CST_types):
                    for j in range(N_pointing_errors_per_CST_case[i]):
                        box_to_add=syst_boxes[i,j,:,:,:]
                        all_boxes[k]=box_to_add
                        k+=1
                self.all_boxes=all_boxes
            self.pb_types=beam_type_distribution(N_NS,N_EW,N_total_beam_types, distribution=self.distribution)

        else:
            pbw_fidu_types=beam_type_distribution(N_NS,N_EW,self.N_fiducial_beam_types, distribution=self.distribution)
            pbw_pert_types=beam_type_distribution(N_NS,N_EW,self.N_pert_types,          distribution="random")

            # K-PERP / ARRAY LAYOUT THINGS
            if fidu_types_prefactors is None:
                fidu_types_prefactors=np.ones(N_fiducial_beam_types)
            self.fidu_types_prefactors=fidu_types_prefactors

            epsilons=np.zeros(N_pert_types+1)
            if (self.N_pbws_pert>0):
                if (self.N_pert_types>1):
                    rng=np.random.default_rng()
                    random_draw=rng.uniform(size=(N_pert_types,))
                    random_perturbations=random_draw*self.pbw_pert_frac
                    epsilons[1:]=random_perturbations
                else: 
                    epsilons[1]=self.pbw_pert_frac
                rng=np.random.default_rng()
                indices_of_ants_w_pert_pbws=rng.randint(0,N_ant,size=self.N_pbws_pert) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
            else:
                indices_of_ants_w_pert_pbws=None
            self.indices_of_ants_w_pert_pbws=indices_of_ants_w_pert_pbws
            self.epsilons=epsilons
            self.per_chan_syst_facs=per_chan_syst_facs

            # K-PAR / CHROMATICITY THINGS
            surv_beam_widths=dif_lim_prefac*surv_wavelengths/D # incr.
            surv_beam_widths=surv_beam_widths.decompose()
            self.surv_beam_widths=surv_beam_widths
            plt.figure()
            plt.plot(surv_channels_MHz,surv_beam_widths,label="diffraction-limited Airy FWHM")    
            per_chan_syst_name="None"        
            if self.per_channel_systematic=="early_transit_measurement_like":
                surv_beam_widths=(surv_beam_widths)**1.2 # keep things dimensionless, but use a steeper decay
                noise_bound_lo=0.95
                noise_bound_hi=1.05
                rng=np.random.default_rng()
                noise_frac=(noise_bound_hi-noise_bound_lo)*rng.random_sample(size=(N_chan,))+noise_bound_lo # random_sample draws fall within [0,1) but I want values between [0.75,1.25)*(that channel's beam width)
                surv_beam_widths*=noise_frac
                per_chan_syst_name="early_transit_measurement_like"
            elif self.per_channel_systematic=="sporadic":
                bad=np.ones(N_chan)
                per_chan_syst_locs=[slice(  N_chan//5,    N_chan//4+1,1), slice(  N_chan//2,  7*N_chan//13+1,1),slice(11*N_chan//12,   None,        1),
                                    slice(7*N_chan//9, 10*N_chan//11  ,1),slice(  N_chan//10,   N_chan//9+ 1,1),slice( 2*N_chan//3 , 5*N_chan//6   ,1),
                                    slice(4*N_chan//5,  9*N_chan//10  ,1),slice(  None,         N_chan//9,   1),slice( 8*N_chan//11, 4*N_chan//5   ,1),
                                    slice(5*N_chan//6,  7*N_chan//8   ,1)] # (not user-specifiable yet)
                per_chan_syst_name="sporadic_"
                for i,fac_i in enumerate(self.per_chan_syst_facs):
                    loc_i=per_chan_syst_locs[i]
                    bad[loc_i]=fac_i
                    per_chan_syst_name=per_chan_syst_name+str(fac_i)+"_"
                surv_beam_widths*=bad
            elif self.per_channel_systematic is None:
                pass
            else:
                raise ValueError("not yet implemented")
            if self.per_channel_systematic is not None:
                plt.plot(surv_channels_MHz,surv_beam_widths,label="chromaticity systematic–laden")
            plt.xlabel("frequency (MHz)")
            plt.ylabel("beam FWHM (rad)")
            plt.title("reference beam widths by frequency bin")
            plt.legend()
            plt.savefig("beam_chromaticity_slice_"+str(self.nu_ctr_MHz)+"_MHz_"+per_chan_syst_name+".png")
            plt.close()
            self.per_chan_syst_name=per_chan_syst_name

        # ungridded instantaneous uv-coverage (baselines in xyz)
        # second use of the loop: iterate over baselines to make arrays of beam type indices     
        uvw_inst=np.zeros((N_bl,3))
        if self.CST:
            indices_of_constituent_ant_pb_types=np.zeros((N_bl,2))
        else:
            indices_of_constituent_ant_pb_fidu_types=np.zeros((N_bl,2))
            indices_of_constituent_ant_pb_pert_types=np.zeros((N_bl,2))
        k=0
        for i in range(N_ant):
            for j in range(i+1,N_ant):
                uvw_inst[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
                if self.CST:
                    indices_of_constituent_ant_pb_types[k]=[self.pb_types[i],self.pb_types[j]]
                else: 
                    indices_of_constituent_ant_pb_fidu_types[k]=[pbw_fidu_types[i],pbw_fidu_types[j]]
                    indices_of_constituent_ant_pb_pert_types[k]=[pbw_pert_types[i],pbw_pert_types[j]]
                k+=1
        uvw_inst=np.vstack((uvw_inst,-uvw_inst))
        self.uvw_inst=uvw_inst
        if self.CST:
            indices_of_constituent_ant_pb_types=np.vstack((indices_of_constituent_ant_pb_types,indices_of_constituent_ant_pb_types)) # get the opposite-permutation baselines for free
            self.indices_of_constituent_ant_pb_types=indices_of_constituent_ant_pb_types
        else:
            indices_of_constituent_ant_pb_fidu_types=np.vstack((indices_of_constituent_ant_pb_fidu_types,indices_of_constituent_ant_pb_fidu_types))
            indices_of_constituent_ant_pb_pert_types=np.vstack((indices_of_constituent_ant_pb_pert_types,indices_of_constituent_ant_pb_pert_types))
            self.indices_of_constituent_ant_pb_fidu_types=indices_of_constituent_ant_pb_fidu_types
            self.indices_of_constituent_ant_pb_pert_types=indices_of_constituent_ant_pb_pert_types
        print("computed ungridded instantaneous uv-coverage")

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*self.N_hrs/12
        hour_angles=np.linspace(0,hour_angle_ceiling,self.N_timesteps)
        thetas=hour_angles.value*15*np.pi/180*u.rad # don't use built-in astropy conversions for this because it won't realize my hr<->rad conversion is about the rotation rate of the earth
        
        try:
            observing_dec.to(u.rad)
        except:
            observing_dec=observing_dec*u.rad
        zenith=np.array([np.cos(observing_dec),0,np.sin(observing_dec)]) # Jon math redux
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        project_to_dec=np.vstack([east,north])

        uv_synth=np.zeros((2*N_bl,2,self.N_timesteps))
        for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
            accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                        [-np.sin(theta),np.cos(theta),0],
                                        [ 0,            0,            1]])
            uvw_rotated=uvw_inst@accumulate_rotation
            uvw_projected=uvw_rotated@project_to_dec.T
            uv_synth[:,:,i]=uvw_projected/self.lambda_obs
        self.uv_synth=uv_synth
        print("synthesized rotation")        

    def calc_dirty_image(self, Npix:int=1024, pbw_fidu_use:float=None,tol:float=img_bin_tol):
        if pbw_fidu_use is None: # otherwise, use the one that was passed
            pbw_fidu_use=self.pbw_fidu
        all_ungridded_u=self.uv_synth[:,0,:]
        all_ungridded_v=self.uv_synth[:,1,:]
        uvmagmax=tol*np.max([np.max(np.abs(all_ungridded_u)),
                             np.max(np.abs(all_ungridded_v))])

        uvmagmin=2*uvmagmax/Npix
        thetamax=1/uvmagmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
        self.thetamax=thetamax

        uvbins=np.linspace(-uvmagmax,uvmagmax,Npix)
        d2u=uvbins[1]-uvbins[0]
        self.d2u=d2u
        uubins,vvbins=np.meshgrid(uvbins,uvbins, indexing="ij")
        uvplane=np.zeros((Npix,Npix),dtype="complex128") # 0.*uubins
        uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])
        pad_lo,pad_hi=get_padding(Npix)

        if self.CST:
            for i in range(self.N_total_beam_types):
                type_i=self.pb_types[i]
                for j in range(i+1):
                    type_j=self.pb_types[j]

                    here=(self.indices_of_constituent_ant_pb_types[:,0]==i
                          )&(self.indices_of_constituent_ant_pb_types[:,1]==j)
                    u_here=self.uv_synth[here,0,:] # [N_bl,2,N_hr_angles]
                    v_here=self.uv_synth[here,1,:]
                    N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                    N_here=N_bl_here*N_hr_angles_here
                    reshaped_u=np.reshape(u_here,N_here,order="C")
                    reshaped_v=np.reshape(v_here,N_here,order="C")
                    gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                    LoS_idx=np.argmin(np.abs(self.nu_obs-self.CST_freqs))
                    image_i=self.all_boxes[type_i,:,:,LoS_idx] # [N_total_beam_types, Nxy, Nxy, Nz]
                    image_j=self.all_boxes[type_j,:,:,LoS_idx]
                    assert np.all(image_i>=0.), "image i beam slice should be entirely nonnegative"
                    assert np.all(image_j>=0.), "image j beam slice should be entirely nonnegative"
                    image_ij=np.sqrt(image_i*image_j) # geo mean of the beams of this baseline's two constituent antennas. still on initial CST grid
                    uv_ij=fftshift(fftn(ifftshift(image_ij*self.CST_dxdy),norm="forward")) # FT to put in uv space 
                    interpolator_Re=RBS(self.uvbins_CST,self.uvbins_CST, uv_ij.real)
                    interpolator_Im=RBS(self.uvbins_CST,self.uvbins_CST, uv_ij.imag)
                    kernel=interpolator_Re(uvbins_use[:-1],uvbins_use[:-1])+1j*interpolator_Im(uvbins_use[:-1],uvbins_use[:-1])
                    
                    kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge")
                    convolution_here=convolve(kernel_padded,gridded,mode="valid") # beam-smeared version of the uv-plane for this perturbation permutation
                    uvplane+=convolution_here
        else:
            for i in range(self.N_pert_types+1):
                eps_i=self.epsilons[i]
                for j in range(i+1):
                    eps_j=self.epsilons[j]
                    for k in range(self.N_fiducial_beam_types):
                        fidu_type_k=self.fidu_types_prefactors[k]
                        for l in range(k+1):
                            fidu_type_l=self.fidu_types_prefactors[l]

                            here=(self.indices_of_constituent_ant_pb_pert_types[:,0]==i
                                )&(self.indices_of_constituent_ant_pb_pert_types[:,1]==j
                                    )&(self.indices_of_constituent_ant_pb_fidu_types[:,0]==k
                                        )&(self.indices_of_constituent_ant_pb_fidu_types[:,1]==l) # which baselines to treat during this loop trip... pbws has shape (N_bl,2) ... one column for antenna a and the other for antenna b
                            u_here=self.uv_synth[here,0,:] # [N_bl,3,N_hr_angles]
                            v_here=self.uv_synth[here,1,:]
                            N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                            N_here=N_bl_here*N_hr_angles_here
                            reshaped_u=np.reshape(u_here,N_here,order="C")
                            reshaped_v=np.reshape(v_here,N_here,order="C")
                            gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                            widco_here=np.sqrt((1-eps_i)*(1-eps_j)*fidu_type_k*fidu_type_l)*pbw_fidu_use
                            kernel=PA_Gaussian(uubins,vvbins,[0.,0.],widco_here)
                            kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge") # no edge effects!! rigorously tested in July 2025
                            convolution_here=convolve(kernel_padded,gridded,mode="valid") # beam-smeared version of the uv-plane for this perturbation permutation
                            uvplane+=convolution_here

        # uvplane*=self.taper_grid # this tapering is to avoid ringing. power spectrum–geared tapering and per-antenna box normalization happen separately, of course
        uv_bin_edges=[uvbins,uvbins]
        return uvplane,uv_bin_edges,thetamax # this is the gridded uvplane

    def stack_to_box(self, tol:float=img_bin_tol):
        if (self.nu_ctr_MHz.value<(350/(1-self.evol_restriction_threshold/2)) or 
            self.nu_ctr_MHz>(nu_HI_z0/(1+self.evol_restriction_threshold/2))):
            raise ValueError("survey out of bounds")
        N_grid_pix=self.N_grid_pix
        taper_1d=Blackman_Harris_safe_for_FFT(N_grid_pix) # centre-origin
        taper_x,taper_y=np.meshgrid(taper_1d,taper_1d, indexing="ij")
        self.taper_grid=np.sqrt(taper_x**2+taper_y**2)

        box_uvz=np.zeros((N_grid_pix,N_grid_pix,self.N_chan),dtype="complex128")
        if not self.CST: # rescale chromatic beam widths by whatever was passed
            xy_beam_widths=np.array((self.surv_beam_widths,self.surv_beam_widths)).T
            ctr_chan_beam_width=(c/(self.nu_ctr_Hz*D))
            xy_beam_widths[:,0]*=(self.pbw_fidu[0]/ctr_chan_beam_width)
            xy_beam_widths[:,1]*=(self.pbw_fidu[1]/ctr_chan_beam_width)
            xy_beam_widths_desc=np.flip(xy_beam_widths,axis=0)

        for i in range(self.N_chan): # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/self.surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=self.surv_wavelengths[i] # update the observing frequency for next time
            nu_obs=c/self.lambda_obs
            self.nu_obs=nu_obs.decompose()

            # compute the dirty image
            if self.CST:
                chan_gridded_uvplane,chan_uv_bin_edges,thetamax=self.calc_dirty_image(Npix=N_grid_pix, tol=tol)
            else:
                xy_beam_width=enumerate(xy_beam_widths_desc)
                chan_gridded_uvplane,chan_uv_bin_edges,thetamax=self.calc_dirty_image(Npix=N_grid_pix, pbw_fidu_use=xy_beam_width, tol=tol)
            uv_bin_edges=chan_uv_bin_edges[0]

            # interpolate to store in stack
            if i==0:
                uv_bin_edges_0=chan_uv_bin_edges[0]
                theta_max_box=thetamax
                interpolated_slice=chan_gridded_uvplane
                d2u=self.d2u
            else: # chunk excision and mode interpolation in one step
                interpolator_Re=RBS(uv_bin_edges,uv_bin_edges, chan_gridded_uvplane.real)
                interpolator_Im=RBS(uv_bin_edges,uv_bin_edges, chan_gridded_uvplane.imag)
                interpolated_slice=interpolator_Re(uv_bin_edges_0,uv_bin_edges_0)+1j*interpolator_Im(uv_bin_edges_0,uv_bin_edges_0)
            box_uvz[:,:,i]=interpolated_slice*self.taper_grid
            if ((i%(self.N_chan//3))==0):
                print("{:7.1f} pct complete".format(i/self.N_chan*100))
        
        # box_xyz=fftshift(irfftn(ifftshift(box_uvz*d2u),
        #                        axes=(0,1),s=(N_grid_pix,N_grid_pix),
        #                        norm="forward")) # mixed coords before; all config space after
        box_xyz=fftshift(irfftn(ifftshift(box_uvz*d2u, axes=(0,1)),
                               axes=(0,1),s=(N_grid_pix,N_grid_pix),
                               norm="forward"), axes=(0,1)) # mixed coords before; all config space after
        for i in range(self.N_chan): # the correct generalization is per-channel normalization
            slice_i=box_xyz[:,:,i]
            norm_i=np.max(slice_i)
            if norm_i>0:
                box_xyz[:,:,i]=slice_i/norm_i # peak-normalize in configuration space
        box_xyz[box_xyz<0.]=0.
        self.box=box_xyz

        # generate a box of r-values (necessary for interpolation to survey modes in the manual beam mode of cosmo_stats as called by beam_effects)
        thetas=np.linspace(-theta_max_box,theta_max_box,N_grid_pix)
        xy_vec=self.ctr_chan_comov_dist*thetas # making the coeval approximation
        z_vec=self.comoving_distances_channels-self.ctr_chan_comov_dist 
        self.xy_vec=xy_vec
        self.z_vec=z_vec
####################################################################################################################################################################################################################################

class reconfigure_CST_beam(object):
    def __init__(self,
                 freq_lo:float=0.580*u.GHz,freq_hi:float=0.620*u.GHz, # low and high frequencies (MHz) for which to translate CST beams
                 delta_nu_CST:float=2e-5*u.GHz,                       # frequency spacing of the CST simulations to use to build up a picture of the beam
                 beam_sim_directory=None,                             # where to import CST beam files from
                 f_head:str="farfield_(f=",                           # beginning of CST beam file names
                 f_mid1:str=")_[1]",f_mid2:str=")_[2]",               # middle of CST beam file names. should include something to distinguish the two polarizations (expected but not strictly enforced... although there's no other part of the file name reading that currently anticipates differences in polarization)
                 f_tail:str="_efield.txt",                            # end of CST beam file names
                 box_outname:str="placeholder",                       # what to call the config space box of CST-informed beam values that results from a complete use of this class
                 mode:str="pathfinder",                               # which CHORD mode you're observing in: full or pathfinder (sets the sky plane scale to interpolate to)
                 Nxy:int=128,                                         # number of pixels per side of frequency slides (get one sky plane square per CST file)
                 multi_CST=True):                                     # set to False to go back to the file name construction order ok for the Jan-Apr 2026 CST
        self.beam_sim_directory=beam_sim_directory
        self.f_head=f_head
        self.f_mid1=f_mid1
        self.f_mid2=f_mid2
        self.f_tail=f_tail
        self.box_outname=box_outname
        self.multi_CST=multi_CST

        freq_hi=freq_hi.to(freq_lo.unit)
        delta_nu_CST=delta_nu_CST.to(freq_lo.unit)
        freqs=np.arange(freq_hi.value,freq_lo.value,delta_nu_CST.value)*delta_nu_CST.unit # descending; usually still in GHz
        freqs=freqs.to(u.MHz) # descending; MHz
        self.freqs=freqs
        Nfreqs=len(freqs)
        self.Nfreqs=Nfreqs
        zs_for_xis=[nu_HI_z0/freq-1 for freq in freqs] # ascending
        xis=[comoving_distance(z) for z in zs_for_xis] # ascending
        xis=Quantity(xis) # for the typical coeval approximation
        self.xis=xis
        self.CST_z_vec=xis-xis[int(Nfreqs//2)]

        if beam_sim_directory is None:
            print("Do you really mean to attempt CST imports from the working directory?")
        N_ant=64
        bmax=np.sqrt((b_NS*10)**2+(b_EW*7)**2)
        if mode=="full":
            N_ant*=8
            bmax=np.sqrt((b_NS*N_NS_full)**2+(b_EW*N_EW_full)**2)
        hemi=pi*(xis[-1]-xis[0])
        xy_for_unwrapping=np.linspace(-hemi,hemi,Nxy)
        self.xy_for_unwrapping=xy_for_unwrapping

        nu_ctr=(freq_lo+freq_hi)/2
        nu_ctr_MHz=nu_ctr.to(u.MHz)
        k_perp=kperp(nu_ctr_MHz,b_EW,bmax)
        L_xy=twopi/k_perp[0]
        xy_for_box=L_xy*fftshift(fftfreq(Nxy))
        self.xy_for_box=xy_for_box
        np.save("xy_vec_for_box"+box_outname,xy_for_box.value)
        self.Nxy=Nxy
        self.xx_grid,self.yy_grid=np.meshgrid(self.xy_for_unwrapping,self.xy_for_unwrapping, indexing="ij") # config space points of interest for the slice (guided by the transverse extent of the eventual config-space box)
        freq_names=np.zeros(Nfreqs,dtype="U6") # store the GHz CST frequencies as strings of the format that Aditya's sims use
        for i,freq in enumerate(self.freqs):
            freq_name=f"{freq:.4f}" # round to four decimal places and convert to string
            if multi_CST: # do not strip trailing zeros because of how those file names are formatted
                freq_names[i]=freq_name
            else:
                freq_names[i]=freq_name.rstrip("0") # strip trailing zeros because of how those file names are formatted
            assert freq_names[i]!="0"
        self.freq_names=freq_names

    def translate_sim_beam_slice(self,CST_filename:str,i:int=0):
        df = pd.read_table(CST_filename, skiprows=[0, 1,], sep='\s+', 
                           names=['theta', 'phi', 'AbsE', 'AbsCr', 'PhCr', 'AbsCo', 'PhCo', 'AxRat'])
        power=10**(df.AbsE.values/10) # non-log values
        theta_deg=df.theta.values*u.deg
        theta=theta_deg.to(u.rad)
        phi_deg=df.phi.values*u.deg
        phi=phi_deg.to(u.rad)
        x=self.xis[i]*theta*np.cos(phi)
        y=self.xis[i]*theta*np.sin(phi)
        sky_xy_points=np.array([x,y]).T
        return sky_xy_points,power
    
    def gen_box_from_simulated_beams(self):
        slice_grid_points=np.array([self.xx_grid,self.yy_grid]).T
        box=np.zeros((self.Nxy,self.Nxy,self.Nfreqs)) # hold interpolated beam slices
        for i in range (self.Nfreqs):
            if self.multi_CST:
                name1=self.beam_sim_directory+self.f_head+self.f_mid1+self.freq_names[i]+self.f_tail
                name2=self.beam_sim_directory+self.f_head+self.f_mid2+self.freq_names[i]+self.f_tail
            else:
                name1=self.beam_sim_directory+self.f_head+self.freq_names[i]+self.f_mid1+self.f_tail
                name2=self.beam_sim_directory+self.f_head+self.freq_names[i]+self.f_mid2+self.f_tail
            sky_xy_points,uninterp_slice_pol1=self.translate_sim_beam_slice(name1, i=i) # both polarizations will be sampled at the same (theta,phi) because they come from the same simulation = same discretization
            _,            uninterp_slice_pol2=self.translate_sim_beam_slice(name2, i=i)            

            product=uninterp_slice_pol1*uninterp_slice_pol2
            product_interpolated=gd(sky_xy_points,product,slice_grid_points,  # assumes pol1, pol2 discretized the same way... they will be, for sensibly-configured simulations
                                    method="nearest") # linear applies nans when extrap would be necessary
            power=product_interpolated/np.max(product_interpolated)
            box[:,:,i]=power

            if ((i%(self.Nfreqs//3))==0):
                print("{:7.1f} pct complete".format(i/self.Nfreqs*100))
        np.save("CST_box_"+self.box_outname,box)
        self.box=box # centre-origin

class CHORD_sense(object): # modified from a notebook helpfully shared by Debanjan Sarkar in April 2025
    def __init__(
        self,
        spacing:np.ndarray=[b_EW,b_NS], # N-S and E-W baselines (m)
        n_side:np.ndarray=[22,24],    # number of dishes per side of the array (N-S, E-W) directions
        orientation=None,             # same comment about CHORD alignment as in the per_antenna documentation (expects rad!)
        center:np.ndarray=[0,0],      # where to put the axis origin of the antenna location x-y coordinates (if you leave the default in place, it'll make the zero point the physical centre of the array)
        
        freq_cen:float = 900.*u.MHz,                  # central frequency of the observation/survey
        dish_diameter:float = 6.*u.m,                 # dish diameter
        Trcv:float = 30.*u.K,                         # receiver temperature. default = optimistic CHORD prognosis
        latitude:float = DRAO_lat,                    # latitude of the observatory (default = DRAO)
        integration_time:float= 10.*u.s,              # duration of a single integration
        time_per_day:float = 6.*u.hour,               # time spent observing per day
        n_days:int = 100 ,                            # number of days in the observation
        bandwidth:float=80.*u.MHz,                    # bandwidth of the survey/observation
        coherent:bool = False,                        # add baselines coherently if they are not instantaneously redundant?
        tsky_ref_freq:float = 400.*u.MHz,             # frequency to which the sky temp is referenced
        tsky_amplitude:float = 25.*u.K,               # sky temp
        
        horizon_buffer:float = 0.1*littleh/u.Mpc, # how many near-the-horizon modes to exclude
        foreground_model:str = "optimistic",      # foreground model for sensitivity calculations

        sv:bool=False, # extract sample variance from 21cmSense? (defaults to false because 21cmSense is ill-suited to performing these calculations for post-EoR experiments with wide fields of view [like CHORD] and I get this info for free for my CHORD forecasts from my Monte Carlo ensembles)
        tn:bool=True   # thermal noise (this is the other big contributor to the noise calculation, and my main motivation for using 21cmSense at all for CHORD forecasts)
    ):
        bl_max=bl_max.to(u.m)
        bl_min=bl_min.to(u.m)
        dish_diameter=dish_diameter.to(u.m)
        freq_cen=freq_cen.to(u.MHz)
        bandwidth=bandwidth.to(u.MHz)
        integration_time=integration_time.to(u.s)
        tsky_ref_freq=tsky_ref_freq.to(u.MHz)
        tsky_amplitude=tsky_amplitude.to(u.K)
        horizon_buffer=horizon_buffer.to(littleh/u.Mpc)
        bl_max=np.sqrt((spacing[0]*n_side[0])**2+(spacing[1]*n_side[1])**2)
        bl_min=np.min(spacing)
        self.spacing = spacing
        self.n_side = n_side
        self.orientation = orientation
        self.center = center
        self.freq_cen = freq_cen
        self.dish_diameter = dish_diameter
        self.Trcv =  Trcv
        self.latitude = latitude
        self.integration_time = integration_time
        self.time_per_day = time_per_day
        self.n_days = n_days
        n_channels = bandwidth.value/CHORD_channel_widco_MHz
        self.n_channels = n_channels
        self.bandwidth = bandwidth
        self.coherent = coherent
        self.bl_max = bl_max
        self.bl_min = bl_min
        self.tsky_ref_freq = tsky_ref_freq
        self.tsky_amplitude = tsky_amplitude
        self. horizon_buffer =  horizon_buffer
        self.foreground_model = foreground_model 
        self.sv=sv
        self.tn=tn

        ant_pos = self.rectangle_generator()
        
        observatory = Observatory(antpos=ant_pos,
                          beam = GaussianBeam(frequency=self.freq_cen,
                                              dish_diameter=self.dish_diameter),
                          Trcv = self.Trcv,   # The receiver temp will dominate over sky temp at this freq. (unlike EoR)
                          latitude = self.latitude)
        
        observation = Observation(observatory = observatory,
                          integration_time = self.integration_time, # The time in sec, telescope integrates to give one sanpshot
                          time_per_day = self.time_per_day,  # The time in hours, to observe per day (a typical choice of 8 hrs)
                          #hours_per_day = self.time_per_day,  # The time in hours, to observe per day (a typical choice of 8 hrs)
                          n_days = self.n_days,    # Total number of days of observation
                          n_channels = self.n_channels, # The number of channels
                          bandwidth = self.bandwidth,  # Bandwidth of obs
                          coherent = self.coherent, # Whether to add different baselines coherently if they are not instantaneously redundant.
                          tsky_ref_freq = self.tsky_ref_freq,
                          tsky_amplitude = self.tsky_amplitude
                          )

        sensitivity = PowerSpectrum(
            observation = observation,
            horizon_buffer = self. horizon_buffer,
            foreground_model = self.foreground_model)
        self.sensitivity=sensitivity
        
    def rectangle_generator(self): # Generate a grid of baseline locations filling a rectangular array for CHORD/HIRAX. 
        if self.spacing is not None:
            if not isinstance(self.spacing, (int, float, list, np.ndarray)):
                raise TypeError('spacing must be a scalar or list/numpy array')
            self.spacing = np.asarray(self.spacing)
            if self.spacing.size < 2:
                self.spacing = np.resize(self.spacing,(1,2))
            if np.all(np.less_equal(self.spacing,np.zeros((1,2)))):
                raise ValueError('spacing must be positive')

        if self.orientation is not None:
            if not isinstance(self.orientation, (int,float)):
                raise TypeError('orientation must be a scalar')

        if self.center is not None:
            if not isinstance(self.center, (list, np.ndarray)):
                raise TypeError('center must be a list or numpy array')
            self.center = np.asarray(self.center)
            if self.center.size != 2:
                raise ValueError('center should be a 2-element vector')
            self.center = self.center.reshape(1,-1)

        if self.n_side is None:
            raise NameError('Atleast one value of n_side must be provided')
        else:
            if not isinstance(self.n_side,  (int, float, list, np.ndarray)):
                raise TypeError('n_side must be a scalar or list/numpy array')
            self.n_side = np.asarray(self.n_side)
            if self.n_side.size < 2:
                self.n_side = np.resize(self.n_side,(1,2))
            if np.all(np.less_equal(self.n_side,np.zeros((1,2)))):
                raise ValueError('n_side must be positive')

            n_total = np.prod(self.n_side, dtype=np.uint8)
            xn,yn = self.n_side
            xs,ys=self.spacing
            n_total = xn*yn

            x = np.arange(0, xn)
            x = x - np.mean(x)
            x = x*xs

            y = np.arange(0, yn)
            y = y - np.mean(y)
            y = y*ys 
        
            z = np.zeros(n_total)
            xv, yv = np.meshgrid(x,y, indexing="ij")
            xy = np.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))

        if len(xy) != n_total:
            raise ValueError('Sizes of x- and y-locations do not agree with n_total')

        try:
            self.orientation.to(u.rad)
        except:
            self.orientation=self.orientation*u.rad
        if self.orientation is not None:   # Perform any rotation
            rot_matrix = np.asarray([[np.cos(self.orientation),-np.sin(self.orientation)], 
                                     [np.sin(self.orientation), np.cos(self.orientation)]])
            xy = np.dot(xy, rot_matrix.T)

        if self.center is not None:   # Shift the center
            xy += self.center
     
        z = np.zeros(shape=(n_total,1))
        XY = np.hstack((xy,z))

        return (np.asarray(XY)*u.m)
    
    def sense_1d(self):
        sense1d = self.sensitivity.calculate_sensitivity_1d(thermal=self.tn, sample=self.sv) #default: only thermal
        self.sense1d_k=self.sensitivity.k1d
        self.sense1d_P=sense1d

    def sense_2d(self):
        sense2d = self.sensitivity.calculate_sensitivity_2d(thermal=self.tn, sample=self.sv) # power_thermal = sensitivity.calculate_sensitivity_1d(thermal=tn, sample=sv)#only thermal
        self.sensitivity.plot_sense_2d(sense2d,plttitle="2d sense case: CHORD-like layout, default cyl k-bins",savename="CHORD_sens_default_k.png")
        kperp_keys=sorted(sense2d.keys())
        self.sense2d_kperp=np.array([k.value for k in kperp_keys]) # keys = sorted(sense2d.keys()); x = np.array([v.value for v in keys])
        self.sense2d_kpar= self.sensitivity.observation.kparallel
        self.sense2d_P=sense2d

def memo_ii_plotter(ensemble_of_spectra:np.ndarray,                       # indexed as (N_complexity_cases, N_k_perp, N_k_par)
                    ensemble_ids:np.ndarray,                              # names for each power spectrum quantity ("spectrum" for short, even though this is a misnomer in the case of ratios and residuals) in the ensemble according to the number of fiducial and perturbed beam types (N_complexity_cases,)
                    colourmap,                                            # for imshowing each power spectrum quantity
                    plot_log:bool,                                        # plot absolute or log of the power spectrum quantity?
                    k_perp:np.ndarray, k_par:np.ndarray,                  # k-perp and k-par bins that anchor each plotted spectrum
                    case_title:str, case_units:str,                       # title describing this power spectrum quantity and the corresponding units
                    save_name:str,                                        # name for the summary figure
                    norm_ext,                                             # if there is a physically motivated natural middle of the colour bar (e.g. 1 for a ratio or 0 for a residual), pass it to the plotter along with the extent of the range about this midpoint (possibly informed by the extent of the systematics you plugged into the simulation)
                    nu_ctr:float,                                         # only necessary if I insist on plotting the wedge 
                    k1_inset:float=0.06/u.Mpc, 
                    k2_inset:float=0.1/u.Mpc,
                    k3_inset:float=2.5/u.Mpc): # k-scales of interest to sample each spectrum in the ensemble
    N_spectra=len(ensemble_of_spectra)
    assert(N_spectra==len(ensemble_ids)), "mismatched number of spectra and spectrum names"
    Na=int(np.ceil(np.sqrt(N_spectra)))
    Nb=int(np.ceil(N_spectra/Na))
    if Na>Nb:
        N_LHS_rows=Nb
        N_LHS_cols=Na
    else:
        N_LHS_rows=Na
        N_LHS_cols=Nb
    k_perp=k_perp.to(1/u.Mpc)
    k_par=k_par.to(1/u.Mpc)
    cyl_extent=[k_perp[0].value,k_perp[-1].value,k_par[0].value,k_par[-1].value]
    k_perp_grid,k_par_grid=np.meshgrid(k_perp,k_par, indexing="ij")*k_par.unit
    k_mag_grid=np.sqrt(k_perp_grid**2+k_par_grid**2)
    values_of_k=np.zeros((N_spectra,3))

    fig = plt.figure(figsize=(N_LHS_cols*4, N_LHS_cols*4),layout="constrained")
    gs = gridspec.GridSpec(N_LHS_rows, N_LHS_cols+2, figure=fig)
    axs = [[fig.add_subplot(gs[row, col]) for col in range(N_LHS_cols)] for row in range(N_LHS_rows)] # grid for the left
    ax_right = fig.add_subplot(gs[:, N_LHS_cols:]) # summary holder on the right

    for k in range(N_spectra):
        i=k//N_LHS_cols
        j=k%N_LHS_cols
        spec=ensemble_of_spectra[k,:,:] # remaining indices: N complexity cases, N k-perp, N k-par
        specshape=spec.shape
        spec_to_plot=np.copy(spec)
        if isinstance(spec_to_plot, Quantity):
            spec_to_plot_de_dimensionalized=spec_to_plot.value
            ensemble_of_spectra_de_dimensionalized=ensemble_of_spectra.value
        else:
            spec_to_plot_de_dimensionalized=spec_to_plot
            ensemble_of_spectra_de_dimensionalized=ensemble_of_spectra
        if plot_log:
            spec_to_plot=np.log10(spec_to_plot_de_dimensionalized)

            off=5 # 0.5
            vminlog=np.log10(np.percentile(spec_to_plot_de_dimensionalized,off))
            if vminlog>0:
                vminlog=-0.01
            vmaxlog=np.log10(np.percentile(spec_to_plot_de_dimensionalized,100-off))
            if vmaxlog<0:
                vmaxlog=0.01
            norm=TwoSlopeNorm(0.,vmin=vminlog,
                                 vmax=vmaxlog)
        else:
            pct995=np.percentile(np.abs(ensemble_of_spectra_de_dimensionalized),99.5)
            half_middle=0.5*pct995 # fallback: put all power spectra in the ensemble on the same colour scales, informed by the extreme range
            if norm_ext is None:
                norm_ext=half_middle # branch for absolute quantities: 
            if (save_name=="cosmo_fidu_syst_fg__minus__cosmo_fidu_fg"):
                ne,vmax=norm_ext
                print("i=",i,"; norm_ext=",norm_ext)
                norm=SymLogNorm(ne,vmin=-vmax,vmax=vmax)
            else:
                ne=norm_ext
                norm=LogNorm(vmin=0.01*norm_ext,vmax=2*norm_ext)
        
        im=axs[i][j].imshow(spec_to_plot.T, cmap=colourmap, origin="lower", extent=cyl_extent, norm=norm)
        xlims_to_use=axs[i][j].get_xlim()
        ylims_to_use=axs[i][j].get_ylim()
        axs[i][j].plot(k_perp,wedge_kpar(nu_ctr,k_perp),c="tab:green")
        axs[i][j].set_xlim(xlims_to_use)
        axs[i][j].set_ylim(ylims_to_use)
        axs[i][j].set_xlabel("k$_\perp$")
        axs[i][j].set_ylabel("k$_{||}$")
        axs[i][j].tick_params(axis='x', labelrotation=30)
        axs[i][j].set_title(ensemble_ids[k])
        axs[i][j].set_aspect("equal")
        if plot_log:
            neg_ticks = np.linspace(vminlog, 0., num=4, endpoint=False)
            pos_ticks = np.linspace(0., vmaxlog, num=4, endpoint=True)
            ticks = [*neg_ticks, 0, *pos_ticks]
            plt.colorbar(im,ax=axs[i][j],shrink=0.6,extend="both", ticks=ticks)
        else: 
            plt.colorbar(im,ax=axs[i][j],shrink=0.6,extend="both")

        idx_for_k1=np.argmin(np.abs(k_mag_grid-k1_inset))
        idx_for_k1=np.unravel_index(idx_for_k1,specshape)
        idx_for_k2=np.argmin(np.abs(k_mag_grid-k2_inset))
        idx_for_k2=np.unravel_index(idx_for_k2,specshape)
        idx_for_k3=np.argmin(np.abs(k_mag_grid-k3_inset))
        idx_for_k3=np.unravel_index(idx_for_k3,specshape)
        values_of_k[k]=[ spec[idx_for_k1], spec[idx_for_k2], spec[idx_for_k3] ]
        print("\nLIM review cut:",spec[idx_for_k3])
        assert not np.any(np.asarray([idx_for_k1,idx_for_k2,idx_for_k3])>np.prod(specshape))

    complexity_indices=np.arange(N_spectra)
    ax_right.scatter(complexity_indices,values_of_k[:,0],label=str(np.round(k1_inset,4))+" (~1st BAO wiggle scale)")
    ax_right.scatter(complexity_indices,values_of_k[:,1],label=str(np.round(k2_inset,4))+" (LIM review comparison scale)")
    ax_right.scatter(complexity_indices,values_of_k[:,2],label=str(np.round(k3_inset,4))+" (~CHIME scale)")
    ax_right.set_xticks(complexity_indices, labels=ensemble_ids, rotation=40)
    if not plot_log:
        ax_right.set_ylim(0,6*ne)
    ax_right.set_xlabel("N CST types, N pointing errors")
    ax_right.set_ylabel("power spectrum quantity "+case_units)
    ax_right.set_title("insets for k closest to...")
    ax_right.legend()

    plt.suptitle("ingredients of this power spectrum quantity: "+case_title)
    plt.savefig(save_name+".png",dpi=250)
    plt.close()

def save_args_to_file(frame:str, filepath:str="settings.json"):
    args, _, _, values = inspect.getargvalues(frame)
    settings = {arg: values[arg] for arg in args}
    with open(filepath, "w") as f:
        json.dump(settings, f, indent=2, default=str)

def get_f_types_prefacs(cases):
    f_types_prefacs=[] # ends up as a ragged array in the general case, so list of lists is generally better. this is so small and quick of a calculation that I don't care about it being slow or stylistically questionable
    for case in cases:
        Nft,_=case
        if Nft==1:
            term= [1.]
        else:
            term= np.linspace(0.95,1.05,Nft)
        f_types_prefacs.append(term)
    return f_types_prefacs

def pointing_family(original_pointing,N,seed=270426):
    rng=np.random.default_rng(seed)
    orig_norm=np.linalg.norm(original_pointing)
    unscaled_pointings=rng.normal(size=(N,3))
    unscaled_norms=    np.linalg.norm(unscaled_pointings,axis=1,keepdims=True)
    rescaled_pointings=unscaled_pointings/unscaled_norms*orig_norm
    return rescaled_pointings

def power_comparison_plots(redo_window_calc:bool=False, redo_box_calc:bool=False, alr_imp_CST=False,
              mode:str="pathfinder", nu_ctr:float=800, epsxy:float=0.1,
              frac_tol_conv=0.1, N_sph=1024,categ="PA", # categ is manual/PA/CST, beam_type is either Gaussian (for PA) or manual (for CST)
              N_fidu_types=1, N_pert_types=0, 
              N_pbws_pert=0, per_channel_systematic=None,
              PA_dist="random", which_power="P",
                  
              wedge_cut=False, layer_foregrounds=True, pointing_errors=[0.,0.,0.],
                  
              freq_bin_width=0.1953125*u.MHz,

              CST_lo=None,CST_hi=None,CST_deltanu=None,
              beam_sim_directory=None,f_mid1=")_[1]",f_mid2=")_[2]",f_tail="_efield.txt",
              CST_f_head_fidu="farfield_(f=",CST_f_head_syst="farfield_(f=",
              
              from_incomplete_MC=False,
              contaminant_or_window=None, k_idx_for_window=0,
              isolated=False,seed=None,
              per_chan_syst_facs=[]):
    save_args_to_file(inspect.currentframe())

    ############################## other survey management factors ########################################################################################################################
    nu_ctr=nu_ctr.to(u.MHz)
    nu_ctr_Hz=nu_ctr.to(u.Hz)
    wl_ctr_m=c/nu_ctr_Hz
    wl_ctr_m=wl_ctr_m.decompose()

    ############################## baselines and beams ########################################################################################################################
    b_NS_CHORD=8.5*u.m
    N_NS_CHORD=24
    b_EW_CHORD=6.3*u.m
    N_EW_CHORD=22
    bminCHORD=np.min([b_NS_CHORD.value,b_EW_CHORD.value])*u.m.decompose() # force astropy to simplify 1/Hz * 1/s

    if (mode=="pathfinder"): # 10x7=70 antennas (64 w/ gaps for receiver huts and site geometry constraints), 123 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*10)**2+(b_EW_CHORD*7)**2) # pathfinder (as per the CHORD-all telecon on May 26th, but without holes)
        N_ant=64
    elif mode=="full": # 24x22=528 antennas (512 w/ receiver hut gaps), 1010 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2)
        N_ant=512
    else:
        raise ValueError("unknown array mode (not pathfinder or full)")

    if categ=="PA":
        print("PA mode only supports a Gaussian beam. Use PA-CST mode for nonuniform distributions of more realistic beams")
    hpbw_x= dif_lim_prefac*wl_ctr_m/D # rad; lambda/D estimate
    hpbw_y= 0.75*hpbw_x # simulations show this is characteristic of the UWB feeds

    ############################## pipeline administration ########################################################################################################################
    if contaminant_or_window is not None:
        c_or_w="wind"
    else:
        c_or_w="cont"
    per_chan_syst_string="none"
    per_chan_syst_name=""
    PA_dist_string="rand"
    if PA_dist=="corner":
        PA_dist_string="corn"
    elif PA_dist=="column":
        PA_dist_string="rwcl"
    elif PA_dist=="frame":
        PA_dist_string="frme"
    elif PA_dist!="random":
        raise ValueError("unknown PA_dist")

    if categ=="PA":
        if per_channel_systematic=="early_transit_measurement_like":
            per_chan_syst_string="D3AL"
        elif per_channel_systematic=="sporadic":
            per_chan_syst_string="spor"
            for fac in per_chan_syst_facs:
                per_chan_syst_name=per_chan_syst_name+str(fac)+"_"
        elif per_channel_systematic is not None:
            raise ValueError("unknown per_channel_systematic")
    
        if type(N_fidu_types)==int:
            N_fidu_types=[N_fidu_types]
            N_pert_types=[N_pert_types]
        complexity_cases=[[a,b] for b in N_pert_types for a in N_fidu_types]
        f_types_prefacs=get_f_types_prefacs(complexity_cases)
    else: # PA-CST
        if type(CST_f_head_syst)==str: # make even the single-CST-type case iterable
            CST_f_head_syst=[CST_f_head_syst]
        assert (type(CST_f_head_syst)==np.ndarray or type(CST_f_head_syst)==list) and type(CST_f_head_syst[0])==str
        N_CST_types=len(CST_f_head_syst)
        if (len(pointing_errors)==3 and type(pointing_errors[0])==int):
            pointing_errors=[pointing_errors] # associate one pointing error with one CST case
        if N_CST_types>1: # length-M list of length-N_m lists of length-3 lists
            N_max_pointing_errors_each_CST=[len(pter_per_CST) for pter_per_CST in pointing_errors]
        else: # pointing_errors is a list of three-element lists
            N_max_pointing_errors_each_CST=[len(pointing_errors)]
        N_pointing_errors_each_CST=[np.arange(0,N_max_pt_er+1) for N_max_pt_er in N_max_pointing_errors_each_CST]
        complexity_cases=[]
        for a in range(1,N_CST_types+1):
            for b in range(N_max_pointing_errors_each_CST[a-1]+1):
                point=N_pointing_errors_each_CST[a-1][b]
                complexity_cases.append([a,point])

    complexity_ids=[str(case) for case in complexity_cases]

    power_quantities_all=[]
    Delta2_quantities_all=[]
    for i,complexity_type in enumerate(complexity_cases):
        print("\n\n\nabout to handle complexity case",complexity_type)
        t00=time.time()
        if N_CST_types==0:
            N_fidu_types_i,N_pert_types_i=complexity_type
            if N_pert_types_i==0: # loop over complexity cases–friendly number of antennas with perturbed beams
                N_pbws_pert_i=0
            else:
                N_pbws_pert_i=N_pbws_pert
            f_types_prefacs_i=f_types_prefacs[i]
            complexity_part="Nreal_"+str(N_fidu_types_i)+"__"\
                            "Npert_"+str(N_pert_types_i)+"_"+str(N_pbws_pert)+"__"\
                            "epsxy_"+str(epsxy)+"__"
            related_to_N_of_types={"N_fidu_types":N_fidu_types_i,"N_pert_types":N_pert_types_i}
        else:
            NCST_i,Npoint_i=complexity_type
            related_to_N_of_types={} # this info comes from unpacking in this mode now
            complexity_id_i=str(complexity_type)
            complexity_part="N_CST_types_"+str(NCST_i)+"__"+"N_ptg_err_"+str(Npoint_i)
            if Npoint_i>0:
                pointing_errors_i=pointing_errors[NCST_i-1][:Npoint_i] # the non-+1 version is actually fine because the indices are zero-based
            else:
                pointing_errors_i=[[0.,0.,0.,]]

            CST_f_head_syst_i=CST_f_head_syst[:NCST_i]
            N_pbws_pert_i=N_pbws_pert
        
        ioname=mode+"_"+c_or_w+"_"+categ+"_"\
           ""+per_chan_syst_string+"_"+per_chan_syst_name+"_"\
           ""+str(int(nu_ctr.value))+"MHz__"+complexity_part+"_"\
           "dist_"+PA_dist_string+"__"\
           "layer_"+str(layer_foregrounds)+"__"\
           "wedge_"+str(wedge_cut)+"__"\
           "seed_"+str(seed)
        
        if (categ=="PA"):
            if (N_fidu_types_i!=4 and PA_dist=="corner"):
                continue
        elif (categ=="PA-CST"):
            if (complexity_type!=4 and PA_dist=="corner"):
                continue

        # PIPELINE ADMIN FOR THIS PA SYSTEMATIC PERMUTATION
        bundled_non_primary_aux=np.array([hpbw_x,hpbw_y])
        pbunc=epsxy
        if categ=="PA":
            windowed_survey=beam_effects(# SCIENCE
                                            # the observation
                                            bminCHORD,bmaxCHORD,                                          
                                            nu_ctr,freq_bin_width,                                         
                                            evol_restriction_threshold=def_evol_restriction_threshold,     
                                                
                                            # beam generalities
                                            primary_beam_categ=categ,primary_beam_type="Gaussian",   
                                            primary_beam_aux=bundled_non_primary_aux,
                                            primary_beam_unc=pbunc,               
                                            primary_beam_modes=None,                                 

                                            # additional considerations for per-antenna systematics
                                            N_pbws_pert=N_pbws_pert_i,
                                            PA_fidu_types_prefactors=f_types_prefacs_i,ioname=ioname,PA_distribution=PA_dist,mode=mode,
                                            per_channel_systematic=per_channel_systematic,per_chan_syst_facs=per_chan_syst_facs,

                                            **related_to_N_of_types,

                                            # additional^2 for per-antenna CST
                                            pointing_errors=pointing_errors_i, # ok whatever this is also useful for per-antenna Gaussian beams but that's not the ultimately interesting case so I'm putting it here instead

                                            # FORECASTING
                                            P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,
                                            wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds,

                                            # NUMERICAL 
                                            n_sph_modes=N_sph,                                        
                                            init_and_box_tol=0.05,CAMB_tol=0.05,                              
                                            frac_tol_conv=frac_tol_conv,seed=seed,                                         
                                            ftol_deriv=1e-16,maxiter=5,                                       
                                            PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,      
                                            radial_taper=True,image_taper=None,

                                            # CONVENIENCE
                                            heavy_beam_recalc=redo_box_calc                                                   
                                            
                                            )
        elif categ=="CST" or categ=="PA-CST":
            if categ=="CST":
                N_pbws_pert_i=N_ant
                PAdist="random"
                pointing_errors_to_use_i=pointing_errors
                PBT="Gaussian"
            else:
                PAdist=PA_dist
                pointing_errors_to_use_i=pointing_errors_i
                PBT="manual"
                if Npoint_i==0:
                    if NCST_i==1:
                        CST_f_head_syst_i=[CST_f_head_fidu] # literally just use fiducial for both num and denom everywhere
                    else:
                        pointing_errors_to_use_i=[[0.,0.,0.,]]                # use the syst beams, but don't apply any pointing errors
            windowed_survey=beam_effects(# SCIENCE
                                        # the observation
                                        bminCHORD,bmaxCHORD,                                                       
                                        nu_ctr,freq_bin_width,                                                 
                                        evol_restriction_threshold=def_evol_restriction_threshold,           
                                            
                                        # beam generalities
                                        primary_beam_categ=categ,primary_beam_type=PBT,           
                                        primary_beam_unc=pbunc,                      
                                        primary_beam_modes=None,                              

                                        # numerical beam perturbation parameters
                                        N_pbws_pert=N_pbws_pert_i,
                                        PA_fidu_types_prefactors=[1.],
                                        PA_distribution=PAdist,mode=mode,

                                        **related_to_N_of_types,

                                        # additional considerations for CST
                                        CST_lo=CST_lo,CST_hi=CST_hi,CST_deltanu=CST_deltanu,ioname=ioname,
                                        beam_sim_directory=beam_sim_directory,f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,
                                        CST_f_head_fidu=CST_f_head_fidu,CST_f_head_syst=CST_f_head_syst_i,

                                        # additional^2 for per-antenna CST
                                        pointing_errors=pointing_errors_to_use_i, # ok whatever this is also useful for per-antenna Gaussian beams but that's not the ultimately interesting case so I'm putting it here instead

                                        # FORECASTING
                                        P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,
                                        wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds,

                                        # NUMERICAL 
                                        n_sph_modes=N_sph,                                        
                                        init_and_box_tol=0.05,CAMB_tol=0.05,                                 
                                        frac_tol_conv=frac_tol_conv,seed=seed,                                         
                                        ftol_deriv=1e-16,maxiter=5,           
                                        radial_taper=True,image_taper=False,

                                        # CONVENIENCE
                                        heavy_beam_recalc=redo_box_calc, already_imported_CST=alr_imp_CST                                                  
                                        
                                        )
        else:
            raise ValueError("unknown systematics category (categ)")
        
        recalc_co_fi_xx_fg=False
        recalc_co_fi_sy_fg=False
        recalc_xx_fi_sy_fg=False
        recalc_xx_fi_xx_fg=False
        recalc_co_fi_xx_xx=False
        recalc_co_fi_sy_xx=False
        recalc_co_xx_xx_fg=False
        recalc_xx_fi_xx_xx=False
        recalc_xx_fi_sy_xx=False
        if isolated==False:
            recalc_co_fi_xx_fg=True
            recalc_co_fi_sy_fg=True
            recalc_xx_fi_sy_fg=True
            recalc_xx_fi_xx_fg=True
            recalc_co_fi_xx_xx=True
            recalc_co_fi_sy_xx=True
            recalc_co_xx_xx_fg=True
            recalc_xx_fi_xx_xx=True
            recalc_xx_fi_sy_xx=True
        
        elif isolated=="co_fi_xx_fg":
            recalc_co_fi_xx_fg=True
        elif isolated=="co_fi_sy_fg":
            recalc_co_fi_sy_fg=True
        elif isolated=="xx_fi_sy_fg":
            recalc_xx_fi_sy_fg=True
        elif isolated=="xx_fi_xx_fg":
            recalc_xx_fi_xx_fg=True
        elif isolated=="co_fi_xx_xx":
            recalc_co_fi_xx_xx=True
        elif isolated=="co_fi_sy_xx":
            recalc_co_fi_sy_xx=True
        elif isolated=="co_xx_xx_fg":
            recalc_co_xx_xx_fg=True
        elif isolated=="xx_fi_xx_xx":
            recalc_xx_fi_xx_xx=True
        elif isolated=="xx_fi_sy_xx":
            recalc_xx_fi_sy_xx=True

        print("about to perform or load Monte Carlos")
        P_unit=u.mK**2 *windowed_survey.Deltabox_xy.unit**3
        if not from_incomplete_MC:
            if redo_window_calc:
                t0=time.time()
                windowed_survey.calc_power_contamination(isolated=isolated) # loops over complexity
                P_co_xx_xx_xx=windowed_survey.P_co_xx_xx_xx
                np.save("P_co_xx_xx_xx_"+ioname+".npy",P_co_xx_xx_xx.value)
                P_xx_xx_xx_fg=windowed_survey.P_xx_xx_xx_fg
                np.save("P_xx_xx_xx_fg_"+ioname+".npy",P_xx_xx_xx_fg.value)
                t1=time.time()
                print("Pcont calculation time was",t1-t0)

                if recalc_co_fi_xx_fg:
                    P_co_fi_xx_fg=windowed_survey.P_co_fi_xx_fg
                    np.save("P_co_fi_xx_fg_"+ioname+".npy",P_co_fi_xx_fg.value)
                if recalc_co_fi_sy_fg:
                    P_co_fi_sy_fg=windowed_survey.P_co_fi_sy_fg
                    np.save("P_co_fi_sy_fg_"+ioname+".npy",P_co_fi_sy_fg.value)
                if recalc_xx_fi_sy_fg:
                    P_xx_fi_sy_fg=windowed_survey.P_xx_fi_sy_fg
                    np.save("P_xx_fi_sy_fg_"+ioname+".npy",P_xx_fi_sy_fg.value)
                if recalc_xx_fi_xx_fg:
                    P_xx_fi_xx_fg=windowed_survey.P_xx_fi_xx_fg
                    np.save("P_xx_fi_xx_fg_"+ioname+".npy",P_xx_fi_xx_fg.value)
                if recalc_co_fi_xx_xx:
                    P_co_fi_xx_xx=windowed_survey.P_co_fi_xx_xx
                    np.save("P_co_fi_xx_xx_"+ioname+".npy",P_co_fi_xx_xx.value)
                if recalc_co_fi_sy_xx:
                    P_co_fi_sy_xx=windowed_survey.P_co_fi_sy_xx
                    np.save("P_co_fi_sy_xx_"+ioname+".npy",P_co_fi_sy_xx.value)
                if recalc_co_xx_xx_fg:
                    P_co_xx_xx_fg=windowed_survey.P_co_xx_xx_fg
                    np.save("P_co_xx_xx_fg_"+ioname+".npy",P_co_xx_xx_fg.value)
                if recalc_xx_fi_xx_xx:
                    P_xx_fi_xx_xx=windowed_survey.P_xx_fi_xx_xx
                    np.save("P_xx_fi_xx_xx_"+ioname+".npy",P_xx_fi_xx_xx.value)
                if recalc_xx_fi_sy_xx:
                    P_xx_fi_sy_xx=windowed_survey.P_xx_fi_sy_xx
                    np.save("P_xx_fi_sy_xx_"+ioname+".npy",P_xx_fi_sy_xx.value)
                TEST=True
                if TEST:
                    P_CO_XX_XX_XX=windowed_survey.P_CO_XX_XX_XX
                    np.save("P_CO_XX_XX_XX__"+ioname+".npy",P_CO_XX_XX_XX.value)

                N_per_realization=windowed_survey.N_per_realization
                np.save("N_per_realization_"+ioname+".npy",N_per_realization)
                kperp_internal=windowed_survey.kperpbins_internal[:-1]
                kpar_internal=windowed_survey.kparbins_internal[:-1]
                np.save("kpar_internal_"+ioname+".npy",kpar_internal.value)
                np.save("kperp_internal_"+ioname+".npy",kperp_internal.value)
                if isolated is not False: # break early if you just calculate one windowed power spectrum at a time
                    return None

            else:
                P_co_fi_xx_fg=np.load("P_co_fi_xx_fg_"+ioname+".npy")*P_unit
                P_co_fi_sy_fg=np.load("P_co_fi_sy_fg_"+ioname+".npy")*P_unit
                P_xx_fi_sy_fg=np.load("P_xx_fi_sy_fg_"+ioname+".npy")*P_unit
                P_xx_fi_xx_fg=np.load("P_xx_fi_xx_fg_"+ioname+".npy")*P_unit
                P_co_fi_xx_xx=np.load("P_co_fi_xx_xx_"+ioname+".npy")*P_unit
                P_co_fi_sy_xx=np.load("P_co_fi_sy_xx_"+ioname+".npy")*P_unit
                P_co_xx_xx_fg=np.load("P_co_xx_xx_fg_"+ioname+".npy")*P_unit
                P_xx_fi_xx_xx=np.load("P_xx_fi_xx_xx_"+ioname+".npy")*P_unit
                P_xx_fi_sy_xx=np.load("P_xx_fi_sy_xx_"+ioname+".npy")*P_unit

                P_co_xx_xx_xx=np.load("P_co_xx_xx_xx_"+ioname+".npy")*P_unit
                P_xx_xx_xx_fg=np.load("P_xx_xx_xx_fg_"+ioname+".npy")*P_unit
                P_CO_XX_XX_XX=np.load("P_CO_XX_XX_XX__"+ioname+".npy")*P_unit
                
                N_per_realization=np.load("N_per_realization_"+ioname+".npy")
                kpar_internal=np.load("kpar_internal_"+ioname+".npy")/u.Mpc # units also by construction
                kperp_internal=np.load("kperp_internal_"+ioname+".npy")/u.Mpc
        else:
            P_co_fi_xx_fg=np.load("P_co_fi_xx_fg_MC_incomplete.npy")*P_unit
            P_co_fi_sy_fg=np.load("P_co_fi_sy_fg_MC_incomplete.npy")*P_unit
            P_xx_fi_sy_fg=np.load("P_xx_fi_sy_fg_MC_incomplete.npy")*P_unit
            P_xx_fi_xx_fg=np.load("P_xx_fi_xx_fg_MC_incomplete.npy")*P_unit
            P_co_fi_xx_xx=np.load("P_co_fi_xx_xx_MC_incomplete.npy")*P_unit
            P_co_fi_sy_xx=np.load("P_co_fi_sy_xx_MC_incomplete.npy")*P_unit
            P_co_xx_xx_fg=np.load("P_co_xx_xx_fg_MC_incomplete.npy")*P_unit
            P_xx_fi_xx_xx=np.load("P_xx_fi_xx_xx_MC_incomplete.npy")*P_unit
            P_xx_fi_sy_xx=np.load("P_xx_fi_sy_xx_MC_incomplete.npy")*P_unit

            P_co_xx_xx_xx=np.load("P_co_xx_xx_xx_MC_incomplete.npy")*P_unit
            P_xx_xx_xx_fg=np.load("P_xx_xx_xx_fg_MC_incomplete.npy")*P_unit
            P_CO_XX_XX_XX=np.load("P_CO_XX_XX_XX__MC_incomplete.npy")*P_unit
            
            N_per_realization=np.load("N_per_realization_MC_incomplete.npy")
            kpar_internal=np.load("kpar_internal_"+ioname+".npy")/u.Mpc # units by construction if importing from same pipeline generation
            kperp_internal=np.load("kperp_internal_"+ioname+".npy")/u.Mpc

        Presidual= P_co_fi_sy_fg-P_co_fi_xx_fg
        Pratio=    P_xx_fi_sy_fg/P_co_xx_xx_xx
        Pisoratio= P_xx_fi_xx_fg/P_co_xx_xx_xx
        assert(Pratio.unit.physical_type=="dimensionless" and Pisoratio.unit.physical_type=="dimensionless")
        co_fg_linearity=(P_co_xx_xx_fg-P_co_xx_xx_xx-P_xx_xx_xx_fg).value
        co_fi_linearity=(P_co_fi_xx_xx-P_co_xx_xx_xx-P_xx_fi_xx_xx).value
        print("np.sum(np.isnan(co_fg_linearity)); np.mean(co_fg_linearity)=",np.sum(np.isnan(co_fg_linearity)),np.mean(co_fg_linearity))
        print("np.sum(np.isnan(co_fi_linearity)); np.mean(co_fi_linearity)=",np.sum(np.isnan(co_fi_linearity)),np.mean(co_fi_linearity))

        k_perp_grid,k_par_grid=np.meshgrid(kperp_internal,kpar_internal, indexing="ij")
        k_mag_grid=np.sqrt(k_perp_grid**2+k_par_grid**2)

        power_quantities_this_complexity=np.array([P_xx_fi_sy_fg.value,  P_co_fi_xx_fg.value, P_co_fi_sy_fg.value, Presidual.value,     Pratio,        
                                                   P_xx_xx_xx_fg.value,  Pisoratio,           P_co_xx_xx_xx.value, P_co_fi_xx_xx.value, P_co_fi_sy_xx.value, 
                                                   P_co_xx_xx_fg.value,  P_xx_fi_xx_xx.value, P_xx_fi_sy_xx.value, P_xx_fi_xx_fg.value, P_CO_XX_XX_XX.value,
                                                   co_fg_linearity,      co_fi_linearity]) # N_pspec_types x Nkperp x Nkpar
        power_quantities_all.append(power_quantities_this_complexity) # N_complexity_cases x N_pspec_types x Nkperp x Nkpar
        
        Delta2_quantities_this_complexity=[P_qty*k_mag_grid**3/(2*pi**2) for P_qty in power_quantities_this_complexity]
        Delta2_quantities_all.append(Delta2_quantities_this_complexity)
        t01=time.time()
        print("handled complexity case",complexity_id_i,"in",t01-t00,"s")

    power_quantities_all=np.asarray(power_quantities_all)
    Delta2_quantities_all=np.asarray(Delta2_quantities_all)
    N_plots=power_quantities_this_complexity.shape[0]
    abs_map=cmasher.voltage # also consider rainforest, fall, ...others
    rel_map=cmasher.prinsenvlag

    if which_power=="P":
        absolute_units="(mK$^2$ Mpc$^3$)"
    elif which_power=="Delta2":
        absolute_units="(mK$^2$)"
    relative_units="(unitless)"

    ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   
    abs_co_no_fg_indices=np.r_[7,8,9,14]
    abs_co_fg_indices=np.r_[1,2,10]
    if which_power=="P":
        abs_co_no_fg=np.percentile(power_quantities_all[:,abs_co_no_fg_indices,:,:],98) 
        abs_co_fg=np.percentile(power_quantities_all[:,abs_co_fg_indices,:,:],90)
        fgext=3*np.std(P_xx_xx_xx_fg).value
        abs_residual=[np.percentile(power_quantities_all[:,3,:,:],90),
                      np.max(np.abs(power_quantities_all[:,3,:,:]))]
    elif which_power=="Delta2":
        # abs_co_no_fg=100*np.max(Delta2_quantities_all[:,abs_co_no_fg_indices,:,:])
        abs_co_no_fg=None
        abs_co_fg=np.percentile(Delta2_quantities_all[:,abs_co_fg_indices,:,:],90) # good for whole dynamic range
        fgext=np.percentile(Delta2_quantities_all[:,5,:,:],90)
        abs_residual=[np.percentile(Delta2_quantities_all[:,3,:,:],90),
                      np.max(np.abs(Delta2_quantities_all[:,3,:,:]))]

    co_fi_sy_fg_str="cosmo + fidu beam + syst + fg"
    co_fi_xx_fg_str="cosmo + fidu beam + fg"
    plot_version_names = ["fidu beam + syst + fg",  co_fi_xx_fg_str,                     co_fi_sy_fg_str,   "("+co_fi_sy_fg_str+") - ("+co_fi_xx_fg_str+")", "log10[ (fidu beam + syst + fg) / cosmo ]", 
                          "fg",                    "log10[ (fidu beam + fg) / cosmo ]", "cosmo",            "cosmo + fidu beam",                             "cosmo + fidu beam + syst",
                          "cosmo + fg",            "fidu beam",                         "fidu beam + syst", "fidu beam + fg",                                "COSMO",
                          "cosmo–fg linearity",    "cosmo–fidu beam linearity"]
    save_names= ["fidu_syst_fg", "cosmo_fidu_fg",         "cosmo_fidu_syst_fg", "cosmo_fidu_syst_fg__minus__cosmo_fidu_fg", "fidu_syst_fg__divby__cosmo", 
                 "fg",           "fidu_fg__divby__cosmo", "cosmo",              "cosmo_fidu",                                "cosmo_fidu_syst",
                 "cosmo_fg",    "fidu",                   "fidu_syst",          "fidu_fg",                                   "COSMOCOSMO",
                 "cosmo_fg_linearity", "cosmo_fidu_linearity"]
    plot_cmaps= [abs_map, abs_map, abs_map, rel_map, rel_map, 
                 abs_map, rel_map, abs_map, abs_map, abs_map,
                 abs_map, abs_map, abs_map, abs_map, abs_map,
                 rel_map, rel_map]
    norm_exts=  [None,      abs_co_fg, abs_co_fg,    abs_residual,         None,        
                 fgext,     None,      abs_co_no_fg, abs_co_no_fg, abs_co_no_fg,
                 abs_co_fg, None,      None,         None,         abs_co_no_fg,
                 None, None]
    plot_log=   [False, False, False, False, True,
                 False, True,  False, False, False,
                 False, False, False, False, False,
                 False, False]
    plot_units=[absolute_units, absolute_units, absolute_units, absolute_units, relative_units, 
                absolute_units, relative_units, absolute_units, absolute_units, absolute_units,
                absolute_units, absolute_units, absolute_units, absolute_units, absolute_units,
                absolute_units, absolute_units]
    ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   

    print("\n\n")
    if which_power=="P":
        power_quantities_all_correct_type=power_quantities_all
    elif which_power=="Delta2":
        power_quantities_all_correct_type=Delta2_quantities_all
    for i in range(N_plots): # iterate over plot cases
        power_quantity_this_plot_case=power_quantities_all_correct_type[:,i,:,:] # [:,i,:,:] = all complexity cases, ith power spectrum quantity, all kperps, all kpars
        memo_ii_plotter(power_quantity_this_plot_case, complexity_ids, plot_cmaps[i], plot_log[i],
                        kperp_internal, kpar_internal, 
                        plot_version_names[i], plot_units[i], save_names[i], norm_exts[i], nu_ctr)
        print("plotted ",plot_version_names[i])