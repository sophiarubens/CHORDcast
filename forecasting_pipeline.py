import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import CenteredNorm

from scipy.fft import fftshift,ifftshift,fftfreq, fftn,ifftn, irfftn, set_workers
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata as gd
from scipy.signal import convolve
from scipy.signal.windows import kaiser

import camb
from camb import model

from astropy.cosmology import Planck18
from astropy.cosmology.units import littleh
from astropy import constants as const
from astropy import units as u
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum

import cmasher
import pandas as pd
import pygtc
import time
import inspect
import json

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

scale=1e-9
dpar_default=1e-3*np.ones(len(pars_fidu))
dpar_default[3]*=scale

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
scale=1e-9
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # intentionally preposterous number of sig figs from Mathematica
eps=1e-15
per_antenna_beta=14
cosmo_stats_beta_par=14 # the starting point recommended in the documentation and, after some quick tests, more suitable than beta=2, 6, or 20. have not re-verified this since before adding foregrounds, per-antenna CST, and maybe even uniform-across-array CST
cosmo_stats_beta_perp=14
dpi_to_use=250

# CHORD
N_NS_full=24
N_EW_full=22
b_NS=8.5*u.m
b_EW=6.3*u.m
b_max_CHORD=np.sqrt((N_NS_full*b_NS)**2+(N_EW_full*b_EW)**2)*u.m
DRAO_lat=49.320791*pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6.*u.m
CHORD_channel_width_MHz=0.1953125*u.MHz
def_observing_dec=pi/60.
def_offset=1.75*pi/180. # for this placeholder state where I build up the CHORD layout using rotation matrices instead of actual measurements. probably add Hans' mask at some point to punch the corners and receiver hut holes out...
def_pbw_pert_frac=1e-2
def_evol_restriction_threshold=1./15.
img_bin_tol=5 # ringing is remarkably insensitive to turning this down; you get really bad scale mismatch by turning it up... the real solution was the "need good resolution in both Fourier and configuration space" thing
def_PA_N_grid_pix=256 # can turn this down from 512 since it doesn't change the deltaxy and a lower number of pixels per side means eval will be faster
N_fid_beam_types=1
integration_s=10*u.s # seconds
hrs_per_night=8*u.hr # borrowed from Debanjan / 21cmSense
N_nights=100 # also borrowed from Debanjan / 21cmSense
# def_N_timesteps=hrs_per_night*3600//integration_s
def_N_timesteps=1

# side calculations
def get_padding(n): # avoid edge effects in a convolution
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi
def synthesized_beam_crossing_time(nu,bmax,dec=30.*u.deg): # to accumulate rotation synthesis
    synthesized_beam_width_rad=dif_lim_prefac*(c/nu)/bmax
    beam_width_deg=synthesized_beam_width_rad*180/pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec*pi/180.)
    return crossing_time_hrs
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None
def comoving_dist_arg(z,Omegam=Omegam,OmegaLambda=OmegaLambda): # this is 1/ E(z)
    return 1/np.sqrt(Omegam*(1+z)**3+OmegaLambda)

def comoving_distance(z=0.5,H0=H0,Omegam=Omegam,OmegaLambda=OmegaLambda):
    integral,_=quad(comoving_dist_arg,0,z,args=(Omegam,OmegaLambda,))
    return (c.value*integral)/(H0.value*1000)*u.Mpc

# typical trivial conversions
def freq2z(nu_rest,nu_obs):
    assert(nu_rest.unit==nu_obs.unit)
    return nu_rest.value/nu_obs.value-1.
def z2freq(nu_rest=600.*u.MHz,z=nu_HI_z0/(600*u.MHz)-1.):
    return nu_rest/(z+1)

# Fourier space
def kpar(nu_ctr=600*u.MHz,chan_width=0.1953125*u.MHz,N_chan=300,H0=H0):
    """
    not "pure theory" kparallel values
    (relies on line-of-sight details of your survey)
    """
    prefac=1e3*twopi*H0.value*nu_HI_z0.value/c.value # 1e3 to account for units of H0/c ... assumes nu_HI_z0 and chan_width have the same units
    z_ctr=freq2z(nu_HI_z0,nu_ctr)
    Ez=1/comoving_dist_arg(z_ctr)
    zterm=Ez/((1+z_ctr)**2*chan_width.value)
    kparmax=prefac*zterm
    kparmin=kparmax/N_chan
    Delta_kpar=kparmin
    kpar_bins=np.arange(kparmin,kparmax+Delta_kpar,Delta_kpar)/u.Mpc
    return kpar_bins # evaluating at the z of the central freq of the survey (trusting slow variation...)
def kperp(nu_ctr=600.*u.MHz,bmin=6.*u.m,bmax=500.*u.m):
    """
    not "pure theory" kperp values
    (relies on sky plane details of your survey)
    """
    Dc=comoving_distance(freq2z(nu_HI_z0,nu_ctr)) # evaluating at the z of the central freq of the survey (rely on slow variation = not worth reevaluating at each freq, as usual)
    prefac=twopi*nu_HI_z0.value*1e6/(c.value*Dc.value)
    kperpmin=prefac*bmin.value
    kperpmax=prefac*bmax.value
    Delta_kperp=kperpmin
    kperp_bins=np.arange(kperpmin,kperpmax+Delta_kperp,Delta_kperp)/u.Mpc
    return kperp_bins
def wedge_kpar(nu_ctr,kperp,H0=H0,nu_rest=nu_HI_z0):
    """
    for some kperps of interest, which kparallels will the interferometer smear the wedge up to?
    """
    assert(nu_rest.unit==u.MHz)
    z=freq2z(nu_rest,nu_ctr)
    E=1/comoving_dist_arg(z)
    Dc=comoving_distance(z)
    prefac=(H0*Dc*E)/(c*(1+z))
    return prefac.value*kperp*1e3/u.Mpc # factor of 1e3 to reconcile the m-km mismatch (c in m/s but H0 in km/s/Mpc)

# beams
def PA_Gaussian(u,v,ctr,fwhm):
    u0,v0=ctr
    fwhmx,fwhmy=fwhm
    evaled=np.exp(-pi**2*((u-u0)**2*fwhmx**2+(v-v0)**2*fwhmy**2)/ln2) # prefactor ((pi*ln2)/(fwhmx*fwhmy)) will be overwritten during normalization anyway
    return evaled

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
                 delta_nu:float=CHORD_channel_width_MHz,                          # channel width
                 evol_restriction_threshold:float=def_evol_restriction_threshold, # how close to coeval is close enough? \Delta z/z
                 
                 # beam generalities
                 primary_beam_categ:str="PA",                # per-antenna Gaussian or uniform-across-array CST primary beams? >>>>>> COMING SOON: PER-ANTENNA CST <<<<<
                 primary_beam_type:str="Gaussian",           # beam type. right now an artifact of older eval modes, but will be useful in the future to specify Gaussian or per-antenna CST beams
                 primary_beam_aux:np.ndarray=None,           # PA mode: beam FWHMs for both polarization cuts; CST mode: evaluated beam boxes (in configuration space)
                 primary_beam_unc:float=None,                # uncertainty in the primary beam width
                 manual_primary_beam_modes:np.ndarray=None,  # config space pts at which a pre–discretely sampled primary beam is known

                 # additional considerations for per-antenna systematics
                 PA_N_pert_types:int=0,                         # number of types of perturbed beam
                 PA_N_pbws_pert:int=0,                          # number of beams to perturb
                 PA_N_fidu_types:int=N_fid_beam_types,          # number of types of fiducial beam
                 PA_fidu_types_prefactors=None,                 # multiplicative prefactors by which the different types of fiducial beam widths differ from those of the the diffraction-limited fiducial value
                 PA_ioname:str="placeholder",                   # unique identifier for saving files and figures related to the uv coverage of this scenario
                 PA_distribution:str="random",                  # random, column, or corner distribution of fiducial beam types?
                 mode:str="full",                               # full or pathfinder CHORD?
                 per_channel_systematic=None,                   # which chunks of the survey band are afflicted by wrong beam widths?
                 per_chan_syst_facs:np.ndarray=[1.05,0.9,1.25], # multiplicative prefracs by which chunks of survey band have the wrong beam width

                 # additional considerations for CST beams
                 CST_lo=None,CST_hi=None,               # low and high frequencies of the CST simulation band (MHz)
                 CST_deltanu=None,                      # frequency spacing of CST simulations (MHz)
                 beam_sim_directory=None,               # directory to import CST simulations from 
                 f_mid1:str=")_[1]",f_mid2:str=")_[2]", # middle part of CST file names... should include something distinguish the two polarizations (not enforced)
                 f_tail:str="_efield.txt",              # trailing part of CST file names 
                 CST_f_head_fidu:str="farfield_(f=",CST_f_head_real:str="farfield_(f=",CST_f_head_thgt:str="farfield_(f=",  # start of CST file names for different beam types (see Memo I for terminology description)

                 # FORECASTING
                 pars_set_cosmo:np.ndarray=pars_fidu,          # cosmo params to condition CAMB calls
                 pars_forecast:np.ndarray=pars_fidu,           # cosmo params of interest for a forecast
                 pars_forecast_names:np.ndarray=parnames_fidu, # >>>>> coming soon: support for derived parameters <<<<<
                 P_fid_for_cont_pwr=None,                      # fiducial power spectrum to use in Monte Carlo... typical choice for forecasting is CAMB (enforced default); some analyses may favour, for example, a flat spectrum
                 k_idx_for_window:int=0,                       # examine contaminant power or window functions?
                 interp_to_survey_modes:bool=False,            # don't bother turning down the k-space resolution to literal instrument-accessible modes
                 wedge_cut:bool=False,                         # excise info from voxels inside the foreground wedge?
                 layer_foregrounds:bool=False,                 # add synchrotron foregrounds on top of cosmo + beam data?
                 pointing_error:np.ndarray=[0.,0.,0.],         # subject the real and thgt beams to a pointing error

                 # NUMERICAL 
                 n_sph_modes:int=256,                          # how many points in the theory power spectrum?
                 dpar=None,                                    # initial guess for numerical partial derivative step size
                 init_and_box_tol:float=0.05,                  # how much wider to make the config space extent of the brightness temp boxes compared to the survey box (numerical insurance factor...)
                 CAMB_tol:float=0.05,                          # same thing but for the CAMB call (if you make a sensible choice here, you will never have to extrapolate the theory spectrum to get info about a part of k-space you're interested in)
                 Nkpar_box=None, Nkperp_box=None,              # number of power spectrum bins along each cylindrical axis
                 frac_tol_conv:float=0.1,                      # fraction (not percent) convergence for Monte Carlo ensemble -> used to determine the number of necessary realizations
                 seed=None,                                    # specify a seed if you want replicable RNG behaviour
                 ftol_deriv:float=1e-16,                       # this numerical tolerance factor * the function you are trying to differentiate gives a pointwise comparison for whether the derivative computation is accurate enough with the current step size
                 maxiter:int=5,                                # maximum number of times the partial derivative computation can recurse with an updated step size estimate
                 PA_N_grid_pix:int=def_PA_N_grid_pix,          # number of pixels per side of gridded uv plane
                 PA_img_bin_tol:int=img_bin_tol,               # how tightly to zoom into the gridded uv plane. there's a tradeoff here for a given number of voxels per side: if you zoom in really closely, you get to resolve more small differences in uv coordinates, but you'll probably incur some prominent ringing at the edges when you try to IFT a fairly sharp feature that extends to the edge of the box. If you don't zoom in much, the observed ringing won't be as stark, but there will be more wasted pixels / more of the uv coverage will be shuffled into central bins
                 radial_taper=None,image_taper=None,           # apply apodization along the line of sight or transverse directions?

                 # CONVENIENCE
                 heavy_beam_recalc:bool=True                   # save time by not repeating per-antenna calculations?
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
        self.layer_foregrounds=layer_foregrounds
        assert(nu_ctr.unit==u.MHz)
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
        self.surv_channels=np.arange(self.nu_lo.value,self.nu_hi.value,self.Deltanu.value)
        self.r0=comoving_distance(self.z_ctr)
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
        kparmin_surv=kpar_surv[0]
        kparmax_surv=kpar_surv[-1]
        self.kpar_surv=kpar_surv
        self.kparmin_surv=kparmin_surv
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        self.bmax=bmax
        kperp_surv=kperp(self.nu_ctr,self.bmin,self.bmax)
        kperpmin_surv=kperp_surv[0]
        kperpmax_surv=kperp_surv[-1]
        self.kperp_surv=kperp_surv
        self.kperpmin_surv=kperpmin_surv
        self.Nkperp_surv=len(self.kperp_surv)

        self.kmin_surv=np.sqrt(kperpmin_surv**2+kparmin_surv**2)
        self.kmax_surv=np.sqrt(kperpmax_surv**2+kparmax_surv**2)

        self.Lsurv_box_xy=twopi/kperpmin_surv
        self.Nvox_box_xy=int(self.Lsurv_box_xy*kperpmax_surv/pi)
        self.Lsurv_box_z=twopi/kparmin_surv
        self.Nvox_box_z=int(self.Lsurv_box_z*kparmax_surv/pi)

        if layer_foregrounds:
            synchrotron_factors= 300*(np.linspace(self.nu_lo.value,self.nu_hi.value,self.Nvox_box_z)/150)**-2.5 # # cf. eq. 11 of Pober et al. 2012 for the normalization
            rng = np.random.default_rng()
            white_noise_box=rng.normal(size=(self.Nvox_box_xy,self.Nvox_box_xy,self.Nvox_box_z)) # loc=0.,scale=1.,
            fg_xy=np.linspace(-self.Lsurv_box_xy/2,self.Lsurv_box_xy/2,self.Nvox_box_xy)
            fg_z= np.linspace(-self.Lsurv_box_z/2, self.Lsurv_box_z/2, self.Nvox_box_z)
            self.foreground_field=white_noise_box*synchrotron_factors[None,None,:]*u.mK
            self.fg_modes=[fg_xy,fg_xy,fg_z]


        # primary beam considerations
        self.primary_beam_categ=primary_beam_categ
        self.fwhm_x,self.fwhm_y=primary_beam_aux
        self.primary_beam_unc= primary_beam_unc

        if (primary_beam_categ.lower()=="pa"):
            if (primary_beam_categ.lower()=="pa"):
                self.per_chan_syst_facs=per_chan_syst_facs

                self.PA_N_pert_types=          PA_N_pert_types
                self.PA_N_pbws_pert=           PA_N_pbws_pert
                if (self.PA_N_pbws_pert>N_ant):
                    print("WARNING: as called, more antennas would be perturbed than present in this array configuration")
                    print("resetting with merely all antennas perturbed...")
                    PA_N_pbws_pert=N_ant
                    self.PA_N_pbws_pert=PA_N_pbws_pert
                self.PA_N_timesteps=           def_N_timesteps
                self.PA_N_grid_pix=            PA_N_grid_pix
                self.img_bin_tol=              PA_img_bin_tol
                self.PA_distribution=          PA_distribution
                self.PA_N_fidu_types= PA_N_fidu_types
                self.PA_fidu_types_prefactors= PA_fidu_types_prefactors
                fwhm=primary_beam_aux 

                fidu=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                pbw_pert_frac=0.,
                                N_timesteps=self.PA_N_timesteps,
                                N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                N_fiducial_beam_types=1,
                                outname=PA_ioname)
                real=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                 pbw_pert_frac=0.,
                                 N_timesteps=self.PA_N_timesteps,
                                 N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                 distribution=self.PA_distribution,
                                 N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                 outname=PA_ioname,
                                 per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
                thgt=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=self.PA_N_pert_types,
                                 pbw_pert_frac=self.primary_beam_unc,
                                 N_timesteps=self.PA_N_timesteps,
                                 N_pbws_pert=PA_N_pbws_pert,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                 distribution=self.PA_distribution,
                                 N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                 outname=PA_ioname,
                                 per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
                per_chan_syst_name=thgt.per_chan_syst_name
                self.per_chan_syst_name=per_chan_syst_name
                self.surv_channels_MHz_from_PA=thgt.surv_channels_MHz
                self.surv_beam_widths_from_PA=thgt.surv_beam_widths

                if heavy_beam_recalc:
                    fidu.stack_to_box()
                    print("constructed fiducially-beamed box")
                    fidu_box=fidu.box

                    real.stack_to_box()
                    print("constructed real-beamed box")
                    real_box=real.box
                    xy_vec=real.xy_vec
                    z_vec=real.z_vec
                    
                    thgt.stack_to_box()
                    print("constructed perturbed-beamed box")
                    thgt_box=thgt.box

                    np.save("fidu_box_"+PA_ioname+".npy",fidu_box)
                    np.save("real_box_"+PA_ioname+".npy",real_box)
                    np.save("thgt_box_"+PA_ioname+".npy",thgt_box)
                    np.save("xy_vec_"+  PA_ioname+".npy",xy_vec.value)
                    np.save("z_vec_"+   PA_ioname+".npy",z_vec.value)
                else:
                    fidu_box=np.load("fidu_box_"+PA_ioname+".npy")
                    real_box=np.load("real_box_"+PA_ioname+".npy")
                    thgt_box=np.load("thgt_box_"+PA_ioname+".npy")
                    xy_vec=  np.load("xy_vec_"+  PA_ioname+".npy")*u.Mpc
                    z_vec=   np.load("z_vec_"+   PA_ioname+".npy")*u.Mpc

                primary_beam_aux=[fidu_box,real_box,thgt_box]
                manual_primary_beam_modes=(xy_vec.value,xy_vec.value,z_vec.value) # might need to re-unit-ify this more robustly later, but for now the main use is interpolation and I don't want to jam up scipy by putting units where they have no business being
            
            # now do the manual-y things
            if (manual_primary_beam_modes is None):
                raise ValueError("not enough info")
            else:
                self.manual_primary_beam_modes=manual_primary_beam_modes
            try:
                self.manual_primary_fidu,self.manual_primary_real,self.manual_primary_thgt=primary_beam_aux # assumed to be sampled at the same config space points
            except: # primary beam samplings not unpackable the way they need to be
                raise ValueError("not enough info")
        elif (primary_beam_categ.lower()=="cst"):
            precalculated_xy_vec=self.Lsurv_box_xy*fftshift(fftfreq(self.Nvox_box_xy))
            if heavy_beam_recalc:
                fidu=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                          beam_sim_directory=beam_sim_directory,f_head=CST_f_head_fidu,
                                          f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="fidu_box_"+PA_ioname)
                fidu.gen_box_from_simulated_beams()
                fidu_box=fidu.box
                CST_z_vec=np.asarray(fidu.CST_z_vec)
                real=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                          beam_sim_directory=beam_sim_directory,f_head=CST_f_head_real,
                                          f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="real_box_"+PA_ioname)
                real.gen_box_from_simulated_beams()
                real_box=real.box
                thgt=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=self.Nvox_box_xy,
                                          beam_sim_directory=beam_sim_directory,f_head=CST_f_head_thgt,
                                          f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="thgt_box_"+PA_ioname)
                thgt.gen_box_from_simulated_beams()
                thgt_box=thgt.box

                np.save("fidu_box_"+PA_ioname+".npy",fidu_box)
                np.save("real_box_"+PA_ioname+".npy",real_box)
                np.save("thgt_box_"+PA_ioname+".npy",thgt_box)
                np.save("z_vec"+PA_ioname+".npy",CST_z_vec.value)
            else:
                fidu_box=np.load("fidu_box_"+PA_ioname+".npy")
                real_box=np.load("real_box_"+PA_ioname+".npy")
                thgt_box=np.load("thgt_box_"+PA_ioname+".npy")
                CST_z_vec=np.load("z_vec"+PA_ioname+".npy")*u.Mpc
            
            manual_primary_beam_modes=(precalculated_xy_vec.value,precalculated_xy_vec.value,CST_z_vec.value)

            if self.PA_CST:
                ///
                # now I need as many pointing errors as beam types.
                # for now, this is how I build a library of CST beam types
                # I need to add more flexibility to what I pass to beam_effects from power_comparison_plots
            else:
                if self.pointing_error!=[0.,0.,0.]: # mathematically nothing wrong with applying a 0º-in-every-direction rotation, but it's a waste of compute. definitely still wasting compute here constructing the same rotation matrix twice, but I'll sort that out later. 
                    real_box=repoint_beam(manual_primary_beam_modes,real_box,pointing_error)
                    thgt_box=repoint_beam(manual_primary_beam_modes,thgt_box,pointing_error)
                primary_beam_aux=[fidu_box,real_box,thgt_box]

            if (manual_primary_beam_modes is None):
                raise ValueError("not enough info")
            else:
                self.manual_primary_beam_modes=manual_primary_beam_modes
            try:
                self.manual_primary_fidu,self.manual_primary_real,self.manual_primary_thgt=primary_beam_aux # assumed to be sampled at the same config space points
            except: # primary beam samplings not unpackable the way they need to be
                raise ValueError("not enough info")

        else:
            raise ValueError("unknown primary_beam_categ") # as far as primary power beam perturbations go, they can all pretty much be described as being applied PA, or in some externally-implemented custom way

        self.primary_beam_type=primary_beam_type
        self.primary_beam_aux=primary_beam_aux
        self.primary_beam_unc=primary_beam_unc

        # groundwork-informed forecasting considerations
        if not self.CST:
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
        ksph,self.Ptruesph=self.get_mps(self.pars_set_cosmo,kmin_CAMB,kmax_CAMB)
        self.ksph=ksph/u.Mpc
        self.Deltabox_xy=self.Lsurv_box_xy/self.Nvox_box_xy
        self.Deltabox_z= self.Lsurv_box_z/ self.Nvox_box_z
        self.radial_taper=radial_taper
        self.image_taper=image_taper
        self.interp_to_surv_modes=interp_to_survey_modes

        # precision control for numerical derivatives
        self.ftol_deriv=ftol_deriv
        self.eps=eps
        self.maxiter=maxiter

        # considerations for power spectrum binning directly from the box
        minbin=25
        maxbin=400
        div=3
        if Nkpar_box is None:
            self.Nkpar_box=np.min([np.max([self.Nvox_box_z//div,minbin]),maxbin])
        else:
            self.Nkpar_box=Nkpar_box
        if Nkperp_box is None:
            self.Nkperp_box=np.min([np.max([self.Nvox_box_xy//div,minbin]),maxbin])
        else:
            self.Nkperp_box=Nkperp_box

        self.frac_tol_conv=frac_tol_conv
        
        # considerations for printing the calculated bias results
        self.pars_forecast_names=pars_forecast_names
        assert (len(pars_forecast)==len(pars_forecast_names))

        # placeholders for forecasting-relevant matrices
        self.del_P_del_pars=np.zeros((self.N_pars_forecast,self.Nkpar_surv,self.Nkperp_surv))
        self.F=None
        self.B=None

    def get_mps(self,pars_use:np.ndarray,minkh:float=1e-4/u.Mpc,maxkh:float=1./u.Mpc): # get matter power spec from CAMB
        z=[self.z_ctr]
        H0=pars_use[0]
        h=H0/100.
        ombh2=pars_use[1]
        omch2=pars_use[2]
        As=pars_use[3]*scale
        ns=pars_use[4]

        pars_use_internal=camb.set_params(H0=H0.value, ombh2=ombh2.value, omch2=omch2.value, ns=ns, mnu=0.06,omk=0)
        pars_use_internal.InitPower.set_params(As=As,ns=ns,r=0)
        assert(maxkh.unit==1/u.Mpc and minkh.unit==1/u.Mpc)
        pars_use_internal.set_matter_power(redshifts=z, kmax=maxkh.value*h.value)
        results = camb.get_results(pars_use_internal)
        pars_use_internal.NonLinear = model.NonLinear_none
        kh,z,pk=results.get_matter_power_spectrum(minkh=minkh.value,maxkh=maxkh.value,npoints=self.n_sph_modes)
        kh/=u.Mpc
        pk*=u.mK**2*u.Mpc**3

        return kh,pk
    
    def unbin_to_Pcyl(self,pars_to_use:np.ndarray,kperp_to_use:np.ndarray=None,kpar_to_use:np.ndarray=None): # interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        if kperp_to_use is None:
            kperp_to_use=self.kperp_surv
        if kpar_to_use is None:
            kpar_to_use=self.kpar_surv
        k,Psph_use=self.get_mps(pars_to_use,minkh=self.kmin_surv,maxkh=self.kmax_surv)
        k=k/u.Mpc
        CAMBlength=Psph_use.shape[1]
        k=k.reshape((CAMBlength,))
        Psph_use=Psph_use.reshape((CAMBlength,))
        k_unique, unique_idx = np.unique(k, return_index=True)
        Psph_use = Psph_use[unique_idx]
        k = k_unique

        self.Psph=Psph_use
        kperp_grid,kpar_grid=np.meshgrid(kperp_to_use,kpar_to_use,indexing="ij")
        kmag_grid=np.sqrt(kpar_grid**2+kperp_grid**2)
        Nkperp_use=len(kperp_to_use)
        Nkpar_use=len(kpar_to_use)
        Nk=Nkperp_use*Nkpar_use
        kmag_grid_flat=np.reshape(kmag_grid,(Nk,))
        sort_array=np.argsort(kmag_grid_flat)
        kmag_grid_flat_sorted=kmag_grid_flat[sort_array]

        Pcyl=np.zeros(Nk)
        interpolator=RGI((k.value,),Psph_use,
                         bounds_error=False,fill_value=None)
        Pcyl[sort_array]=interpolator(kmag_grid_flat_sorted[:, None])
        Pcyl=np.reshape(Pcyl,(Nkperp_use,Nkpar_use))

        return kpar_grid,kperp_grid,Pcyl

    def calc_power_contamination(self, isolated:bool=False): # Monte Carlo numerical windowing of beam-aware brightness temp boxes to yield several cylindrically power spectra of interest for forecasting and diagnostics. various states of beam knowledge and fiducial spectrum as appropriate (see Memos I-II)
        if self.P_fid_for_cont_pwr is None:
            P_fid=np.reshape(self.Ptruesph,(self.n_sph_modes))
        elif self.P_fid_for_cont_pwr=="window": # make the fiducial power spectrum a numerical top hat
            P_fid=np.zeros(self.n_sph_modes)
            P_fid[self.k_idx_for_window]=1.
            P_fid*=u.mK**2*u.Mpc**3
        else:
            raise ValueError("unknown P_fid_for_cont_pwr")

        if (self.primary_beam_categ=="PA" or self.primary_beam_categ=="CST"):
            fi=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           P_fid=P_fid,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                           primary_beam_num=self.manual_primary_fidu,primary_beam_type_num="manual",
                           primary_beam_den=self.manual_primary_fidu,primary_beam_type_den="manual",
                           Nkperp=self.Nkperp_box,Nkpar=self.Nkpar_box,
                           frac_tol=self.frac_tol_conv,seed=self.seed,
                           kperpbins_interp=self.kperp_surv,kparbins_interp=self.kpar_surv,
                           k_fid=self.ksph,
                           manual_primary_beam_modes=self.manual_primary_beam_modes, no_monopole=True,
                           radial_taper=self.radial_taper,image_taper=self.image_taper,
                           wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,layer_foregrounds=self.layer_foregrounds,foreground_field=self.foreground_field,fg_modes=self.fg_modes)
            self.kperpbins_internal=fi.kperpbins
            self.kparbins_internal=fi.kparbins
            rt=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           P_fid=P_fid,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                           primary_beam_num=self.manual_primary_real,primary_beam_type_num="manual",
                           primary_beam_den=self.manual_primary_thgt,primary_beam_type_den="manual",
                           Nkperp=self.Nkperp_box,Nkpar=self.Nkpar_box,
                           frac_tol=self.frac_tol_conv,seed=self.seed,
                           kperpbins_interp=self.kperp_surv,kparbins_interp=self.kpar_surv,
                           k_fid=self.ksph,
                           manual_primary_beam_modes=self.manual_primary_beam_modes, no_monopole=True,
                           radial_taper=self.radial_taper,image_taper=self.image_taper,
                           wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,layer_foregrounds=self.layer_foregrounds,foreground_field=self.foreground_field,fg_modes=self.fg_modes)
            sf=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           P_fid=np.ones(self.n_sph_modes)*u.mK**2*u.Mpc**3,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                           primary_beam_num=self.manual_primary_real,primary_beam_type_num="manual",
                           primary_beam_den=self.manual_primary_thgt,primary_beam_type_den="manual",
                           Nkperp=self.Nkperp_box,Nkpar=self.Nkpar_box,
                           frac_tol=self.frac_tol_conv,seed=self.seed,
                           kperpbins_interp=self.kperp_surv,kparbins_interp=self.kpar_surv,
                           k_fid=self.ksph,
                           manual_primary_beam_modes=self.manual_primary_beam_modes, no_monopole=True,
                           radial_taper=self.radial_taper,image_taper=self.image_taper,
                           wedge_cut=self.wedge_cut,nu_ctr_for_wedge=self.nu_ctr,layer_foregrounds=self.layer_foregrounds,foreground_field=self.foreground_field,fg_modes=self.fg_modes)
        
        else:
            raise ValueError("unknown primary_beam_categ") 
        
        recalc_fi=False
        recalc_rt=False
        recalc_sf=False
        if isolated==False:     # recalculate all three MC-windowed power spectra [see i, ii, iii below]
            recalc_fi=True
            recalc_rt=True
            recalc_sf=True
        if isolated=="realthgt": # recalculate only the theory + fidu beam + syst + meas errs + ?fg? power spec [i]
            recalc_rt=True
        if isolated=="fidufidu": # recalculate only the theory + fidu beam + ?fg? power spec [ii]
            recalc_fi=True
        if isolated=="contam":   # recalculate only the above two power spectra
            recalc_fi=True
            recalc_rt=True
        if isolated=="flatrlth": # recalculate only the fidu beam + syst + meas errs + ?fg? power spec [iii]
            recalc_sf=True

        if recalc_fi:
            fi.power_Monte_Carlo(interfix="fi")
            self.N_per_realization=fi.N_per_realization
            self.Pfiducial_cyl=fi.P_binned_converged
            self.kperp_for_theory=fi.kperpbins
            self.kpar_for_theory=fi.kparbins
            print("theory + fidu beam +                    ?fg? MC complete")
        if recalc_rt:
            rt.power_Monte_Carlo(interfix="rt")
            if not recalc_fi:
                self.N_per_realization=rt.N_per_realization
                self.kperp_for_theory=rt.kperpbins
                self.kpar_for_theory=rt.kparbins
            self.Prealthought_cyl=rt.P_binned_converged
            print("theory + fidu beam + syst + meas errs + ?fg? MC complete")
        if (recalc_sf):
            sf.power_Monte_Carlo(interfix="sf")
            if not recalc_fi:
                self.N_per_realization=sf.N_per_realization
                self.kperp_for_theory=sf.kperpbins
                self.kpar_for_theory=sf.kparbins
            self.Pnotheory_cyl=sf.P_binned_converged
            print("         fidu beam + syst + meas errs + ?fg? MC complete")

        _,_,self.Ptheory_cyl=self.unbin_to_Pcyl(self.pars_set_cosmo, kperp_to_use=self.kperp_for_theory, kpar_to_use=self.kpar_for_theory)# unbin_to_Pcyl(self,pars_to_use,kperp_to_use=None,kpar_to_use=None)
        if isolated==False:
            self.Pcont_cyl=self.Pfiducial_cyl-self.Prealthought_cyl

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
        self.sample_variance=np.sqrt(2/self.N_per_realization)*self.Pfiducial_cyl # rescale according to the number of realizations 

        sen=CHORD_sense(spacing=[self.b_EW,self.b_NS], n_side=[self.N_EW,self.N_NS], orientation=def_offset, center=None, dish_diameter=D, # array layout
                        freq_cen=self.nu_ctr, integration_time=integration_s*u.s, time_per_day=hrs_per_night, n_days=100, bandwidth=self.bw, # obs config
                        Trcv=35*u.K, latitude=DRAO_lat*u.radian, tsky_ref_freq=400.*u.MHz, tsky_amplitude=25*u.K, # what's going on with the sky?
                        coherent=False, horizon_buffer=0.1*littleh/u.Mpc, foreground_model="optimistic") # processing details
        sen.sense2d()
        kperp_from_21cmSense=sen.sense2d_kperp
        kpar_from_21cmSense=sen.sense2d_kpar
        thnoise_21cmSense=sen.sense2d_P
        kperp_surv_grid,kpar_surv_grid=np.meshgrid(self.kperp_surv,self.kpar_surv,
                                                   indexing="ij")
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
        samples=np.random.multivariate_normal(np.zeros(self.N_pars_forecast),C,size=N_Fisher_samples)
        pygtc.plotGTC(chains=samples, 
                      paramNames=self.pars_forecast_names,
                      truths=self.pars_forecast,
                      plot_name="forecast_corner_plot.png")

    def print_survey_characteristics(self):
        print("survey properties.......................................................................")
        print("........................................................................................")
        print("survey centred at.......................................................................\n    nu ={:>7.4} \n    z  = {:>9.4} \n    Dc = {:>9.4f} \n".format(self.nu_ctr,self.z_ctr,self.r0))
        print("survey spans............................................................................\n    nu =  {:>5.4}    -  {:>5.4}     (deltanu = {:>6.4} ) \n    z =  {:>9.4} - {:>9.4}     (deltaz  = {:>9.4}    ) \n    Dc = {:>9.4f} - {:>9.4f} (deltaDc = {:>9.4f})\n".format(self.nu_lo,self.nu_hi,self.bw,self.z_hi,self.z_lo,self.z_hi-self.z_lo,self.Dc_hi,self.Dc_lo,self.Dc_hi-self.Dc_lo))
        if (self.primary_beam_type.lower()!="manual"):
            print("characteristic instrument response widths...............................................\n    beamFWHM0 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_x,self.primary_beam_unc))
            print("specific to the cylindrically asymmetric beam...........................................\n    beamFWHM1 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_y,self.primary_beam_unc))
        print("cylindrically binned wavenumbers of the survey..........................................\n    kperp     {:>8.4} - {:>8.4} ({:>4} bins of width {:>8.4} \n    kparallel {:>8.4} - {:>8.4} ({:>4} channels of width {:>7.4}  ) \n".format(self.kperpmin_surv,self.kperp_surv[-1],self.Nkperp_surv,self.kperp_surv[-1]-self.kperp_surv[-2],    self.kparmin_surv,self.kpar_surv[-1],self.Nkpar_surv,self.kpar_surv[-1]-self.kpar_surv[-2]))

    def print_results(self):
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(self.pars_forecast):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(self.pars_forecast_names[p], par, self.biases[p], self.biases[p]/par))
        return None
####################################################################################################################################################################################################################################

def repoint_beam(domain,beam,rot_angles=[0.,0.,0.,]):
    rot_x,rot_y,rot_z=rot_angles
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
    x_grid,y_grid,z_grid=np.meshgrid(xvec,yvec,zvec,indexing="ij")
    x_flat=np.reshape(x_grid,(N,))
    y_flat=np.reshape(y_grid,(N,))
    z_flat=np.reshape(z_grid,(N,))
    xyz_flat=np.asarray([x_flat,y_flat,z_flat]).T # 3xN

    # philosophy here: need 3xN for R@ compatibility, but can't just use R@xyz_flat because RGI needs something with shape ((Nx,),(Ny,),(Nz,)), not (3,N)
    x_prime_vec,_,_=R@[x_grid[:,0,0],y_grid[:,0,0],z_grid[:,0,0]] # this is probably going to take some reslicing, re-transposing, and reassembling
    _,y_prime_vec,_=R@[x_grid[0,:,0],y_grid[0,:,0],z_grid[0,:,0]]
    _,_,z_prime_vec=R@[x_grid[0,0,:],y_grid[0,0,:],z_grid[0,0,:]]

    interpolator=RGI((x_prime_vec,y_prime_vec,z_prime_vec),beam,
                     bounds_error=False,fill_value=None)
    rotated_beam_sampled_at_original_domain=interpolator(xyz_flat)

    return rotated_beam_sampled_at_original_domain

"""
this class helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. Monte Carlo effective windowing of a power spectrum by a primary beam
4. interpolate a power spectrum (sph, cyl, or sph->grid)
"""

class cosmo_stats(object):
    def __init__(self,
                 Lxy:float=600.*u.MHz,Lz:float=None,                                         # physical box length (Mpc). one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
                 T_pristine:np.ndarray=None,T_primary:np.ndarray=None,                       # brightness temperature box realizations without ("_pristine") or with ("_primary") the primary beam multiplied in
                 P_fid:np.ndarray=None,                                                      # power spectrum you want to window. probably comes from theory (like CAMB) or is flat (for a reference calculation)
                 k_fid:np.ndarray=None,                                                      # Fourier space points where the fiducial power spectrum is sampled
                 Nvox:int=None,Nvoxz:int=None,                                               # number of voxels in the x/y or z directions
                 primary_beam_num:np.ndarray=None,     primary_beam_den:np.ndarray=None,     # numerator/denominator (of power spectrum estimator) version of the primary beam (box of values evaluated in config space)
                 primary_beam_aux_num:np.ndarray=None, primary_beam_aux_den:np.ndarray=None, # numerator/denominator version of helpful quantities that go along with the primary beam (characteristic widths for a per-antenna Gaussian beam; x/y and z vectors for a CST beam)
                 primary_beam_type_num:str="Gaussian", primary_beam_type_den:str="Gaussian", # USED TO BE Airy/Gaussian for achromatic uniform-across-array beams. CURRENTLY can only be Gaussian, but SOON will be generalized to admit per-antenna CST beams
                 Nkperp:int=10,Nkpar:int=0,                                                  # number of k-bins in the sky plane and line of sight directions
                 binning_mode:str="lin",                                                     # bin linearly or logarithmically
                 bin_each_realization:bool=False,                                            # bin each realization of the Monte Carlo? (with the current implementation there's no typical use case where this would be necessary, but the option is there)
                 frac_tol:float=0.1,                                                         # fractional tolerance in cosmic variance of the Monte Carlo ensemble -> used to calculate the number of realizations
                 kperpbins_interp:np.ndarray=None,kparbins_interp:np.ndarray=None,           # bins where you want to know about the power spectrum (if you're interested in interpolating to some binning scheme other than what you get from chopping up the box)
                 P_converged:np.ndarray=None,                                                # converged Monte Carlo power spectrum
                 kind:str="cubic",avoid_extrapolation:bool=False,                            # conditioning choices for interpolation: degree of interpolation; whether or not to avoid extrapolation
                 no_monopole:bool=True,seed=None,                                            # Monte Carlo realization logistics: whether or not to subtract the monopole moment when you generate boxes (the option is mostly there if you're interested in off-label uses of this code to compute power spectra from fields that are not cosmological overdensity fields); RNG seed for predictable ensemble behaviour
                 manual_primary_beam_modes:np.ndarray=None,                                  # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 radial_taper=None,image_taper=None,                                         # apodize along the sky plane or line-of-sight directions to suppress ringing originating from features that cut off sharply?
                 wedge_cut:bool=False,nu_ctr_for_wedge=None,                                 # throw away info from k-modes inside the foreground wedge?; when using synchrotron foregrounds AND performing a wedge cut, the calling routine should specify the central frequency of the survey in question to have a physical anchor for the foregrounds
                 layer_foregrounds:bool=False,foreground_field=None,fg_modes=None):          # layer synchrotron foregrounds on top of brightness temp realizations?; fg fields and modes computed by a calling routine
        
        # spectrum and box
        if (Lz is None): # cubic box
            self.Lz=Lxy
            self.Lxy=Lxy
        else:            # rectangular prism box (useful for e.g. dirty image stacking)
            self.Lz=Lz
            self.Lxy=Lxy
        physical_volume=self.Lxy**2*self.Lz
        self.physical_volume=physical_volume
        self.P_fid=P_fid
        self.T_primary=T_primary
        self.T_pristine=T_pristine
        self.no_monopole=no_monopole
        if ((T_primary is None) and (T_pristine is None) and (P_fid is None)): # require either a box or a fiducial power spec (il faut some way of determining #voxels/side; passing just Nvox is not good enough)
            raise ValueError("not enough info")
        else:                                                                  # there is possibly enough info to proceed, but still need to check for conflicts and gaps
            if ((T_pristine is not None) and (T_primary is not None)):
                print("WARNING: T_pristine and T_primary both passed; T_primary will be temporarily ignored and then internally overwritten to ensure consistency with primary_beam")
                if (T_pristine.shape!=T_primary.shape):
                    raise ValueError("conflicting info")
                else:                                                          # use box shape to set cubic/ rectangular prism box attributes
                    self.Nvox,_,self.Nvoxz=T_primary.shape
            if ((Nvox is not None) and (T_pristine is not None)):              # possible conflict: if both Nvox and a box are passed, 
                T_pristine_shape0,_,T_pristine_shape2=T_pristine.shape
                if (Nvox!=T_pristine.shape[0]):                                # but Nvox and the box shape disagree,
                    raise ValueError("conflicting info")                       # estamos en problemas
                else:
                    self.Nvox= T_pristine_shape0                               # otherwise, initialize the Nvox attributes
                    self.Nvoxz=T_pristine_shape2
            elif (Nvox is not None):                                           # if Nvox was passed but T was not, use Nvox to initialize the Nvox attributes
                self.Nvox=Nvox 
                if (Nvoxz is None):                                            # if no Nvoxz was provided, make the box cubic
                    Nvoxz=Nvox
                self.Nvoxz=Nvoxz
            else:                                                              # remaining case: T was passed but Nvox was not, so use the shape of T to initialize the Nvox attributes
                self.Nvox= T_pristine_shape0
                self.Nvoxz=T_pristine_shape2
            if ((T_primary is not None) and (T_pristine is None)):             # passing T_primary but not T_pristine is not handled anywhere up to this point
                self.Nvox,_,self.Nvoxz=T_primary.shape

            if (P_fid is not None): # no hi fa res si the fiducial power spectrum has a different dimensionality or bin width than the realizations you plan to generate (boxes will be generated from a grid-interpolated P_fid, anyway)
                Pfidshape=P_fid.shape
                Pfiddims=len(Pfidshape)
                if (Pfiddims==2):
                    if primary_beam_num is None: # trying to do a minimalistic instantiation where I merely provide a fiducial power spectrum and interpolate it
                        self.fid_Nkperp,self.fid_Nkpar=Pfidshape
                        if primary_beam_num is not None: 
                            raise ValueError("conflicting info") # numerator primary beam needs to be the fiducial one; doesn't make sense to claim you have a perturbed but not fiducial pb
                    else:
                        try: # see if the power spec is a CAMB-esque (1,npts) array
                            self.P_fid=np.reshape(P_fid,(Pfidshape[-1],)) # make the CAMB MPS shape amenable to the calcs internal to this class
                        except: # barring that...
                            pass # treat the power spectrum as being truly cylindrically binned
                elif (Pfiddims==1):
                    self.fid_Nkperp=Pfidshape[0] # already checked that P_fid is 1d, so no info is lost by extracting the int in this one-element tuple, and fid_Nkperp being an integer makes things work the way they should down the line
                    self.fid_Nkpar=0
                else:
                    raise ValueError("unsupported binning mode")
        
        # config space
        self.Deltaxy=self.Lxy/self.Nvox                           # sky plane: voxel side length
        self.xy_vec_for_box=self.Lxy*fftshift(fftfreq(self.Nvox)) # sky plane Cartesian config space coordinate axis
        self.Deltaz= self.Lz/self.Nvoxz                           # line of sight voxel side length
        self.z_vec_for_box= self.Lz*fftshift(fftfreq(self.Nvoxz)) # line of sight Cartesian config space coordinate axis
        self.d3r=self.Deltaz*self.Deltaxy**2                      # volume element = voxel volume

        self.xx_grid,self.yy_grid,self.zz_grid=np.meshgrid(self.xy_vec_for_box,
                                                           self.xy_vec_for_box,
                                                           self.z_vec_for_box,
                                                           indexing="ij")      # box-shaped Cartesian coords
        self.r_grid=np.sqrt(self.xx_grid**2+self.yy_grid**2+self.zz_grid**2)   # r magnitudes at each voxel

        # Fourier space
        self.Deltakxy=twopi/self.Lxy                                        # voxel side length
        self.Deltakz= twopi/self.Lz
        self.d3k=self.Deltakxy**2*self.Deltakz                              # volume element / voxel volume
        self.kxy_vec_for_box_corner=twopi*fftfreq(self.Nvox,d=self.Deltaxy) # one Cartesian coordinate axis - non-fftshifted/ corner origin
        self.kz_vec_for_box_corner= twopi*fftfreq(self.Nvoxz,d=self.Deltaz)
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.kxy_vec_for_box_corner,
                                                                                self.kxy_vec_for_box_corner,
                                                                                self.kz_vec_for_box_corner,
                                                                                indexing="ij")               # box-shaped Cartesian coords
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the box generation direction)
        self.kmag_grid_centre=fftshift(self.kmag_grid_corner) 
        self.kmag_grid_corner_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**2*self.Nvoxz,))

        # foreground groundwork
        self.wedge_cut=wedge_cut
        if wedge_cut:
            assert(nu_ctr_for_wedge is not None), "an arbitrary box <-> power spectrum translation doesn't require frequency\n"+\
                                                  "info. But, when you opt into the wedge cut, you must override the None\n"+\
                                                  "default in the nu_ctr_for_wedge keyword."
            self.kperp_corner=np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2)
            wedge_kpar_threshold_corner=wedge_kpar(nu_ctr_for_wedge,self.kperp_corner)
            self.voxels_in_wedge_corner=self.kz_grid_corner<=wedge_kpar_threshold_corner
        self.layer_foregrounds=layer_foregrounds
        if layer_foregrounds:
            assert foreground_field is not None
            assert fg_modes is not None
            self.fg_modes=fg_modes
            self.foreground_field=RGI(fg_modes,foreground_field,
                                      bounds_error=False,fill_value=None)(np.array([self.xx_grid.value,self.yy_grid.value,self.zz_grid.value]).T).T*u.mK # interpolate beam_effects voxelization to cosmo_stats discretization... following the same strategy as beam interpolation

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
                self.P_fid_interp_1d_to_3d()
            elif (len(self.P_fid.shape)==2):
                self.k_fid0,self.kfid1=self.k_fid # fiducial k-modes should be unpackable, since P_fid has been verified to be truly 2d
                raise ValueError("not yet implemented")
            else: # so far, I do not anticipate working with "truly three dimensional"/ unbinned power spectra
                raise ValueError("not yet implemented")
        
        # binning considerations
        self.bin_each_realization=bin_each_realization
        self.binning_mode=binning_mode
        self.Nkperp=Nkperp # the number of bins to put in power spec realizations you construct
        self.Nkpar=Nkpar
        self.kmax_box_xy= pi/self.Deltaxy
        self.kmax_box_z=  pi/self.Deltaz
        self.kmin_box_xy= twopi/self.Lxy
        self.kmin_box_z=  twopi/self.Lz
        self.kperpbins,self.limiting_spacing_0=self.calc_bins(self.Nkperp,self.Nvox,self.kmin_box_xy,self.kmax_box_xy)
        if self.limiting_spacing_0<self.Deltakxy: # trying to bin more finely than the box can tell you about (guaranteed to have >=1 empty bin)
            raise ValueError("resolution error")
        
        if (self.Nkpar>0):
            # self.kparbins,self.limiting_spacing_1=self.calc_bins(self.Nkpar,self.Nvox,self.kmin_box_xy,self.kmax_box_xy)
            self.kparbins,self.limiting_spacing_1=self.calc_bins(self.Nkpar,self.Nvoxz,self.kmin_box_z,self.kmax_box_z)
            if (self.limiting_spacing_1<self.Deltakz): # idem ^
                raise ValueError("resolution error")
            self.kperpbins_grid,self.kparbins_grid=np.meshgrid(self.kperpbins,self.kparbins,indexing="ij")
        else:
            self.kparbins=None
        
            # voxel grids for sph binning        
        self.sph_bin_indices_centre=      np.digitize(self.kmag_grid_centre,self.kperpbins,right=True)
        self.sph_bin_indices_1d_centre=   np.reshape(self.sph_bin_indices_centre, (self.Nvox**2*self.Nvoxz,))

            # voxel grids for cyl binning
        if (self.Nkpar>0):
            self.kpar_column_centre= np.abs(fftshift(self.kz_vec_for_box_corner))                                      # magnitudes of kpar for a representative column along the line of sight (z-like)
            self.kperp_slice_centre= np.sqrt(fftshift(self.kx_grid_corner)**2+fftshift(self.ky_grid_corner)**2)[:,:,0] # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
            perpbin_indices_slice_centre=    np.digitize(self.kperp_slice_centre,self.kperpbins,right=True)          # cyl kperp bin that each voxel falls into
            self.perpbin_indices_slice_centre=perpbin_indices_slice_centre
            self.perpbin_indices_slice_1d_centre= np.reshape(self.perpbin_indices_slice_centre,(self.Nvox**2,))        # 1d version of ^ (compatible with np.bincount)
            parbin_indices_column_centre=    np.digitize(self.kpar_column_centre,self.kparbins,right=True)          # cyl kpar bin that each voxel falls into
            self.parbin_indices_column_centre=parbin_indices_column_centre
        
        # tapering/apodization
        taper_xyz=1.
        if radial_taper is not None:
            taper_z=radial_taper(self.Nvoxz,cosmo_stats_beta_par)
        else:
            taper_z=1.
        if image_taper is not None:
            taper_xy=image_taper(self.Nvox,cosmo_stats_beta_perp)
        else:
            taper_xy=1.
        taper_xxx,taper_yyy,taper_zzz=np.meshgrid(taper_xy,taper_xy,taper_z,indexing="ij")
        taper_xyz=taper_xxx*taper_yyy*taper_zzz
        self.taper_xyz=taper_xyz

        # primary beam
        self.primary_beam_num=primary_beam_num
        self.primary_beam_aux_num=primary_beam_aux_num
        self.primary_beam_type_num=primary_beam_type_num
        self.manual_primary_beam_modes=manual_primary_beam_modes # _fi and _rt assumed to be sampled at the same modes, if this is the case
        if (self.primary_beam_num is not None): # non-identity FIDUCIAL primary beam
            if (self.primary_beam_type_num=="Gaussian" or self.primary_beam_type_num=="Airy"):
                self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_aux_num
                evaled_primary_num=  self.primary_beam_num(self.xx_grid,self.yy_grid,self.fwhm_x,  self.fwhm_y,  self.r0)     
            elif (self.primary_beam_type_num=="manual"):
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable
                    primary_beam_num.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ValueError("conflicting info") 
                if self.manual_primary_beam_modes is None:
                    raise ValueError("not enough info")

                x_manual_primary,y_manual_primary,z_manual_primary=manual_primary_beam_modes
                x_manual_primary*=u.Mpc
                y_manual_primary*=u.Mpc
                z_manual_primary*=u.Mpc
                x_have_lo=x_manual_primary[0]
                x_have_hi=x_manual_primary[-1]
                y_have_lo=y_manual_primary[0]
                y_have_hi=y_manual_primary[-1]
                z_have_lo=z_manual_primary[0]
                z_have_hi=z_manual_primary[-1]
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
                evaled_primary_num=RGI(manual_primary_beam_modes,self.primary_beam_num,
                                       bounds_error=False,fill_value=None)(np.array([self.xx_grid.value,self.yy_grid.value,self.zz_grid.value]).T).T
                self.evaled_primary_num=evaled_primary_num
            
            else:
                raise ValueError("not yet implemented")

        self.primary_beam_den=primary_beam_den
        self.primary_beam_aux_den=primary_beam_aux_den
        self.primary_beam_type_den=primary_beam_type_den
        self.manual_primary_beam_modes=manual_primary_beam_modes # _fi and _rt assumed to be sampled at the same modes, if this is the case
        if (self.primary_beam_den is not None): # non-identity PERTURBED primary beam
            if (self.primary_beam_type_den=="Gaussian" or self.primary_beam_type_den=="Airy"):
                self.fwhm_x,self.fwhm_y,r0=self.primary_beam_aux_den
                self.r0=r0*u.Mpc
                evaled_primary_den=  self.primary_beam_den(self.xx_grid,self.yy_grid,self.fwhm_x,  self.fwhm_y,  self.r0)                
            elif (self.primary_beam_type_den=="manual"):
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable... so, no danger of attribute errors
                    primary_beam_den.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ValueError("conflicting info") 
                if self.manual_primary_beam_modes is None:
                    raise ValueError("not enough info")

                evaled_primary_den=RGI(manual_primary_beam_modes,self.primary_beam_den,
                                       bounds_error=False,fill_value=None)(np.array([self.xx_grid.value,self.yy_grid.value,self.zz_grid.value]).T).T
                self.evaled_primary_den=evaled_primary_den

            else:
                evaled_primary_den=None    

            if evaled_primary_den is not None:
                evaled_primary_use_for_eff_vol=evaled_primary_den
            else:
                evaled_primary_use_for_eff_vol=evaled_primary_num

            self.effective_volume=np.sum((evaled_primary_use_for_eff_vol*self.taper_xyz)**2*self.d3r)

        else:                               # identity primary beam
            self.effective_volume=physical_volume
            self.evaled_primary_num=1.
        if (self.T_pristine is not None):
            self.T_primary=self.T_pristine*self.evaled_primary_num # APPLY THE FIDUCIAL BEAM
        
        # strictness control for realization averaging
        self.frac_tol=frac_tol
        self.N_realizations=int(np.round(self.frac_tol**-2))

        # P_converged interpolation bins
        self.kperpbins_interp=kperpbins_interp
        self.kparbins_interp=kparbins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        self.P_unbinned_running_sum=np.zeros((self.Nvox,self.Nvox,self.Nvoxz))*u.mK**2*u.Mpc**3
        if (P_converged is not None):          # maybe you have a converged power spec average from a previous calc and just want to interpolate or generate more box realizations?
            self.P_converged=P_converged
        else:
            self.P_converged=None
        self.P_interp=None
        self.not_converged=None
        self.N_cumul=np.zeros((self.Nkperp,self.Nkpar))

    def calc_bins(self,Nki:int,Nvox_to_use:int,kmin_to_use:float,kmax_to_use:float):
        """
        generate a set of bins spaced according to the desired scheme with max and min
        """
        if (self.binning_mode=="log"):
            kbins=np.logspace(np.log10(kmin_to_use),np.log10(kmax_to_use),num=Nki)
            limiting_spacing=twopi*(10.**(kmax_to_use)-10.**(kmax_to_use-(np.log10(Nvox_to_use)/Nki)))/u.Mpc
        elif (self.binning_mode=="lin"):
            kbins=np.linspace(kmin_to_use,kmax_to_use,Nki)
            limiting_spacing=twopi*(0.5*Nvox_to_use-1)/(Nki)/u.Mpc # version for a kmax that is "aware that" there are +/- k-coordinates in the box
        else:
            raise ValueError("unsupported binning mode")
        return kbins,limiting_spacing # kbins            -> floors of the bins to which the power spectrum will be binned (along one axis)
                                      # limiting_spacing -> smallest spacing between adjacent bins (uniform if linear; otherwise, depends on the binning strategy)
    
    def P_fid_interp_1d_to_3d(self):
        """
        interpolate a "physics-only" (spherically symmetric) power spectrum (e.g. from CAMB) to a 3D cosmological box.
        """
        k_fid_unique,unique_idx=np.unique(self.k_fid,return_index=True)
        Pfid_unique=self.P_fid[unique_idx]
        sort_array=np.argsort(self.kmag_grid_corner_flat)
        kmag_grid_corner_flat_sorted=self.kmag_grid_corner_flat[sort_array]
        P_fid_flattened_box=np.zeros(self.Nvox**2*self.Nvoxz)
        interpolator=RGI((k_fid_unique.value,),Pfid_unique,
                          bounds_error=False,fill_value=None)
        P_fid_flattened_box[sort_array]=interpolator(kmag_grid_corner_flat_sorted.value[:,None])
        self.P_fid_box=np.reshape(P_fid_flattened_box,(self.Nvox,self.Nvox,self.Nvoxz))
            
    def generate_P(self,send_to_P_fid:bool=False,T_use=None):
        """
        philosophy: 
        * compute the power spectrum of a known cosmological box 
        * defer binning to another method
        * add to running sum of realizations
        """
        if (T_use is None or T_use=="primary"):
            T_use=self.T_primary
        else:
            T_use=self.T_pristine
        if (self.T_primary is None):    # power spec has to come from a box
            self.generate_box() # populates/overwrites self.T_pristine and self.T_primary
        assert(T_use.unit==u.mK)
        
        T_tilde=fftshift(fftn((ifftshift(T_use.value*self.taper_xyz)*self.d3r)))
        modsq_T_tilde=(T_tilde*np.conjugate(T_tilde)).real*T_use.unit**2*self.d3r.unit**2
        P_unbinned=modsq_T_tilde/self.effective_volume # box-shaped, but calculated according to the power spectrum estimator equation
        self.P_unbinned=P_unbinned
        if self.bin_each_realization:
            self.bin_power()
        
        if send_to_P_fid: # if generate_P was called speficially to have a spec from which all future box realizations will be generated
            if self.bin_each_realization:
                self.P_fid=self.P_binned
            else:
                self.P_fid=self.P_unbinned
        else:             # the "normal" case where you're just accumulating a realization (any binning happens at the end of the Monte Carlo)
            self.P_unbinned_running_sum+=P_unbinned

    def bin_power(self,power_to_bin=None):
        if power_to_bin is None:
            power_to_bin=self.unbinned_power
        if (self.Nkpar==0):   # bin to sph
            unbinned_power_1d= np.reshape(power_to_bin,    (self.Nvox**2*self.Nvoxz,))

            sum_unbinned_power= np.bincount(self.sph_bin_indices_1d_centre, 
                                           weights=unbinned_power_1d, 
                                           minlength=self.Nkperp)*u.mK**2*u.Mpc**3  # for the ensemble avg: sum    of unbinned_power values in each bin
            N_unbinned_power=   np.bincount(self.sph_bin_indices_1d_centre,
                                           minlength=self.Nkperp)       # for the ensemble avg: number of unbinned_power values in each bin
            sum_unbinned_power_truncated=sum_unbinned_power[:-1]       # excise sneaky corner modes: I devised my binning to only tell me about voxels w/ k<=(the largest sphere fully enclosed by the box), and my bin edges are floors. But, the highest floor corresponds to the point of intersection of the box and this largest sphere. To stick to my self-imposed "the stats are not good enough in the corners" philosophy, I must explicitly set aside the voxels that fall into the "catchall" uppermost bin. 
            N_unbinned_power_truncated=  N_unbinned_power[:-1]         # idem ^
            final_shape=(self.Nkperp,)
        elif (self.Nkpar!=0): # bin to cyl
            sum_unbinned_power= np.zeros((self.Nkperp+1,self.Nkpar))*u.mK**2*u.Mpc**3 # for the ensemble avg: sum    of unbinned_power values in each bin  ...upon each access, update the kparBIN row of interest, but all Nkperp columns
            N_unbinned_power=   np.zeros((self.Nkperp+1,self.Nkpar)) # for the ensemble avg: number of unbinned_power values in each bin
            for i in range(self.Nvoxz): # iterate over the kpar axis of the box to capture all LoS slices
                if (i==0): # stats for the representative "bull's eye" slice transverse to the LoS
                    slice_bin_counts=np.bincount(self.perpbin_indices_slice_1d_centre, minlength=self.Nkperp)
                unbinned_power_slice= power_to_bin[:,:,i]                    # take the slice of interest of the preprocessed box values !!kpar is z-like
                unbinned_power_slice_1d= np.reshape(unbinned_power_slice, 
                                                   (self.Nvox**2,))          # 1d for bincount compatibility
                slice_bin_sums= np.bincount(self.perpbin_indices_slice_1d_centre,
                                             weights=unbinned_power_slice_1d, 
                                             minlength=self.Nkperp)             # this slice's update to the numerator of the ensemble average
                current_par_bin=self.parbin_indices_column_centre[i]

                sum_unbinned_power[:,current_par_bin]+= slice_bin_sums  # update the numerator   of the ensemble avg
                N_unbinned_power[  :,current_par_bin]+= slice_bin_counts # update the denominator of the ensemble avg
            
            sum_unbinned_power_truncated= sum_unbinned_power[:-1,:] # excise sneaky corner modes (see the analogous operation in the sph branch for an explanation)
            N_unbinned_power_truncated=   N_unbinned_power[  :-1,:] # idem ^
            final_shape=(self.Nkperp,self.Nkpar)

        N_unbinned_power_truncated[N_unbinned_power_truncated==0]=maxint # avoid division-by-zero errors during the division the estimator demands
        self.N_modes_per_bin=N_unbinned_power_truncated
        self.N_cumul+=self.N_modes_per_bin

        avg_unbinned_power=sum_unbinned_power_truncated/N_unbinned_power_truncated # actual estimator quotient
        P_binned=np.array(avg_unbinned_power)
        P_binned.reshape(final_shape)
        self.P_binned=P_binned
    
    def generate_box(self):
        """
        generate a box representing a random realization of a known power spectrum
        """
        if (self.Nvox<self.Nkperp):
            raise ValueError("Nvox should be >= Nkperp")
        if (self.P_fid is None):
            try:
                self.generate_P(store_as_P_fid=True) # T->P_fid is deterministic, so, even if you start with a random realization, it'll be helpful to have a power spec summary stat to generate future realizations
            except: # something goes wrong in the P_fid calculation
                raise ValueError("not enough info")
        
        assert(self.P_fid_box is not None)
        sigmas=np.sqrt(self.physical_volume*self.P_fid_box/2.) # from inverting the estimator equation and turning variances into std devs
        T_tilde_Re,T_tilde_Im=self.rng.normal(loc=0.*sigmas,scale=sigmas,size=np.insert(sigmas.shape,0,2))
        
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        if self.wedge_cut:
            T_tilde[self.voxels_in_wedge_corner]=0.
        T=fftshift(irfftn(T_tilde*self.d3k.value,
                          s=(self.Nvox,self.Nvox,self.Nvoxz),
                          axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: fftshiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
        T*=u.mK
        if self.layer_foregrounds:
            T+=self.foreground_field
        if self.no_monopole:
            T-=np.mean(T) # subtract monopole moment
        
        self.T_pristine=T
        self.T_primary=T*self.evaled_primary_num

    def power_Monte_Carlo(self,interfix:str=""):
        """
        philosophy:
        * since P->box is not deterministic,
        * compute the power spectra from a bunch of generated boxes and average them together
        * realization ceiling precalculated from the Poisson noise–related fractional tolerance
        """
        assert(self.P_fid is not None), "cannot average over numerically windowed realizations without a fiducial power spec"
        self.not_converged=True
        i=0

        t0=time.time()
        for i in range(self.N_realizations):
            self.generate_box()
            self.generate_P(T_use="primary")
            ti=time.time()
            if ((ti-t0)>3600): # actually save the realizations every hour
                np.save("P_"+interfix+"_unconverged.npy",self.P_unbinned_running_sum/i)
                t0=time.time()

        P_unbinned_converged=self.P_unbinned_running_sum/self.N_realizations
        self.P_unbinned_converged=P_unbinned_converged
        self.bin_power(power_to_bin=P_unbinned_converged)
        P_binned_converged=self.P_binned

        if (self.Nkpar>0):
            self.P_binned_converged=np.reshape(P_binned_converged,(self.Nkperp,self.Nkpar))
        else:
            self.P_binned_converged=np.reshape(P_binned_converged,(self.Nkperp,))

        self.N_per_realization=self.N_cumul/self.N_realizations

    def interpolate_P(self,use_P_fid:bool=False):
        """
        typical use: interpolate a power spectrum binned sph/cyl to modes accessible by the box to modes of interest for the survey being forecast

        notes
        * default behaviour upon requesting extrapolation: 
          "ValueError: One of the requested xi is out of bounds in dimension 0"
        * if extrapolation is acceptable for your purposes:
          run with avoid_extrapolation=False
          (bounds_error supersedes fill_value, so there's no issue with 
          fill_value always being set to what it needs to be to permit 
          extrapolation [None for the nd case, "extrapolate" for the 1d case])
        """
        if use_P_fid:
            self.P_converged=self.P_fid
        else:
            if (self.P_converged is None):
                print("WARNING: P_converged DNE yet. \nAttempting to calculate it now...")
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
        P_interpolator=RGI(modes_defined_at,self.P_converged,
                           bounds_error=self.avoid_extrapolation,fill_value=None)
        P_interp=P_interpolator(modes_to_eval_at)
        if self.kparbins_interp is not None:
            P_interp=P_interp.T # anticipate the RGI behaviour
        self.P_interp=P_interp
####################################################################################################################################################################################################################################

def beam_type_distribution(N_NS,N_EW,N_types,distribution="random"):
    N_ant=N_NS*N_EW
    if N_types>0:
        if distribution=="random":
            per_antenna_types=np.random.randint(0,N_types,size=(N_ant,))
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
            per_antenna_types=np.zeros((N_NS,N_EW))
            rng=np.random.default_rng()
            per_antenna_types[1:-1,1:-1]=rng.integers(1,high=N_types,
                                                    size=(N_NS-2,N_EW-2))
        else:
            raise ValueError("beam distribution pattern not yet implemented")
        
        per_antenna_types=np.reshape(per_antenna_types,(N_ant,))
    else:
        per_antenna_types=np.zeros(N_ant)

    return per_antenna_types

"""
this class helps compute numerical windowing boxes for brightness temp boxes
resulting from primary beams that have the flexibility to differ on a per-
antenna basis. (beam chromaticity built in).
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
                 Delta_nu:float=CHORD_channel_width_MHz,                           # channel width in frequency (MHz)
                 distribution:str="random",                                        # distribution of per-antenna systematics. the options I've encoded for now are random, column, and corner, based on where the fiducial beam types are placed within the array
                 fidu_types_prefactors=None,                                       # ** multiplicative prefactors by which the different fiducial beam types differ from the physics-informed fiducial beam width for a given frequency channel
                 per_channel_systematic=None,                                      # apply a systematic that corrupts the 1/lambda scaling of the beam width? options encoded so far are sporadic (multiply the beam widths for a contiguous chunk of frequency channels by a different multiplicative prefactor for the different fiducial beam types) and D3A-like (noise + too wide at low frequencies... inspired by early three-dish transit beam measurements)
                 per_chan_syst_facs=None,                                          # the multiplicative prefactors for the sporadic per-antenna systematic (see above)
                 evol_restriction_threshold:float=def_evol_restriction_threshold,  # max \delta z/z you will tolerate for the survey of interest and still consider the box close enough to coeval
    
                 ensemble_of_CST_beams=None
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
        bmax=np.sqrt(N_NS*b_NS**2+N_EW*b_EW**2)
        N_ant=N_NS*N_EW
        N_bl=N_ant*(N_ant-1)//2
        self.nu_ctr_MHz=nu_ctr
        self.nu_ctr_Hz=nu_ctr.value*1e6*u.Hz
        self.Dc_ctr=comoving_distance(nu_HI_z0/nu_ctr-1)
        self.N_hrs=synthesized_beam_crossing_time(self.nu_ctr_Hz,bmax=bmax,dec=observing_dec) # freq needs to be in Hz
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
        offset_from_latlon_rotmat=np.array([[np.cos(offset_rad),-np.sin(offset_rad)],
                                            [np.sin(offset_rad), np.cos(offset_rad)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1)) # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T

        # array layout indexing
        
        # line-of-sight quantities
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/self.Delta_nu)
        self.N_chan=N_chan
        nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(nu_hi,nu_lo,N_chan) # decr.
        surv_channels_Hz=1e6*surv_channels_MHz.value*u.Hz
        surv_wavelengths=c/surv_channels_Hz # incr.
        self.surv_wavelengths=surv_wavelengths.decompose()
        z_channels=nu_HI_z0/surv_channels_MHz-1.
        comoving_distances_channels=np.asarray([comoving_distance(chan).value for chan in z_channels]) # incr.
        self.comoving_distances_channels=comoving_distances_channels*u.Mpc
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]
        self.surv_channels_MHz=surv_channels_MHz

        # helper args specific to Gaussian or CST calculations
        self.CST=False if ensemble_of_CST_beams is None else True
        if self.CST:
            assert(N_PA_CST_types==len(ensemble_of_CST_beams))
            self.N_PA_CST_types=N_PA_CST_types
            self.CST_ensemble=ensemble_of_CST_beams
            self.pb_types=beam_type_distribution(N_NS,N_EW,self.N_PA_CST_types, distribution=self.distribution)

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
                    random_draw=np.random.uniform(size=(N_pert_types,))
                    random_perturbations=random_draw*self.pbw_pert_frac
                    epsilons[1:]=random_perturbations
                else: 
                    epsilons[1]=self.pbw_pert_frac
                indices_of_ants_w_pert_pbws=np.random.randint(0,N_ant,size=self.N_pbws_pert) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
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
                noise_frac=(noise_bound_hi-noise_bound_lo)*np.random.random_sample(size=(N_chan,))+noise_bound_lo # random_sample draws fall within [0,1) but I want values between [0.75,1.25)*(that channel's beam width)
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
        uvw_inst=np.zeros((N_bl,3))
        indices_of_constituent_ant_pb_fidu_types=np.zeros((N_bl,2))
        indices_of_constituent_ant_pb_pert_types=np.zeros((N_bl,2))
        k=0
        for i in range(N_ant):
            for j in range(i+1,N_ant):
                uvw_inst[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
                indices_of_constituent_ant_pb_fidu_types[k]=[pbw_fidu_types[i],pbw_fidu_types[j]]
                indices_of_constituent_ant_pb_pert_types[k]=[pbw_pert_types[i],pbw_pert_types[j]]
                k+=1
        uvw_inst=np.vstack((uvw_inst,-uvw_inst))
        indices_of_constituent_ant_pb_fidu_types=np.vstack((indices_of_constituent_ant_pb_fidu_types,indices_of_constituent_ant_pb_fidu_types))
        indices_of_constituent_ant_pb_pert_types=np.vstack((indices_of_constituent_ant_pb_pert_types,indices_of_constituent_ant_pb_pert_types))
        self.uvw_inst=uvw_inst
        self.indices_of_constituent_ant_pb_fidu_types=indices_of_constituent_ant_pb_fidu_types
        self.indices_of_constituent_ant_pb_pert_types=indices_of_constituent_ant_pb_pert_types
        print("computed ungridded instantaneous uv-coverage")

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*self.N_hrs/12
        hour_angles=np.linspace(0,hour_angle_ceiling,self.N_timesteps)
        thetas=hour_angles*15*np.pi/180*u.rad
        
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
        uubins,vvbins=np.meshgrid(uvbins,uvbins,indexing="ij")
        uvplane=0.*uubins
        uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])
        pad_lo,pad_hi=get_padding(Npix)

        if self.CST:
            for i in range(self.N_PA_CST_types):
                type_i=self.pb_types[i]
                for j in range(i+1):
                    type_j=self.pb_types[j]

                    here=(self.indices_of_constituent_ant_pb_types[:,0]==i
                          )&(self.indices_of_constituent_ant_pb_types[:,1]==j) //////// # need to add this indices thingy upstream
                    u_here=self.uv_synth[here,0,:] # [N_bl,2,N_hr_angles]
                    v_here=self.uv_synth[here,1,:]
                    N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                    N_here=N_bl_here*N_hr_angles_here
                    reshaped_u=np.reshape(u_here,N_here)
                    reshaped_v=np.reshape(v_here,N_here)
                    gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                    LoS_idx=np.argmin(np.abs(self.nu_obs-self.CST_freqs)) /////////// # need to add CST freqs upstream
                    image_i=self.CST_ensemble[type_i,:,:,LoS_idx] # [N_beam_types, Nxy, Nxy, Nz]
                    image_j=self.CST_ensemble[type_j,:,:,LoS_idx]
                    image_ij=np.sqrt(image_i*image_j) # geo mean of the beams of this baseline's two constituent antennas. still on initial CST grid
                    uv_ij=fftshift(fftn(ifftshift(image_ij*self.dxdy))) /////// # need to add dxdy for CST # FT to put in uv space 
                    interpolator=RBS(uvbins_CST,uvbins_CST, uv_ij)
                    kernel=interpolator(uvbins_use,uvbins_use) # interpolate from corresponding-to-CST-coordinate-change domain to the gridded uv bins of this slice

                    """
                    nterpolated_slice=RBS(uv_bin_edges,uv_bin_edges,
                                       chan_gridded_uvplane)(uv_bin_edges_0,uv_bin_edges_0)
                    """
                    kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge") # no edge effects!! rigorously tested in July 2025
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
                            reshaped_u=np.reshape(u_here,N_here)
                            reshaped_v=np.reshape(v_here,N_here)
                            gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                            width_here=np.sqrt((1-eps_i)*(1-eps_j)*fidu_type_k*fidu_type_l)*pbw_fidu_use
                            kernel=PA_Gaussian(uubins,vvbins,[0.,0.],width_here)
                            kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge") # no edge effects!! rigorously tested in July 2025
                            convolution_here=convolve(kernel_padded,gridded,mode="valid") # beam-smeared version of the uv-plane for this perturbation permutation
                            uvplane+=convolution_here

        uvplane*=self.kaiser_grid # this tapering is to avoid ringing. power spectrum–geared tapering and per-antenna box normalization happen separately, of course
        uv_bin_edges=[uvbins,uvbins]
        return uvplane,uv_bin_edges,thetamax # this is the gridded uvplane

    def stack_to_box(self, tol:float=img_bin_tol):
        if (self.nu_ctr_MHz.value<(350/(1-self.evol_restriction_threshold/2)) or 
            self.nu_ctr_MHz>(nu_HI_z0/(1+self.evol_restriction_threshold/2))):
            raise ValueError("survey out of bounds")
        N_grid_pix=self.N_grid_pix
        kaiser_1d=kaiser(N_grid_pix,per_antenna_beta)
        kaiser_x,kaiser_y=np.meshgrid(kaiser_1d,kaiser_1d,indexing="ij")
        self.kaiser_grid=np.sqrt(kaiser_x**2+kaiser_y**2)

        # rescale chromatic beam widths by whatever was passed
        xy_beam_widths=np.array((self.surv_beam_widths,self.surv_beam_widths)).T
        ctr_chan_beam_width=(c/(self.nu_ctr_Hz*D))
        xy_beam_widths[:,0]*=(self.pbw_fidu[0]/ctr_chan_beam_width)
        xy_beam_widths[:,1]*=(self.pbw_fidu[1]/ctr_chan_beam_width)

        box_uvz=np.zeros((N_grid_pix,N_grid_pix,self.N_chan))
        xy_beam_widths_desc=np.flip(xy_beam_widths,axis=0)

        for i,xy_beam_width in enumerate(xy_beam_widths_desc): # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/self.surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=self.surv_wavelengths[i] # update the observing frequency for next time
            nu_obs=c/self.lambda_obs
            self.nu_obs=nu_obs.reduce()

            # compute the dirty image
            chan_gridded_uvplane,chan_uv_bin_edges,thetamax=self.calc_dirty_image(Npix=N_grid_pix, pbw_fidu_use=xy_beam_width, tol=tol)
            uv_bin_edges=chan_uv_bin_edges[0]
            
            # interpolate to store in stack
            if i==0:
                uv_bin_edges_0=chan_uv_bin_edges[0]
                theta_max_box=thetamax
                interpolated_slice=chan_gridded_uvplane
                d2u=self.d2u
            else: # chunk excision and mode interpolation in one step
                interpolated_slice=RBS(uv_bin_edges,uv_bin_edges,
                                       chan_gridded_uvplane)(uv_bin_edges_0,uv_bin_edges_0)
            box_uvz[:,:,i]=interpolated_slice
            if ((i%(self.N_chan//3))==0):
                print("{:7.1f} pct complete".format(i/self.N_chan*100))
        box_xyz=fftshift(ifftn(ifftshift(box_uvz*d2u),
                               axes=(0,1),norm="forward").real) # mixed coords before; all config space after
        for i in range(self.N_chan): # the correct generalization is per-channel normalization
            slice_i=box_xyz[:,:,i]
            box_xyz[:,:,i]=slice_i/np.max(slice_i)# peak-normalize in configuration space, just like I did for unif. across array beams
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
                 Nxy:int=128):                                        # number of pixels per side of frequency slides (get one sky plane square per CST file)
        self.beam_sim_directory=beam_sim_directory
        self.f_head=f_head
        self.f_mid1=f_mid1
        self.f_mid2=f_mid2
        self.f_tail=f_tail
        self.box_outname=box_outname
        freqs=np.arange(freq_lo,freq_hi,delta_nu_CST)
        self.freqs=freqs
        Nfreqs=len(freqs)
        self.Nfreqs=Nfreqs
        freqs_MHz_flipped=np.flip(freqs)*1000 # flip to get the ascending comoving distances I expect
        zs_for_xis=[nu_HI_z0/freq-1 for freq in freqs_MHz_flipped]
        xis=[comoving_distance(z) for z in zs_for_xis]
        xis=np.asarray(xis) # for the typical coeval approximation
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

        nu_ctr=(freq_lo+freq_hi)*500 # take an arithmetic average but also *1000 for GHz to MHz
        N_bl=int(N_ant*(N_ant-1)/2)
        k_perp=kperp(nu_ctr,b_EW,bmax)
        L_xy=twopi/k_perp[0]
        xy_for_box=L_xy*fftshift(fftfreq(Nxy))
        self.xy_for_box=xy_for_box
        np.save("xy_vec_for_box"+box_outname,xy_for_box.value)
        self.Nxy=Nxy
        self.xx_grid,self.yy_grid=np.meshgrid(self.xy_for_unwrapping,self.xy_for_unwrapping,
                                              indexing="ij") # config space points of interest for the slice (guided by the transverse extent of the eventual config-space box)
        freq_names=np.zeros(Nfreqs,dtype=str) # store the GHz CST frequencies as strings of the format that Aditya's sims use
        for i,freq in enumerate(self.freqs):
            freq_name=str(np.round(freq,4)) # round to four decimal places and convert to string
            freq_names[i]=freq_name.rstrip("0") # strip trailing zeros
        self.freq_names=freq_names

    def translate_sim_beam_slice(self,CST_filename:str,i:int=0):
        df = pd.read_table(CST_filename, skiprows=[0, 1,], sep='\s+', 
                           names=['theta', 'phi', 'AbsE', 'AbsCr', 'PhCr', 'AbsCo', 'PhCo', 'AxRat'])
        power=10**(df.AbsE.values/10) # non-log values
        theta_deg=df.theta.values
        theta=theta_deg*pi/180
        phi_deg=df.phi.values
        phi=phi_deg*pi/180
        x=self.xis[i]*theta*np.cos(phi)
        y=self.xis[i]*theta*np.sin(phi)
        sky_xy_points=np.array([x,y]).T
        return sky_xy_points,power
    
    def gen_box_from_simulated_beams(self):
        slice_grid_points=np.array([self.xx_grid,self.yy_grid]).T
        box=np.zeros((self.Nxy,self.Nxy,self.Nfreqs)) # hold interpolated beam slices
        for i,freq in enumerate(self.freqs):
            sky_xy_points,uninterp_slice_pol1=self.translate_sim_beam_slice(self.beam_sim_directory+
                                                                            self.f_head+str(np.round(freq,2))+
                                                                            self.f_mid1+self.f_tail,
                                                                            i=i) # both polarizations will be sampled at the same (theta,phi) because they come from the same simulation = same discretization
            _,            uninterp_slice_pol2=self.translate_sim_beam_slice(self.beam_sim_directory+
                                                                            self.f_head+str(np.round(freq,2))+
                                                                            self.f_mid2+self.f_tail,
                                                                            i=i)            

            product=uninterp_slice_pol1*uninterp_slice_pol2
            product_interpolated=gd(sky_xy_points,product,slice_grid_points,  # assumes pol1, pol2 discretized the same way... they will be, for sensibly-configured simulations
                                    method="nearest") # linear applies nans when extrap would be necessary
            power=product_interpolated/np.max(product_interpolated)
            box[:,:,i]=power
        np.save("CST_box_"+self.box_outname,box)
        self.box=box

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
        latitude:float = DRAO_lat*u.radian,           # latitude of the observatory (default = DRAO)
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
        assert(bl_max.unit==u.m and bl_min.unit==u.m and dish_diameter.unit==u.m)
        assert(freq_cen.unit==u.MHz and bandwidth.unit==u.MHz)
        assert(integration_time.unit==u.s)
        assert(tsky_ref_freq.unit==u.MHz and tsky_amplitude.unit==u.K and horizon_buffer.unit==littleh/u.Mpc)
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
        n_channels = bandwidth.value/CHORD_channel_width_MHz
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
            xv, yv = np.meshgrid(x,y)
            xy = np.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))

        if len(xy) != n_total:
            raise ValueError('Sizes of x- and y-locations do not agree with n_total')

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

def memo_ii_plotter(ensemble_of_spectra:np.ndarray,                      # indexed as (N_complexity_cases, N_k_perp, N_k_par)
                    ensemble_ids:np.ndarray,                             # names for each power spectrum quantity ("spectrum" for short, even though this is a misnomer in the case of ratios and residuals) in the ensemble according to the number of fiducial and perturbed beam types (N_complexity_cases,)
                    colourmap,                                           # for imshowing each power spectrum quantity
                    k_perp:np.ndarray, k_par:np.ndarray,                 # k-perp and k-par bins that anchor each plotted spectrum
                    case_title:str, case_units:str,                      # title describing this power spectrum quantity and the corresponding units
                    save_name:str,                                       # name for the summary figure
                    norm_mid, norm_ext,                                  # if there is a physically motivated natural middle of the colour bar (e.g. 1 for a ratio or 0 for a residual), pass it to the plotter along with the extent of the range about this midpoint (possibly informed by the extent of the systematics you plugged into the simulation)
                    k1_inset:float=0.06/u.Mpc, k2_inset:float=2.5/u.Mpc, # k-scales of interest to sample each spectrum in the ensemble
                    qty_to_plot:str="P"):                                # pre-established options: P, $\Delta^2$
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
    assert(k_perp.unit==1/u.Mpc and k_par.unit==1/u.Mpc)
    cyl_extent=[k_perp[0].value,k_perp[-1].value,k_par[0].value,k_par[-1].value]
    k_perp_grid,k_par_grid=np.meshgrid(k_perp,k_par)
    k_mag_grid=np.sqrt(k_perp_grid**2+k_par_grid**2)
    values_of_k=np.zeros((N_spectra,2))

    fig = plt.figure(figsize=(N_LHS_cols*4, N_LHS_cols*4),layout="constrained")
    gs = gridspec.GridSpec(N_LHS_rows, N_LHS_cols+2, figure=fig)
    axs = [[fig.add_subplot(gs[row, col]) for col in range(N_LHS_cols)] for row in range(N_LHS_rows)] # grid for the left
    ax_right = fig.add_subplot(gs[:, N_LHS_cols:]) # summary holder on the right

    for k in range(N_spectra):
        i=k//N_LHS_cols
        j=k%N_LHS_cols
        spec=ensemble_of_spectra[k,:,:] # remaining indices: N complexity cases, N k-perp, N k-par
        specshape=spec.shape
        if qty_to_plot=="Delta2":
            spec_to_plot=spec*k_mag_grid**3/(2*pi**2)
        elif qty_to_plot=="P":
            spec_to_plot=np.copy(spec)
        else:
            raise ValueError("P and Delta2 are the only pre-established plotting options for now")

        if (norm_mid is not None and norm_ext is not None): # branch for relative quantities: use a known centre (0 or 1 for residual or ratio) and desired half-range
            norm=CenteredNorm(vcenter=norm_mid,halfrange=norm_ext)
        else:
            halfmax=0.5*np.percentile(ensemble_of_spectra,99.5) # branch for absolute quantities: put all power spectra in the ensemble on the same colour scales, informed by the extreme range
            norm=CenteredNorm(vcenter=halfmax,halfrange=halfmax)
        im=axs[i][j].imshow(spec_to_plot.T, cmap=colourmap, origin="lower", extent=cyl_extent, norm=norm)
        axs[i][j].set_xlabel("k$_\perp$")
        axs[i][j].set_ylabel("k$_{||}$")
        axs[i][j].tick_params(axis='x', labelrotation=30)
        axs[i][j].set_title(ensemble_ids[k])
        axs[i][j].set_aspect("equal")
        plt.colorbar(im,ax=axs[i][j],shrink=0.6,extend="both")

        idx_for_k1=np.argmin(np.abs(k_mag_grid-k1_inset))
        idx_for_k1=np.unravel_index(idx_for_k1,specshape)
        idx_for_k2=np.argmin(np.abs(k_mag_grid-k2_inset))
        idx_for_k2=np.unravel_index(idx_for_k2,specshape)
        values_of_k[k,0]=spec[idx_for_k1]
        values_of_k[k,1]=spec[idx_for_k2]

    complexity_indices=np.arange(N_spectra)
    ax_right.scatter(complexity_indices,values_of_k[:,0],label=str(np.round(k1_inset,4))+" (~1st BAO wiggle scale)")
    ax_right.scatter(complexity_indices,values_of_k[:,1],label=str(np.round(k2_inset,4))+" (~CHIME scale)")
    ax_right.set_xticks(complexity_indices, labels=ensemble_ids, rotation=40)
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

def power_comparison_plots(redo_window_calc:bool=False, redo_box_calc:bool=False,
              mode:str="pathfinder", nu_ctr:float=800, epsxy:float=0.1,
              frac_tol_conv=0.1, N_sph=256,categ="PA", # categ is manual/PA/CST, beam_type is either Gaussian (for PA) or manual (for CST)
              N_fidu_types=1, N_pert_types=0, 
              N_pbws_pert=0, per_channel_systematic=None,
              PA_dist="random", plot_qty="P",
              Nkpar_box=None,Nkperp_box=None, 
                  
              wedge_cut=False, layer_foregrounds=False, pointing_error=[0.,0.,0.],
                  
              freq_bin_width=0.1953125*u.MHz,

              CST_lo=None,CST_hi=None,CST_deltanu=None,
              beam_sim_directory=None,f_mid1=")_[1]",f_mid2=")_[2]",f_tail="_efield.txt",
              CST_f_head_fidu="farfield_(f=",CST_f_head_real="farfield_(f=",CST_f_head_thgt="farfield_(f=",
              
              from_incomplete_MC=False,
              contaminant_or_window=None, k_idx_for_window=0,
              isolated=False,seed=None,
              per_chan_syst_facs=[]): # the default chromaticity systematic
    save_args_to_file(inspect.currentframe())

    ############################## other survey management factors ########################################################################################################################
    assert(nu_ctr.unit==u.MHz)
    nu_ctr_Hz=nu_ctr.value*1e6*u.Hz
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
        print("PA mode currently only supports a Gaussian beam")
    hpbw_x= dif_lim_prefac*wl_ctr_m/D # rad; lambda/D estimate
    hpbw_y= 0.75*hpbw_x # simulations show this is characteristic of the UWB feeds

    ############################## pipeline administration ########################################################################################################################
    if contaminant_or_window is not None:
        c_or_w="wind"
    else:
        c_or_w="cont"
    per_chan_syst_string="none"
    per_chan_syst_name=""
    if per_channel_systematic=="early_transit_measurement_like":
        per_chan_syst_string="D3AL"
    elif per_channel_systematic=="sporadic":
        per_chan_syst_string="spor"
        for fac in per_chan_syst_facs:
            per_chan_syst_name=per_chan_syst_name+str(fac)+"_"
    elif per_channel_systematic is not None:
        raise ValueError("unknown per_channel_systematic")
    PA_dist_string="rand"
    if PA_dist=="corner":
        PA_dist_string="corn"
    elif PA_dist=="column":
        PA_dist_string="rwcl"
    elif PA_dist!="random":
        raise ValueError("unknown PA_dist")

    # setup for the new regime 
    if type(N_fidu_types)==int:
        N_fidu_types=[N_fidu_types]
        N_pert_types=[N_pert_types]

    complexity_cases=[[a,b] for b in N_pert_types for a in N_fidu_types]
    complexity_ids=[str(case) for case in complexity_cases]
    f_types_prefacs=get_f_types_prefacs(complexity_cases)
    power_quantities_all=[]
    for i,complexity_type in enumerate(complexity_cases):
        t00=time.time()
        N_fidu_types_i,N_pert_types_i=complexity_type
        if N_pert_types_i==0: # loop over complexity cases–friendly number of antennas with perturbed beams
            N_pbws_pert_i=0
        else:
            N_pbws_pert_i=N_pbws_pert
        f_types_prefacs_i=f_types_prefacs[i]
        ioname=mode+"_"+c_or_w+"_"+categ+"_"\
           ""+per_chan_syst_string+"_"+per_chan_syst_name+"_"\
           ""+str(int(nu_ctr.value))+"MHz__"\
           "Nreal_"+str(N_fidu_types_i)+"__"\
           "Npert_"+str(N_pert_types_i)+"_"+str(N_pbws_pert)+"__"\
           "dist_"+PA_dist_string+"__"\
           "epsxy_"+str(epsxy)+"__"\
           "layer_"+str(layer_foregrounds)+"__"\
           "wedge_"+str(wedge_cut)+"__"\
           "seed_"+str(seed)
        
        if (N_fidu_types_i!=4 and PA_dist=="corner"):
            continue

        # PIPELINE ADMIN FOR THIS PA SYSTEMATIC PERMUTATION
        bundled_non_manual_primary_aux=np.array([hpbw_x,hpbw_y])
        pbunc=epsxy
        if categ=="PA":
            windowed_survey=beam_effects(# SCIENCE
                                            # the observation
                                            bminCHORD,bmaxCHORD,                                          
                                            nu_ctr,freq_bin_width,                                         
                                            evol_restriction_threshold=def_evol_restriction_threshold,     
                                                
                                            # beam generalities
                                            primary_beam_categ=categ,primary_beam_type="Gaussian",   
                                            primary_beam_aux=bundled_non_manual_primary_aux,
                                            primary_beam_unc=pbunc,               
                                            manual_primary_beam_modes=None,                                 

                                            # additional considerations for per-antenna systematics
                                            PA_N_pert_types=N_pert_types_i,PA_N_pbws_pert=N_pbws_pert_i,PA_N_fidu_types=N_fidu_types_i,
                                            PA_fidu_types_prefactors=f_types_prefacs_i,PA_ioname=ioname,PA_distribution=PA_dist,mode=mode,
                                            per_channel_systematic=per_channel_systematic,per_chan_syst_facs=per_chan_syst_facs,

                                            # FORECASTING
                                            P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,
                                            wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds, pointing_error=pointing_error,

                                            # NUMERICAL 
                                            n_sph_modes=N_sph,                                        
                                            init_and_box_tol=0.05,CAMB_tol=0.05,                              
                                            Nkpar_box=Nkpar_box,Nkperp_box=Nkperp_box,frac_tol_conv=frac_tol_conv,                  
                                            seed=seed,                                         
                                            ftol_deriv=1e-16,maxiter=5,                                       
                                            PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,      
                                            radial_taper=kaiser,image_taper=None,

                                            # CONVENIENCE
                                            heavy_beam_recalc=redo_box_calc                                                    
                                            
                                            )

        elif categ=="CST":
            windowed_survey=beam_effects(# SCIENCE
                                        # the observation
                                        bminCHORD,bmaxCHORD,                                                       
                                        nu_ctr,freq_bin_width,                                                 
                                        evol_restriction_threshold=def_evol_restriction_threshold,           
                                            
                                        # beam generalities
                                        primary_beam_categ=categ,primary_beam_type="Gaussian",           
                                        primary_beam_aux=bundled_non_manual_primary_aux,
                                        primary_beam_unc=pbunc,                      
                                        manual_primary_beam_modes=None,                              

                                        # numerical beam perturbation parameters
                                        PA_N_pert_types=1,PA_N_pbws_pert=N_ant,
                                        PA_N_fidu_types=1,
                                        PA_fidu_types_prefactors=[1.],
                                        PA_distribution="random",mode=mode,

                                        # additional considerations for CST
                                        CST_lo=CST_lo,CST_hi=CST_hi,CST_deltanu=CST_deltanu,PA_ioname=ioname,
                                        beam_sim_directory=beam_sim_directory,f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,
                                        CST_f_head_fidu=CST_f_head_fidu,CST_f_head_real=CST_f_head_real,CST_f_head_thgt=CST_f_head_thgt,

                                        # FORECASTING
                                        P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,
                                        wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds, pointing_error=pointing_error,

                                        # NUMERICAL 
                                        n_sph_modes=N_sph,                                        
                                        init_and_box_tol=0.05,CAMB_tol=0.05,                                 
                                        Nkpar_box=Nkpar_box,Nkperp_box=Nkperp_box,frac_tol_conv=frac_tol_conv,                         
                                        seed=seed,                                         
                                        ftol_deriv=1e-16,maxiter=5,           
                                        radial_taper=kaiser,image_taper=None,

                                        # CONVENIENCE
                                        heavy_beam_recalc=redo_box_calc                                                   
                                        
                                        )
        else:
            raise ValueError("unknown systematics category (categ)")
        
        handle_fi=False
        handle_rt=False
        handle_sf=False
        if isolated==False:     # recalculate all three MC-windowed power spectra [see i, ii, iii below]
            handle_fi=True
            handle_rt=True
            handle_sf=True
        if isolated=="realthgt": # recalculate only the theory + fidu beam + syst + meas errs + ?fg? power spec [i]
            handle_rt=True
        if isolated=="fidufidu": # recalculate only the theory + fidu beam + ?fg? power spec [ii]
            handle_fi=True
        if isolated=="contam":   # recalculate only the above two power spectra
            handle_fi=True
            handle_rt=True
        if isolated=="flatrlth": # recalculate only the fidu beam + syst + meas errs + ?fg? power spec [iii]
            handle_sf=True

        windowed_survey.print_survey_characteristics()
        if not from_incomplete_MC:
            if redo_window_calc:
                t0=time.time()
                windowed_survey.calc_power_contamination(isolated=isolated) # loops over complexity
                Ptheory=windowed_survey.Ptheory_cyl
                np.save("Ptheory_"+ioname+".npy",Ptheory)
                t1=time.time()
                print("Pcont calculation time was",t1-t0)

                if handle_fi:
                    Pfiducial=windowed_survey.Pfiducial_cyl
                    np.save("Pfiducial_cyl_"+ioname+".npy",Pfiducial)
                if handle_rt:
                    Prealthought=windowed_survey.Prealthought_cyl
                    np.save("Prealthought_"+ioname+".npy",Prealthought)
                if handle_sf:
                    Pnotheory=windowed_survey.Pnotheory_cyl
                    np.save("Pnotheory_"+ioname+".npy",Pnotheory)
                N_per_realization=windowed_survey.N_per_realization
                np.save("N_per_realization_"+ioname+".npy",N_per_realization)
                kperp_internal=windowed_survey.kperpbins_internal[:-1]
                kpar_internal=windowed_survey.kparbins_internal[:-1]
                np.save("kpar_internal_"+ioname+".npy",kpar_internal.value)
                np.save("kperp_internal_"+ioname+".npy",kperp_internal.value)
                if isolated is not False: # break early if you just calculate one windowed power spectrum at a time
                    return None
            else:
                Prealthought=np.load("Prealthought_"+ioname+".npy")
                Pfiducial=np.load("Pfiducial_cyl_"+ioname+".npy")
                Pnotheory=np.load("Pnotheory_"+ioname+".npy")
                Ptheory=np.load("Ptheory_"+ioname+".npy")
                N_per_realization=np.load("N_per_realization_"+ioname+".npy")
                kpar_internal=np.load("kpar_internal_"+ioname+".npy")/u.Mpc
                kperp_internal=np.load("kperp_internal_"+ioname+".npy")/u.Mpc
        else:
            Prealthought=np.load("P_rt_unconverged.npy")
            Pfiducial=np.load("P_fi_unconverged.npy")
            Pnotheory=np.load("P_sf_unconverged.npy")
            N_per_realization=np.load("N_per_realization_"+ioname+".npy")
            kpar_internal=np.load("kpar_internal_"+ioname+".npy")/u.Mpc
            kperp_internal=np.load("kperp_internal_"+ioname+".npy")/u.Mpc

        Presidual= Prealthought-Pfiducial
        Pratio=    Pnotheory/Ptheory

        power_quantities_this_complexity=np.array([Pnotheory, Pfiducial, Prealthought, Presidual, Pratio]) # 5 x Nkperp x Nkpar
        power_quantities_all.append(power_quantities_this_complexity) # N_complexity_cases x 5 x Nkperp x Nkpar
        t01=time.time()
        print("handled complexity case",complexity_ids[i],"in",t01-t00,"s")

    power_quantities_all=np.asarray(power_quantities_all)
    N_plots=5 # hard-coded for this generation of plots where I can look at the same feasibility analysis for different systematics families
    abs_map=cmasher.voltage # also consider cosmic, eclipse, amber, dusk, rainforest, fall, ...others
    rel_map=cmasher.prinsenvlag # also consider viola, ...others

    absolute_units="(mK$^2$ Mpc$^3$)"
    relative_units="(unitless)"

    ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   
    plot_version_names = ["fidu beam + syst + meas errs + fg", "theory + fidu beam + fg", "theory + fidu beam + syst + meas errs + fg", 
                          "(theory + fidu beam + syst + meas errs + fg) - (theory + fidu beam + fg)", "(fidu beam + syst + meas err + fg) / theory"]
    save_names= ["fidu_syst_measerrs_fg", "theory_fidu_fg", "theory_fidu_syst_measerrs_fg", 
                 "theory_fidu_syst_measerrs_fg__minus__theory_fidu_fg", "fidu_syst_measerrs_fg__divby__theory"]
    plot_cmaps= [abs_map, abs_map, abs_map,
                 rel_map, rel_map]
    norm_mids=  [None,None,None,
                 None,None]
                #  0.,1.]
    norm_exts=  [None,None,None,
                 None,None]
                #  250,10e11]
    plot_units=[absolute_units,absolute_units,absolute_units,
                absolute_units,relative_units]
    ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   

    for i in range(N_plots): # iterate over plot cases
        power_quantity_this_plot_case=power_quantities_all[:,i,:,:] # [:,i,:,:] = all complexity cases, ith power spectrum quantity, all kperps, all kpars
        memo_ii_plotter(power_quantity_this_plot_case, complexity_ids, plot_cmaps[i], 
                        kperp_internal, kpar_internal, 
                        plot_version_names[i], plot_units[i], save_names[i], norm_mids[i], norm_exts[i],
                        qty_to_plot=plot_qty)