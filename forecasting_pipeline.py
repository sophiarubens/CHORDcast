import numpy as np
from numpy.fft import fftshift,ifftshift,fftfreq, fftn,ifftn, irfftn

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import CenteredNorm,LogNorm

from scipy.signal import convolve
from scipy.signal.windows import kaiser
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata as gd
from scipy.special import j1

import camb
from camb import model

from astropy import units as u
from astropy.cosmology.units import littleh
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum

import cmasher
import pandas as pd
from itertools import permutations
import pygtc
import time
import inspect
import json

from cosmo_distances import *

# cosmological
Omegam_Planck18=0.3158
Omegabh2_Planck18=0.022383
Omegach2_Planck18=0.12011
OmegaLambda_Planck18=0.6842
lntentenAS_Planck18=3.0448
tentenAS_Planck18=np.exp(lntentenAS_Planck18)
AS_Planck18=tentenAS_Planck18/10**10
ns_Planck18=0.96605
H0_Planck18=67.32
h_Planck18=H0_Planck18/100.
w=-1
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_fidu=    [H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18,w] # suitable for get_mps
parnames_fidu=['H_0',       'Omega_b h^2',      'Omega_c h^2',      '10^9 * A_S',        'n_s'  ,"w"     ]

pars_forecast=    [H0_Planck18, Omegabh2_Planck18, Omegach2_Planck18, w  ]
parnames_forecast=["H_0",       "Omega_b h^2",     "Omega_c h^2",     "w"]

scale=1e-9
dpar_default=1e-3*np.ones(len(pars_fidu))
dpar_default[3]*=scale

# physical
nu_HI_z0=1420.405751768 # MHz
c=2.998e8 # m/s
dif_lim_prefac=1.029

# mathematical
pi=np.pi
twopi=2.*pi
ln2=np.log(2)

# computational
infty=np.inf 
maxfloat= np.finfo(np.float64).max
huge=np.sqrt(maxfloat)
maxfloat= np.finfo(np.float64).max
maxint=   np.iinfo(np.int64  ).max
nearly_zero=1e-30
symbols=["o","*","v","s", # circle, star, eq tri vtx dwn, sq edge up
         "H","d","1","8", # hex exdge up, diamond, thirds-division pt dwn, octagon
         "p","P","h","X"] # pentagon, filled +, hex vtx up, filled x

# numerical
scale=1e-9
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # intentioanally preposterous number of sig figs from Mathematica
eps=1e-15
per_antenna_beta=14
cosmo_stats_beta_par=14 # the starting point recommended in the documentation and, after some quick tests, more suitable than beta=2, 6, or 20
cosmo_stats_beta_perp=14
dpi_to_use=250

# CHORD
N_NS_full=24
N_EW_full=22
b_NS=8.5
b_EW=6.3
DRAO_lat=49.320791*pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6. # m
CHORD_channel_width_MHz=0.1953125
def_observing_dec=pi/60.
def_offset_deg=1.75*pi/180. # for this placeholder state where I build up the CHORD layout using rotation matrices instead of actual measurements. probably add Hans' mask at some point to punch the corners and receiver hut holes out...
def_pbw_pert_frac=1e-2
def_evol_restriction_threshold=1./15.
img_bin_tol=5 # ringing is remarkably insensitive to turning this down; you get really bad scale mismatch by turning it up... the real solution was the "need good resolution in both Fourier and configuration space" thing
def_PA_N_grid_pix=256 # can turn this down from 512 since it doesn't change the deltaxy and a lower number of pixels per side means eval will be faster
N_fid_beam_types=1
integration_s=10 # seconds
hrs_per_night=8 # borrowed from Debanjan / 21cmSense
N_nights=100 # also borrowed from Debanjan / 21cmSense

# side calculations
def get_padding(n): # avoid edge effects in a convolution
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi
def synthesized_beam_crossing_time(nu,bmax,dec=30.): # to accumulate rotation synthesis
    synthesized_beam_width_rad=1.029*(c/nu)/bmax
    beam_width_deg=synthesized_beam_width_rad*180/pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec*pi/180)
    return crossing_time_hrs
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None

# beams
def PA_Gaussian(u,v,ctr,fwhm):
    u0,v0=ctr
    fwhmx,fwhmy=fwhm
    evaled=np.exp(-pi**2*((u-u0)**2*fwhmx**2+(v-v0)**2*fwhmy**2)/np.log(2)) # prefactor ((pi*ln2)/(fwhmx*fwhmy)) will be overwritten during normalization anyway
    return evaled

# the actual pipeline!!
"""
this class helps compute contaminant power and cosmological parameter biases
using a Fisher-based formalism and numerical windowing for power beams with  
assorted properties and systematics.
"""

class beam_effects(object):
    def __init__(self,
                 # SCIENCE
                 # the observation
                 bmin:float,bmax:float,                                                             # extreme baselines of the array
                 nu_ctr:float,delta_nu:float,                                                       # for the survey of interest
                 evol_restriction_threshold:float=def_evol_restriction_threshold,             # how close to coeval is close enough?
                 
                 # beam generalities
                 primary_beam_categ:str="PA",primary_beam_type:str="Gaussian",                  # modelling choices
                 primary_beam_aux=None,primary_beam_uncs=None,                          # helper arguments... usage depends on systematics mode. see below
                 manual_primary_beam_modes=None,                                        # config space pts at which a pre–discretely sampled primary beam is known

                 # additional considerations for per-antenna systematics
                 PA_N_pert_types=0,PA_N_pbws_pert=0,                                    # numbers of perturbation types, primary beam widths to perturb
                 PA_N_fidu_types=N_fid_beam_types,PA_fidu_types_prefactors=None,        # how many kinds of fiducial beams and how to set them apart
                 PA_ioname="placeholder",                                               # numbers of timesteps to put in rotation synthesis, in/output file name
                 PA_distribution="random",mode="full",per_channel_systematic=None,      # how to spread beam types throughout the array and along the frequency axis; whether use use full or pf CHOR
                 per_chan_syst_facs=[1.05,0.9,1.25],                                    # multiplicative prefracs by which chunks of survey band have the wrong beam width

                 # additional considerations for CST beams
                 CST_lo=None,CST_hi=None,CST_deltanu=None,                                                      # lo, hi, spacing of CST frequencies to read
                 beam_sim_directory=None,f_mid1=")_[1]",f_mid2=")_[2]",f_tail="_efield.txt",                    # info about files to import from 
                 CST_f_head_fidu="farfield_(f=",CST_f_head_real="farfield_(f=",CST_f_head_thgt="farfield_(f=",  # more of the same

                 # FORECASTING
                 pars_set_cosmo=pars_fidu,pars_forecast=pars_fidu, pars_forecast_names=parnames_fidu, # implement soon: add the flexibility to put derived and not just base parameters in pars_forecast
                 P_fid_for_cont_pwr=None, k_idx_for_window=0,                                         # examine contaminant power or window functions?
                 interp_to_survey_modes=False,                                                        # don't bother turning down the k-space resolution to literal instrument-accessible modes
                 wedge_cut=False,layer_foregrounds=False,                                             # foreground toggles

                 # NUMERICAL 
                 n_sph_modes=256,dpar=None,                                             # conditioning the CAMB/etc. call
                 init_and_box_tol=0.05,CAMB_tol=0.05,                                   # considerations for k-modes at different steps
                 Nkpar_box=None,Nkperp_box=None,frac_tol_conv=0.1,                      # considerations for cyl binned power spectra from boxes
                 seed=None,                                                             # if you want a particular rng
                 ftol_deriv=1e-16,maxiter=5,                                            # guardrails for numerical derivative calculation
                 PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,            # pixels per side of gridded uv plane, uv binning chunk snapshot tightness
                 radial_taper=None,image_taper=None,                                    # apply apodization along the line of sight or transverse directions?

                 # CONVENIENCE
                 heavy_beam_recalc=True                                                 # save time by not repeating per-antenna calculations? 

                 ):                                                                                                                                                     
                
        """
        bmin,bmax                  :: floats                       :: max and min baselines of the array       :: m
        primary_beam_categ         :: str                          :: * PA  = per-antenna                      :: ---
                                                                      * CST = computer simulation technology
        primary_beam_type          :: str                          :: * PA: Gaussian                           :: ---
        primary_beam_aux           :: (N_args,) of floats          :: * manual: primary beams evaluated on the :: r0:           Mpc
                                                                      * grid of interest, a list ordered as       fwhms:        rad
                                                                        [fidu,pert]                               evaled beams: ---
                                                                      * PA: FWHMs 
        primary_beam_uncs          :: (2,) of floats               :: fractional uncertainties for x and y     :: ---
        pars_set_cosmo             :: (N_fid_pars,) of floats      :: params to condition a CAMB/etc. call     :: as found in ΛCDM
        pars_forecast              :: (N_forecast_pars,) of floats :: params for which you'd like to forecast  :: as found in ΛCDM
        n_sph_modes                :: int                          :: # modes to put in CAMB/etc. MPS          :: ---
        dpar                       :: (N_forecast_pars,) of floats :: initial guess of num. dif. step sizes    :: same as for pars_forecast
        nu_ctr                     :: float                        :: central freq for survey of interest      :: MHz
        delta_nu                   :: float                        :: channel width for survey of interest     :: MHz
        evol_restriction_threshold :: float                        :: ~$\frac{\Delta z}{z}$ w/in survey box    :: ---
        init_and_box_tol, CAMB_tol :: floats                       :: how much wider do you want the k-ranges  :: ---
                                                                      of preceding steps to be? (frac tols)
        ftol_deriv                 :: float                        :: frac tol relating to scale of fcn range  :: ---
        eps                        :: float                        :: tiny offset factor to protect against    :: --- 
                                                                      numerical division-by-zero errors
        maxiter                    :: int                          :: maximum # of times to let the step size  :: ---
                                                                      optimization recurse before giving up
        Nkpar_box,Nkperp_box       :: ints                         :: # modes to put along cyl axes in power   :: ---
                                                                      spec calcs from boxes
        frac_tol_conv              :: float                        :: how much the Poisson noise must fall off :: ---
        pars_forecast_names        :: (N_pars_forecast,) or equiv. :: names of the pars being forecast         :: ---
                                      of strs
        manual_primary_beam_modes  :: x,y,z coordinate axes        :: domain of a discrete sampling            :: Mpc
                                      (if primary_beam !callable)
        PA_N_pert_types            :: int                          :: # classes of PB (per-antenna only)       :: ---
        PA_N_pbws_pert             :: int                          :: # antennas w/ pertn PBs (per-ant only)   :: ---
        PA_N_timesteps             :: int                          :: # time steps in rotation synthesis (per- :: ---
                                                                      antenna only)
        PA_N_grid_pix              :: int                          :: # bins per side for uv plane gridding    :: ---
                                                                      (per-antenna only)
        PA_img_bin_tol             :: float                        :: # how much padding (to avoid ringing) to :: ---
                                                                      put in uv-plane gridding (per-ant only)
        PA_ioname                  :: str                          :: fname to save/load stacked per-ant boxes :: ---
        heavy_beam_recalc                  :: bool                         :: recalculate per-antenna beamed boxes?    :: ---
        PA_distribution            :: str                          :: how to distribute perturbation types     :: ---
        PA_N_fidu_types   :: int
        PA_fidu_types_prefactors   :: (PA_N_fidu_types,)  :: initial inroads into making the dif fidu :: ---
                                      of floats                       beam classes actually dif (multiplic.
                                                                      prefactor compared to lambda/D)
        mode                       :: str                          :: full, PF, or intermed states tbd later   :: ---

        short-term extensions:
        * the flexibility to introduce per-channel chromaticity systematics for each fiducial beam class
        """
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
        self.surv_channels=np.arange(self.nu_lo,self.nu_hi,self.Deltanu)
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
        N_bl=int(N_ant*(N_ant-1)/2)
        
        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        kparmin_surv=kpar_surv[0]
        kparmax_surv=kpar_surv[-1]
        self.kpar_surv=kpar_surv
        self.kparmin_surv=kparmin_surv
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        bmax=bmax
        kperp_surv=kperp(self.nu_ctr,N_bl,self.bmin,bmax)
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
        print("beam_effects.__init__: Nxy,Nz=",self.Nvox_box_xy,self.Nvox_box_z)

        if layer_foregrounds:
            synchrotron_factors= 300*(np.linspace(self.nu_lo,self.nu_hi,self.Nvox_box_z)/150)**-2.5 # # cf. eq. 11 of Pober et al. 2012 for the normalization
            rng = np.random.default_rng()
            white_noise_box=rng.normal(size=(self.Nvox_box_xy,self.Nvox_box_xy,self.Nvox_box_z)) # loc=0.,scale=1.,
            fg_xy=np.linspace(-self.Lsurv_box_xy/2,self.Lsurv_box_xy/2,self.Nvox_box_xy)
            fg_z= np.linspace(-self.Lsurv_box_z/2, self.Lsurv_box_z/2, self.Nvox_box_z)
            self.foreground_field=white_noise_box*synchrotron_factors[None,None,:]
            self.fg_modes=[fg_xy,fg_xy,fg_z]


        # primary beam considerations
        self.primary_beam_categ=primary_beam_categ
        self.fwhm_x,self.fwhm_y=primary_beam_aux
        self.primary_beam_uncs= primary_beam_uncs
        self.epsx,self.epsy=    self.primary_beam_uncs

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
                self.PA_N_timesteps=           hrs_per_night*3600//integration_s
                self.PA_N_grid_pix=            PA_N_grid_pix
                self.img_bin_tol=              PA_img_bin_tol
                self.PA_distribution=          PA_distribution
                self.PA_N_fidu_types= PA_N_fidu_types
                self.PA_fidu_types_prefactors= PA_fidu_types_prefactors
                fwhm=primary_beam_aux 
                self.eps=primary_beam_uncs 

                fidu=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                pbw_pert_frac=[0.,0.],
                                N_timesteps=self.PA_N_timesteps,
                                N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                N_fiducial_beam_types=1,
                                outname=PA_ioname)
                real=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                 pbw_pert_frac=[0.,0.],
                                 N_timesteps=self.PA_N_timesteps,
                                 N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                 distribution=self.PA_distribution,
                                 N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                 outname=PA_ioname,
                                 per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
                thgt=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=self.PA_N_pert_types,
                                 pbw_pert_frac=self.primary_beam_uncs,
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
                    np.save("xy_vec_"+  PA_ioname+".npy",xy_vec)
                    np.save("z_vec_"+   PA_ioname+".npy",z_vec)
                else:
                    fidu_box=np.load("fidu_box_"+PA_ioname+".npy")
                    real_box=np.load("real_box_"+PA_ioname+".npy")
                    thgt_box=np.load("thgt_box_"+PA_ioname+".npy")
                    xy_vec=  np.load("xy_vec_"+  PA_ioname+".npy")
                    z_vec=   np.load("z_vec_"+   PA_ioname+".npy")

                primary_beam_aux=[fidu_box,real_box,thgt_box]
                manual_primary_beam_modes=(xy_vec,xy_vec,z_vec)
            
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
                np.save("z_vec"+PA_ioname+".npy",CST_z_vec)
            else:
                fidu_box=np.load("fidu_box_"+PA_ioname+".npy")
                real_box=np.load("real_box_"+PA_ioname+".npy")
                thgt_box=np.load("thgt_box_"+PA_ioname+".npy")
                CST_z_vec=np.load("z_vec"+PA_ioname+".npy")
            primary_beam_aux=[fidu_box,real_box,thgt_box]
            manual_primary_beam_modes=(precalculated_xy_vec,precalculated_xy_vec,CST_z_vec)

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
        self.primary_beam_uncs=primary_beam_uncs

        # groundwork-informed forecasting considerations
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
            self.perturbed_primary_beam_aux=(self.fwhm_x*(1-self.epsx),self.fwhm_y*(1-self.epsy))
            self.primary_beam_aux=np.array([self.fwhm_x,self.fwhm_y,self.r0]) 
            self.perturbed_primary_beam_aux=np.append(self.perturbed_primary_beam_aux,self.r0)
        else:
            raise ValueError("unknown primary_beam_type")
        self.P_fid_for_cont_pwr=P_fid_for_cont_pwr
        self.k_idx_for_window=k_idx_for_window

        # numerical protections for assorted k-ranges
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init*np.sqrt(3) # factor of sqrt(3) from pythag theorem for box to prevent the need for extrap
        self.ksph,self.Ptruesph=self.get_mps(self.pars_set_cosmo,kmin_CAMB,kmax_CAMB)
        self.Deltabox_xy=self.Lsurv_box_xy/self.Nvox_box_xy
        self.Deltabox_z= self.Lsurv_box_z/ self.Nvox_box_z
        self.radial_taper=radial_taper
        self.image_taper=image_taper

        # considerations for power spectra binned to survey k-modes
        # _,_,self.Pcyl=self.unbin_to_Pcyl(self.pars_set_cosmo)

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

        with open("settings.txt", "w") as file:
            file.write("primary beam width systematics category           = "+str(primary_beam_categ)+"\n")
            file.write("                               distribution       = "+str(PA_distribution)+"\n")
            file.write("central frequency of survey                       = "+str(nu_ctr)+"\n")
            file.write("observing setup                                   = "+str(mode)+"\n")
            file.write("Poisson noise convergence threshold               = "+str(self.frac_tol_conv)+"\n")
            file.write("per-channel systematic                            = "+str(per_channel_systematic)+"\n")
            file.write("number of fiducial beam types (if applicable)     = "+str(PA_N_fidu_types)+"\n")
            file.write("number of perturbed beam types (if applicable)    = "+str(PA_N_pert_types)+"\n")
            file.write("number of perturbed primary beams (if applicable) = "+str(PA_N_pbws_pert)+"\n")
            file.write("wedge cut in config space?                        = "+str(wedge_cut)+"\n")

    def get_mps(self,pars_use,minkh=1e-4,maxkh=1):
        """
        get matter power spectrum from CAMB
        """
        z=[self.z_ctr]
        H0=pars_use[0]
        h=H0/100.
        ombh2=pars_use[1]
        omch2=pars_use[2]
        As=pars_use[3]*scale
        ns=pars_use[4]

        pars_use_internal=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
        pars_use_internal.InitPower.set_params(As=As,ns=ns,r=0)
        pars_use_internal.set_matter_power(redshifts=z, kmax=maxkh*h)
        results = camb.get_results(pars_use_internal)
        pars_use_internal.NonLinear = model.NonLinear_none
        kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=self.n_sph_modes)

        return kh,pk
    
    def unbin_to_Pcyl(self,pars_to_use,kperp_to_use=None,kpar_to_use=None):
        """
        interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        """
        if kperp_to_use is None:
            kperp_to_use=self.kperp_surv
        if kpar_to_use is None:
            kpar_to_use=self.kpar_surv
        k,Psph_use=self.get_mps(pars_to_use,minkh=self.kmin_surv,maxkh=self.kmax_surv)
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
        interpolator=RGI((k,),Psph_use,
                         bounds_error=False,fill_value=None)
        Pcyl[sort_array]=interpolator(kmag_grid_flat_sorted[:, None])
        Pcyl=np.reshape(Pcyl,(Nkperp_use,Nkpar_use))

        return kpar_grid,kperp_grid,Pcyl

    def calc_power_contamination(self, isolated=False):
        """
        calculate a cylindrically binned Pcont from an average over the power spectra formed from beam-aware brightness temp boxes
        contaminant power, calculated as [see memo] useful combinations of three different instrument responses
        """
        if self.P_fid_for_cont_pwr is None:
            P_fid=np.reshape(self.Ptruesph,(self.n_sph_modes))
        elif self.P_fid_for_cont_pwr=="window": # make the fiducial power spectrum a numerical top hat
            P_fid=np.zeros(self.n_sph_modes)
            P_fid[self.k_idx_for_window]=1.
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
                           P_fid=np.ones(self.n_sph_modes),Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
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
            fi.avg_realizations(interfix="fi")
            self.N_per_realization=fi.N_per_realization
            self.Pfiducial_cyl=fi.P_binned_converged
            self.kperp_for_theory=fi.kperpbins
            self.kpar_for_theory=fi.kparbins
            print("theory + fidu beam +                    ?fg? MC complete")
        if recalc_rt:
            rt.avg_realizations(interfix="rt")
            if not recalc_fi:
                self.N_per_realization=rt.N_per_realization
                self.kperp_for_theory=rt.kperpbins
                self.kpar_for_theory=rt.kparbins
            self.Prealthought_cyl=rt.P_binned_converged
            print("theory + fidu beam + syst + meas errs + ?fg? MC complete")
        if (recalc_sf):
            sf.avg_realizations(interfix="sf")
            if not recalc_fi:
                self.N_per_realization=sf.N_per_realization
                self.kperp_for_theory=sf.kperpbins
                self.kpar_for_theory=sf.kparbins
            self.Pnotheory_cyl=sf.P_binned_converged
            print("         fidu beam + syst + meas errs + ?fg? MC complete")

        _,_,self.Ptheory_cyl=self.unbin_to_Pcyl(self.pars_set_cosmo, kperp_to_use=self.kperp_for_theory, kpar_to_use=self.kpar_for_theory)# unbin_to_Pcyl(self,pars_to_use,kperp_to_use=None,kpar_to_use=None)
        if isolated==False:
            self.Pcont_cyl=self.Pfiducial_cyl-self.Prealthought_cyl

    def cyl_partial(self,n):  
        """        
        cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
        """
        dparn=self.dpar[n]
        pcopy=self.pars_set_cosmo.copy()
        pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

        _,_,Pcyl=self.unbin_to_Pcyl(pcopy)
        P0=np.mean(np.abs(self.Pcyl))+self.eps
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
        if (np.mean(Pcyl_dif)<tol): # consider relaxing this to np.any if it ever seems like too strict a condition?!
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

    def compute_del_P_del_pars(self):
        """
        builds a (N_pars_forecast,Nkpar,Nkperp) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
        """
        for n in range(self.N_pars_set_cosmo):
            self.iter=0 # bc starting a new partial deriv calc.
            self.cyl_partial(n)

    def compute_noise(self):
        assert self.N_per_realization is not None, "try calling the compute_noise() method again after running calc_power_contamination()"
        self.sample_variance=np.sqrt(2/self.N_per_realization)*self.Pfiducial_cyl # rescale according to the number of realizations 

        sen=CHORD_sense(spacing=[self.b_EW,self.b_NS],
                        n_side=[self.N_EW,self.N_NS],
                        orientation=def_offset_deg,
                        center=None,
                        freq_cen=self.nu_ctr*u.MHz,
                        dish_size=D*u.m,
                        Trcv=35*u.K,
                        latitude=DRAO_lat*u.radian,
                        integration_time=integration_s*u.s, # OoM from CHIME
                        time_per_day=hrs_per_night*u.hour, # made up
                        n_days=100, # also made up
                        bandwidth=self.bw*u.MHz,
                        coherent=False,
                        tsky_ref_freq=400.*u.MHz,
                        tsky_amplitude=25*u.K,
                        horizon_buffer=0.1*littleh/u.Mpc,
                        foreground_model="optimistic") # arguments to propagate for maximal flexibility
        sen.sense2d()
        kperp_from_21cmSense=sen.sense2d_kperp
        kpar_from_21cmSense=sen.sense2d_kpar
        thnoise_21cmSense=sen.sense2d_P
        kperp_surv_grid,kpar_surv_grid=np.meshgrid(self.kperp_surv,self.kpar_surv,
                                                   indexing="ij")
        thnoise_surv=RGI((kperp_from_21cmSense,kpar_from_21cmSense),thnoise_21cmSense,
                          bounds_error=False,fill_value=None)(np.array([kperp_surv_grid,kpar_surv_grid]).T).T
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

    def forecast_corner_plot(self,N_Fisher_samples=10000):
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
        print("survey centred at.......................................................................\n    nu ={:>7.4}     MHz \n    z  = {:>9.4} \n    Dc = {:>9.4f}  Mpc\n".format(float(self.nu_ctr),self.z_ctr,self.r0))
        print("survey spans............................................................................\n    nu =  {:>5.4}    -  {:>5.4}    MHz (deltanu = {:>6.4}    MHz) \n    z =  {:>9.4} - {:>9.4}     (deltaz  = {:>9.4}    ) \n    Dc = {:>9.4f} - {:>9.4f} Mpc (deltaDc = {:>9.4f} Mpc)\n".format(self.nu_lo,self.nu_hi,self.bw,self.z_hi,self.z_lo,self.z_hi-self.z_lo,self.Dc_hi,self.Dc_lo,self.Dc_hi-self.Dc_lo))
        if (self.primary_beam_type.lower()!="manual"):
            print("characteristic instrument response widths...............................................\n    beamFWHM0 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_x,self.epsx))
            print("specific to the cylindrically asymmetric beam...........................................\n    beamFWHM1 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_y,self.epsy))
        print("cylindrically binned wavenumbers of the survey..........................................\n    kperp     {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} bins of width {:>8.4} Mpc**(-1))\n    kparallel {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} channels of width {:>7.4}  Mpc**(-1)) \n".format(self.kperpmin_surv,self.kperp_surv[-1],self.Nkperp_surv,self.kperp_surv[-1]-self.kperp_surv[-2],    self.kparmin_surv,self.kpar_surv[-1],self.Nkpar_surv,self.kpar_surv[-1]-self.kpar_surv[-2]))

    def print_results(self):
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(self.pars_forecast):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(self.pars_forecast_names[p], par, self.biases[p], self.biases[p]/par))
        return None
####################################################################################################################################################################################################################################

"""
this class helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. iterate power spec calcs from different box realizations until convergence
4. interpolate a power spectrum (sph, cyl, or sph->grid)
"""

class cosmo_stats(object):
    def __init__(self,
                 Lxy,Lz=None,                                                                       # one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
                 T_pristine=None,T_primary=None,P_fid=None,Nvox=None,Nvoxz=None,                    # need one of either T (pristine or primary) or P to get started; I also check for any conflicts with Nvox
                 primary_beam_num=None,primary_beam_aux_num=None, primary_beam_type_num="Gaussian", # primary beam considerations
                 primary_beam_den=None,primary_beam_aux_den=None, primary_beam_type_den="Gaussian", # systematic-y beam (optional)
                 Nkperp=10,Nkpar=0,binning_mode="lin",bin_each_realization=False,                        # binning considerations for power spec realizations (log mode not fully tested yet b/c not impt. for current pipeline)
                 frac_tol=0.1,                                                                      # max number of realizations
                 kperpbins_interp=None,kparbins_interp=None,                                             # bins where it would be nice to know about P_converged
                 P_converged=None,verbose=False,                                                    # status updates for averaging over realizations
                 k_fid=None,kind="cubic",avoid_extrapolation=False,                                 # helper vars for converting a 1d fid power spec to a box sampling
                 no_monopole=True,seed=None,                                                        # consideration when generating boxes
                 manual_primary_beam_modes=None,                                                    # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 radial_taper=None,image_taper=None,                                                # implement soon: quick way to use an Airy beam in per-antenna mode
                 wedge_cut=False,nu_ctr_for_wedge=None,layer_foregrounds=False,foreground_field=None,fg_modes=None):
        """
        Lxy,Lz                    :: float                       :: side length of cosmo box          :: Mpc
        T_pristine                :: (Nvox,Nvox,Nvox) of floats  :: cosmo box (just physics/no beam)  :: K
        T_primary                 :: (Nvox,Nvox,Nvox) of floats  :: cosmo box * primary beam          :: K
        P_fid                     :: (Nkperp_fid,) of floats        :: sph binned fiducial power spec    :: K^2 Mpc^3
        Nvox,Nvoxz                :: float                       :: cosmobox#vox/side,z-ax can differ :: ---
        primary_beam              :: callable (or, if            :: power beam in Cartesian coords    :: ---
                                     primary_beam_type=="manual" 
                                     a 3D array)          
        primary_beam_aux         :: tuple of floats             :: Gaussian, Airy: μ, σ      :: Gaussian: r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_type         :: str                         :: for now: Gaussian / Airy          :: ---
        Nkperp, Nkpar                  :: int                         :: # power spec bins for axis 0/1    :: ---
        binning_mode              :: str                         :: lin/log spacing                   :: ---
        frac_tol                  :: float                       :: max fractional amount by which    :: ---
                                                                    the p.s. avg can change w/ the 
                                                                    addition of the latest realiz. 
                                                                    and the ensemble average is 
                                                                    considered converged
        kperpbins_interp,            :: (Nkperp_interp,) of floats     :: bins to which to interpolate the  :: 1/Mpc
        kparbins_interp                (Nkpar_interp,) of floats        converged power spec (prob set
                                                                    by survey considerations)
        P_converged               :: same as that of P_fid       :: average of realizations         :: K^2 Mpc^3
        verbose                   :: bool                        :: every 10% of realization_ceil     :: ---
        k_fid                     :: (Nkperp_fid,) of floats        :: modes where P_fid is sampled      :: 1/Mpc
        kind                      :: str                         :: interp type                       :: ---
        avoid_extrapolation       :: bool                        :: when calling scipy interpolators  :: ---
        no_monopole               :: bool                        :: y/n subtr. from generated boxes   :: ---
        manual_primary_beam_modes :: primary_beam.shape of       :: domain of a discrete sampling     :: Mpc
                                     floats (when not callable) 
        """
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
        # print("min, max of self.xy_vec_for_box:",np.min(self.xy_vec_for_box),np.max(self.xy_vec_for_box))
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
        # print("cosmo_stats.__init__: kxy vec max = ",np.max(self.kxy_vec_for_box_corner))
        # print("cosmo_stats.__init__: kz  vec max = ",np.max(self.kz_vec_for_box_corner))
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.kxy_vec_for_box_corner,
                                                                                self.kxy_vec_for_box_corner,
                                                                                self.kz_vec_for_box_corner,
                                                                                indexing="ij")               # box-shaped Cartesian coords
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the generate_box direction)
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
            print("len(manual_primary_beam_modes)=",len(manual_primary_beam_modes))
            print("manual_primary_beam_modes[0].shape=",manual_primary_beam_modes[0].shape)
            print("foreground_field.shape=",foreground_field.shape)
            print("self.xx_grid.shape=",self.xx_grid.shape)
            print("self.yy_grid.shape=",self.yy_grid.shape)
            print("self.zz_grid.shape=",self.zz_grid.shape)
            self.fg_modes=fg_modes
            self.foreground_field=RGI(fg_modes,foreground_field,
                                      bounds_error=False,fill_value=None)(np.array([self.xx_grid,self.yy_grid,self.zz_grid]).T).T# interpolate beam_effects voxelization to cosmo_stats discretization... following the same strategy as beam interpolation

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
            # perpbin_indices_slice_centre[perpbin_indices_slice_centre==Nkperp]=Nkperp-1
            self.perpbin_indices_slice_centre=perpbin_indices_slice_centre
            self.perpbin_indices_slice_1d_centre= np.reshape(self.perpbin_indices_slice_centre,(self.Nvox**2,))        # 1d version of ^ (compatible with np.bincount)
            parbin_indices_column_centre=    np.digitize(self.kpar_column_centre,self.kparbins,right=True)          # cyl kpar bin that each voxel falls into
            # parbin_indices_column_centre[parbin_indices_column_centre==Nkpar]=Nkpar-1
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
                                       bounds_error=False,fill_value=None)(np.array([self.xx_grid,self.yy_grid,self.zz_grid]).T).T
                self.evaled_primary_num=evaled_primary_num
            
            else:
                raise ValueError("not yet implemented")

        self.primary_beam_den=primary_beam_den
        self.primary_beam_aux_den=primary_beam_aux_den
        self.primary_beam_type_den=primary_beam_type_den
        self.manual_primary_beam_modes=manual_primary_beam_modes # _fi and _rt assumed to be sampled at the same modes, if this is the case
        if (self.primary_beam_den is not None): # non-identity PERTURBED primary beam
            if (self.primary_beam_type_den=="Gaussian" or self.primary_beam_type_den=="Airy"):
                self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_aux_den
                evaled_primary_den=  self.primary_beam_den(self.xx_grid,self.yy_grid,self.fwhm_x,  self.fwhm_y,  self.r0)                
            elif (self.primary_beam_type_den=="manual"):
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable... so, no danger of attribute errors
                    primary_beam_den.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ValueError("conflicting info") 
                if self.manual_primary_beam_modes is None:
                    raise ValueError("not enough info")

                evaled_primary_den=RGI(manual_primary_beam_modes,self.primary_beam_den,
                                       bounds_error=False,fill_value=None)(np.array([self.xx_grid,self.yy_grid,self.zz_grid]).T).T
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
        self.verbose=verbose

        # P_converged interpolation bins
        self.kperpbins_interp=kperpbins_interp
        self.kparbins_interp=kparbins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        self.P_unbinned_running_sum=np.zeros((self.Nvox,self.Nvox,self.Nvoxz))
        if (P_converged is not None):          # maybe you have a converged power spec average from a previous calc and just want to interpolate or generate more box realizations?
            self.P_converged=P_converged
        else:
            self.P_converged=None
        self.P_interp=None
        self.not_converged=None
        self.N_cumul=np.zeros((self.Nkperp,self.Nkpar))

    def calc_bins(self,Nki,Nvox_to_use,kmin_to_use,kmax_to_use):
        """
        generate a set of bins spaced according to the desired scheme with max and min
        """
        if (self.binning_mode=="log"):
            kbins=np.logspace(np.log10(kmin_to_use),np.log10(kmax_to_use),num=Nki)
            limiting_spacing=twopi*(10.**(kmax_to_use)-10.**(kmax_to_use-(np.log10(Nvox_to_use)/Nki))) 
        elif (self.binning_mode=="lin"):
            kbins=np.linspace(kmin_to_use,kmax_to_use,Nki)
            limiting_spacing=twopi*(0.5*Nvox_to_use-1)/(Nki) # version for a kmax that is "aware that" there are +/- k-coordinates in the box
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
        interpolator=RGI((k_fid_unique,),Pfid_unique,
                          bounds_error=False,fill_value=None)
        P_fid_flattened_box[sort_array]=interpolator(kmag_grid_corner_flat_sorted[:,None])
        self.P_fid_box=np.reshape(P_fid_flattened_box,(self.Nvox,self.Nvox,self.Nvoxz))
            
    def generate_P(self,send_to_P_fid=False,T_use=None):
        """
        philosophy: 
        * compute the power spectrum of a known cosmological box 
        * defer binning to another function (keyword to control whether or not this is activated)
        * add to running sum of realizations
        """
        if (T_use is None or T_use=="primary"):
            T_use=self.T_primary
        else:
            T_use=self.T_pristine
        if (self.T_primary is None):    # power spec has to come from a box
            self.generate_box() # populates/overwrites self.T_pristine and self.T_primary
        
        T_tilde=fftshift(fftn((ifftshift(T_use*self.taper_xyz)*self.d3r)))
        modsq_T_tilde=(T_tilde*np.conjugate(T_tilde)).real
        P_unbinned=modsq_T_tilde/self.effective_volume # box-shaped, but calculated according to the power spectrum estimator equation
        self.P_unbinned=P_unbinned
        if self.bin_each_realization:
            self.bin_power()
        
        if send_to_P_fid: # if generate_P was called speficially to have a spec from which all future box realizations will be generated
            if self.bin_each_realization:
                self.P_fid=self.P_binned
            else:
                self.P_fid=self.P_unbinned
        else:             # the "normal" case where you're just accumulating a realization (any binning at the end)
            self.P_unbinned_running_sum+=P_unbinned

    def bin_power(self,power_to_bin=None):
        if power_to_bin is None:
            power_to_bin=self.unbinned_power
        # print("range of column par bin indices:",np.min(self.parbin_indices_column_centre),np.max(self.parbin_indices_column_centre))
        if (self.Nkpar==0):   # bin to sph
            unbinned_power_1d= np.reshape(power_to_bin,    (self.Nvox**2*self.Nvoxz,))

            sum_unbinned_power= np.bincount(self.sph_bin_indices_1d_centre, 
                                           weights=unbinned_power_1d, 
                                           minlength=self.Nkperp)       # for the ensemble avg: sum    of unbinned_power values in each bin
            N_unbinned_power=   np.bincount(self.sph_bin_indices_1d_centre,
                                           minlength=self.Nkperp)       # for the ensemble avg: number of unbinned_power values in each bin
            sum_unbinned_power_truncated=sum_unbinned_power[:-1]       # excise sneaky corner modes: I devised my binning to only tell me about voxels w/ k<=(the largest sphere fully enclosed by the box), and my bin edges are floors. But, the highest floor corresponds to the point of intersection of the box and this largest sphere. To stick to my self-imposed "the stats are not good enough in the corners" philosophy, I must explicitly set aside the voxels that fall into the "catchall" uppermost bin. 
            N_unbinned_power_truncated=  N_unbinned_power[:-1]         # idem ^
            final_shape=(self.Nkperp,)
        elif (self.Nkpar!=0): # bin to cyl
            sum_unbinned_power= np.zeros((self.Nkperp+1,self.Nkpar)) # for the ensemble avg: sum    of unbinned_power values in each bin  ...upon each access, update the kparBIN row of interest, but all Nkperp columns
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

                # print("shape of sum_unbinned_power; current_par_bin:",sum_unbinned_power.shape,current_par_bin)
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
            T_tilde[self.voxels_in_wedge_corner]=0
        T=fftshift(irfftn(T_tilde*self.d3k,
                          s=(self.Nvox,self.Nvox,self.Nvoxz),
                          axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: fftshiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
        if self.layer_foregrounds:
            T+=self.foreground_field
        if self.no_monopole:
            T-=np.mean(T) # subtract monopole moment
        
        self.T_pristine=T
        self.T_primary=T*self.evaled_primary_num

    def avg_realizations(self,interfix=""):
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
            if self.verbose:
                if (i%(self.N_realizations//10)==0):
                    print("realization",i)
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

    def interpolate_P(self,use_P_fid=False):
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
                self.avg_realizations()
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
            modes_defined_at=(self.kperpbins,)
            modes_to_eval_at=(self.kperpbins_interp,)
        P_interpolator=RGI(modes_defined_at,self.P_converged,
                           bounds_error=self.avoid_extrapolation,fill_value=None)
        P_interp=P_interpolator(modes_to_eval_at)
        if self.kparbins_interp is not None:
            P_interp=P_interp.T # anticipate the RGI behaviour
        self.P_interp=P_interp
####################################################################################################################################################################################################################################

"""
this class helps compute numerical windowing boxes for brightness temp boxes
resulting from primary beams that have the flexibility to differ on a per-
antenna basis. (beam chromaticity built in).
"""

class per_antenna(beam_effects):
    def __init__(self,
                 mode="full",b_NS=b_NS,b_EW=b_EW,observing_dec=def_observing_dec,offset_deg=def_offset_deg,
                 N_fiducial_beam_types=N_fid_beam_types,N_pert_types=0,N_pbws_pert=0,pbw_pert_frac=def_pbw_pert_frac,
                 N_timesteps=hrs_per_night*3600//integration_s,nu_ctr=nu_HI_z0,
                 pbw_fidu=None,N_grid_pix=def_PA_N_grid_pix,Delta_nu=CHORD_channel_width_MHz,
                 distribution="random",fidu_types_prefactors=None,
                 outname=None,per_channel_systematic=None,per_chan_syst_facs=None,
                 evol_restriction_threshold=def_evol_restriction_threshold
                 ):
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
        self.nu_ctr_Hz=nu_ctr*1e6
        self.Dc_ctr=comoving_distance(nu_HI_z0/nu_ctr-1)
        self.N_hrs=synthesized_beam_crossing_time(self.nu_ctr_Hz,bmax=bmax,dec=observing_dec) # freq needs to be in Hz
        self.lambda_obs=c/self.nu_ctr_Hz
        if (pbw_fidu is None):
            pbw_fidu=self.lambda_obs/D
            pbw_fidu=[pbw_fidu,pbw_fidu]
        self.pbw_fidu=np.array(pbw_fidu) # NEEDS TO BE UNPACKABLE AS X,Y ... but pointless to re-cast to np array here bc I've already done so in the calling routine
        
        # antenna positions xyz
        antennas_EN=np.zeros((N_ant,2))
        for i in range(N_NS):
            for j in range(N_EW):
                antennas_EN[i*N_EW+j,:]=[j*b_EW,i*b_NS]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        offset=offset_deg*pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset.
        offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # use this rotation matrix to adjust the NS/EW-only coords
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

        N_beam_types=(self.N_pert_types+1)*self.N_fiducial_beam_types 

        # array layout, organized and indexed by fiducial beam type
        if fidu_types_prefactors is None:
            fidu_types_prefactors=np.ones(N_fiducial_beam_types)
        self.fidu_types_prefactors=fidu_types_prefactors
        pbw_fidu_types=np.zeros((N_ant,))
        if self.distribution=="random":
            pbw_fidu_types=np.random.randint(0,self.N_fiducial_beam_types,size=(N_ant,))
            np.savetxt("pbw_fidu_types.txt",pbw_fidu_types)
        elif self.distribution=="corner":
            if self.N_fiducial_beam_types!=4:
                raise ValueError("conflicting info") # in order to use corner mode, you need four fiducial beam types
            pbw_fidu_types=np.zeros((N_NS,N_EW))
            half_NS=N_NS//2
            half_EW=N_EW//2
            pbw_fidu_types[:half_NS,half_EW:]=1
            pbw_fidu_types[half_NS:,:half_EW]=2
            pbw_fidu_types[half_NS:,half_EW:]=3 # the quarter of the array with no explicit overwriting keeps its idx=0 (as necessary)
            pbw_fidu_types=np.reshape(pbw_fidu_types,(N_ant,))
        elif self.distribution=="diagonal":
            raise ValueError("not yet implemented")
        elif self.distribution=="rowcol":
            pbw_fidu_types=np.zeros((N_NS,N_EW))
            for i in range(1,self.N_fiducial_beam_types):
                pbw_fidu_types[:,i::self.N_fiducial_beam_types]=i
            pbw_fidu_types=np.reshape(pbw_fidu_types,(N_ant,))
        elif self.distribution=="ring":
            raise ValueError("not yet implemented")
        else:
            raise ValueError("not yet implemented")
        
        # seed the systematics (still doing this randomly throughout the array)
        pbw_pert_types=np.zeros((N_ant,))
        epsilons=np.zeros(N_pert_types+1)
        if (self.N_pbws_pert>0):
            if (self.N_pert_types>1):
                epsilons[1:]=self.pbw_pert_frac*np.random.uniform(size=np.insert(self.N_pert_types,0,1))
            else: 
                epsilons=self.pbw_pert_frac
            indices_of_ants_w_pert_pbws=np.random.randint(0,N_ant,size=self.N_pbws_pert) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
            pbw_pert_types[indices_of_ants_w_pert_pbws]=np.random.randint(1,high=(self.N_pert_types+1),size=np.insert(self.N_pbws_pert,0,1)) # leaves as zero the indices associated with unperturbed antennas
            np.savetxt("pbw_pert_types.txt",pbw_pert_types)
        else:
            indices_of_ants_w_pert_pbws=None
        self.indices_of_ants_w_pert_pbws=indices_of_ants_w_pert_pbws
        self.epsilons=epsilons
        self.per_chan_syst_facs=per_chan_syst_facs
        
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
        
        # enough nonredundant symbols and colours available for <~O(10) classes (each) of perturbation and fiducial beam 
        if (outname is not None and self.N_pert_types>1): # only useful to plot if different antennas have different perturbations
            print("perturbed-beam per-antenna computation underway. plotting...")
            fig=plt.figure(figsize=(12,8))
            for i in range(N_pert_types+1):
                for j in range(N_fiducial_beam_types):
                    keep=np.nonzero(np.logical_and(pbw_pert_types==i, pbw_fidu_types==j))
                    plt.scatter(antennas_xyz[keep,0],antennas_xyz[keep,1],c="C"+str(j),marker=symbols[i],label=str(j)+str(i),lw=0.3,s=50) # change j and i to permute
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("CHORD "+str(self.nu_ctr_MHz)+" MHz pointing dec="+str(round(observing_dec,5))+" rad \n"
                      "projected antenna positions by primary beam status\n"
                      "[antenna fiducial status][antenna perturbation status]=")
            fig.legend(loc="outside right upper")
            plt.savefig("layout_"+outname+".png",dpi=dpi_to_use)
        
            ant_a_pert_type,ant_b_pert_type=indices_of_constituent_ant_pb_pert_types.T
            ant_a_fidu_type,ant_b_fidu_type=indices_of_constituent_ant_pb_fidu_types.T
            Nrow=9 # make this less hard-coded
            Ncol=np.max([int(np.ceil(N_beam_types**2/Nrow)),2])
            fig,axs=plt.subplots(Nrow,Ncol,figsize=(N_beam_types*2.25,N_beam_types*2.25))
            num=0
            u_inst=uvw_inst[:,0]
            v_inst=uvw_inst[:,1]
            for i in range(self.N_pert_types+1):
                for j in range(self.N_pert_types+1):
                    pert_class_condition=np.logical_and(ant_a_pert_type==i, ant_b_pert_type==j)
                    for k in range(self.N_fiducial_beam_types):
                        for l in range(self.N_fiducial_beam_types):
                            fidu_class_condition=np.logical_and(ant_a_fidu_type==k, ant_b_fidu_type==l)
                            current_row=num//Ncol
                            current_col=num%Ncol

                            keep=np.nonzero(np.logical_and(pert_class_condition,fidu_class_condition))
                            u_inst_ab=u_inst[keep]
                            v_inst_ab=v_inst[keep]
                            axs[current_row,current_col].scatter(u_inst_ab,v_inst_ab,edgecolors="k",lw=0.15,s=4)
                            axs[current_row,current_col].set_xlabel("u (λ)")
                            axs[current_row,current_col].set_ylabel("v (λ)")
                            axs[current_row,current_col].set_title(str(i)+str(j)+str(k)+str(l))
                            axs[current_row,current_col].axis("equal")                
                            num+=1
            plt.suptitle("CHORD "+str(self.nu_ctr_MHz)+" MHz instantaneous uv coverage; antenna status [Apert][Bpert][Afidu][Bfidu]=")
            plt.tight_layout()
            plt.savefig("inst_uv_"+outname+".png",dpi=dpi_to_use)

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*self.N_hrs/12
        hour_angles=np.linspace(0,hour_angle_ceiling,self.N_timesteps)
        thetas=hour_angles*15*np.pi/180
        
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

        # prep for beam chromaticity calcs (out here so it's easier to hand info off to beam_effects even if I don't recalc the box or redo the windowing)
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/self.Delta_nu)
        self.N_chan=N_chan
        nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(nu_hi,nu_lo,N_chan) # decr.
        surv_channels_Hz=1e6*surv_channels_MHz
        surv_wavelengths=c/surv_channels_Hz # incr.
        self.surv_wavelengths=surv_wavelengths
        z_channels=nu_HI_z0/surv_channels_MHz-1.
        self.comoving_distances_channels=np.asarray([comoving_distance(chan) for chan in z_channels]) # incr.
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]
        surv_beam_widths=dif_lim_prefac*surv_wavelengths/D # incr.
        self.surv_beam_widths=surv_beam_widths
        plt.figure()
        plt.plot(surv_channels_MHz,surv_beam_widths,label="diffraction-limited Airy FWHM")    
        per_chan_syst_name="None"        
        if self.per_channel_systematic=="D3A_like":
            surv_beam_widths=(surv_beam_widths)**1.2 # keep things dimensionless, but use a steeper decay
            noise_bound_lo=0.9
            noise_bound_hi=1.1
            noise_frac=(noise_bound_hi-noise_bound_lo)*np.random.random_sample(size=(N_chan,))+noise_bound_lo # random_sample draws fall within [0,1) but I want values between [0.75,1.25)*(that channel's beam width)
            surv_beam_widths*=noise_frac
            per_chan_syst_name="D3A_like"
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
        self.per_chan_syst_name=per_chan_syst_name
        self.surv_channels_MHz=surv_channels_MHz
        self.surv_beam_widths=surv_beam_widths

    def calc_dirty_image(self, Npix=1024, pbw_fidu_use=None,tol=img_bin_tol):
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

    def stack_to_box(self, tol=img_bin_tol):
        if (self.nu_ctr_MHz<(350/(1-self.evol_restriction_threshold/2)) or 
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
    def __init__(self,freq_lo,freq_hi,delta_nu_CST,
                 beam_sim_directory=None,f_head="farfield_(f=",f_mid1=")_[1]",f_mid2=")_[2]",f_tail="_efield.txt",
                 box_outname="placeholder",mode="pathfinder",Nxy=128):
        self.beam_sim_directory=beam_sim_directory
        self.f_head=f_head
        self.f_mid1=f_mid1
        self.f_mid2=f_mid2
        self.f_tail=f_tail
        self.box_outname=box_outname
        freqs=np.arange(freq_lo,freq_hi,delta_nu_CST) # GHz
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
        k_perp=kperp(nu_ctr,N_bl,b_EW,bmax)
        L_xy=twopi/k_perp[0]
        xy_for_box=L_xy*fftshift(fftfreq(Nxy))
        self.xy_for_box=xy_for_box
        np.save("xy_vec_for_box"+box_outname,xy_for_box)
        self.Nxy=Nxy
        self.xx_grid,self.yy_grid=np.meshgrid(self.xy_for_unwrapping,self.xy_for_unwrapping,
                                              indexing="ij") # config space points of interest for the slice (guided by the transverse extent of the eventual config-space box)
        freq_names=np.zeros(Nfreqs,dtype=str) # store the GHz CST frequencies as strings of the format that Aditya's sims use
        for i,freq in enumerate(self.freqs):
            freq_name=str(np.round(freq,4)) # round to four decimal places and convert to string
            freq_names[i]=freq_name.rstrip("0") # strip trailing zeros
        self.freq_names=freq_names

    def translate_sim_beam_slice(self,CST_filename,i=0):
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
        spacing=[6.3,8.5],
        n_side=[22,24],
        orientation=None,
        center=[0,0],
        
        freq_cen = 900*u.MHz,
        dish_size = 6*u.m,
        Trcv = 30*u.K,
        latitude = (49.3*np.pi/180.0)*u.radian,
        integration_time= 10*u.s, 
        time_per_day = 6*u.hour,  # time observing per day
        n_days = 100 ,    # num days in obs
        bandwidth=80*u.MHz,
        coherent = False, # add baselines coherently if they are not instantaneously redundant?
        tsky_ref_freq = 400.0 * u.MHz, 
        tsky_amplitude = 25 *u.K,
        
        horizon_buffer = 0.1 * littleh/ u.Mpc,
        foreground_model = 'optimistic',

        sv=False, # sample variance
        tn=True  # thermal noise
    ):
        bl_max=np.sqrt((spacing[0]*n_side[0])**2+(spacing[1]*n_side[1])**2)*u.m
        bl_min=np.min(spacing)*u.m
        self.spacing = spacing
        self.n_side = n_side
        self.orientation = orientation
        self.center = center
        self.freq_cen = freq_cen
        self.dish_size = dish_size
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
                                              dish_size=self.dish_size),
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
        
    def rectangle_generator(self):

        """
        ------------------------------------------------------------------------
        Generate a grid of baseline locations filling a rectangular array for CHORD/HIRAX. 
    
        Inputs:
            spacing      [2-element list or numpy array] positive integers specifying
                 the spacing between antennas. Must be specified, no default.
            n_side       [2-element list or numpy array] positive integers specifying
                 the number of antennas on each side of the rectangular array.
                 Atleast one value should be specified, no default.
            orientation  [scalar] counter-clockwise angle (in degrees) by which the
                 principal axis of the rectangular array is to be rotated.
                 Default = None (means 0 degrees)
            center       [2-element list or numpy array] specifies the center of the
                 array. Must be in the same units as spacing. The rectangular
                 array will be centered on this position.
        Outputs:
            Two element tuple with these elements in the following order:
            xy           [2-column array] x- and y-locations. x is in the first
                 column, y is in the second column. Number of xy-locations
                 is equal to the number of rows which is equal to n_total
            id           [numpy array of string] unique antenna identifier. Numbers
                 from 0 to n_antennas-1 in string format.
                 Notes:
        ------------------------------------------------------------------------
        """

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
            angle = np.radians(self.orientation)
            rot_matrix = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
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

def memo_ii_plotter(ensemble_of_spectra, ensemble_ids, colourmap, k_perp, k_par, case_title, k1_inset=0.06, k2_inset=2.5, qty_to_plot="P"):
    N_spectra=len(ensemble_of_spectra)
    assert(N_spectra==len(ensemble_ids)), "mismatched number of spectra and spectrum names"
    N_LHS_rows=int(np.ceil(np.sqrt(N_spectra)))
    N_LHS_cols=int(np.ceil(N_spectra/N_LHS_rows))
    cyl_extent=[k_perp[0],k_perp[-1],k_par[0],k_par[-1]]
    k_perp_grid,k_par_grid=np.meshgrid(k_perp,k_par)
    k_mag_grid=np.sqrt(k_perp_grid**2+k_par_grid**2)
    values_of_k1=np.zeros(N_spectra)
    values_of_k2=np.zeros(N_spectra)

    fig = plt.figure(figsize=(14, 8),layout="constrained")
    gs = gridspec.GridSpec(N_LHS_rows, N_LHS_cols+1, figure=fig)
    axs = [[fig.add_subplot(gs[row, col]) for col in range(4)] for row in range(4)] # grid for the left
    ax_right = fig.add_subplot(gs[:, 4]) # summary holder on the right

    for k,spec in enumerate(ensemble_of_spectra):
        i=k//N_spectra
        j=k%N_spectra
        if qty_to_plot=="Delta2":
            spec_to_plot=spec*k_mag_grid**3/(2*pi**2)
        elif qty_to_plot=="P":
            spec_to_plot=np.copy(spec)
        else:
            raise ValueError("P and Delta2 are the only pre-established plotting options for now")

        im=axs[i,j].imshow(spec_to_plot.T, cmap=colourmap, origin="lower", extent=cyl_extent)
        axs[i,j].set_xlabel("k$_{||}$")
        axs[i,j].set_ylabel("k_\perp")
        axs[i,j].set_title(ensemble_ids[k])
        axs[i,j].set_aspect("equal")
        plt.colorbar(im,ax=axs[i,j],shrink=0.3,extend="both")

        idx_for_k1=np.nonzero(np.min(np.abs(k_mag_grid-k1_inset)))
        idx_for_k2=np.nonzero(np.min(np.abs(k_mag_grid-k2_inset)))
        values_of_k1[k]=spec[idx_for_k1]
        values_of_k2[k]=spec[idx_for_k2]

    ax_right.scatter(values_of_k1,label="inset for k closest to "+str(np.round(k1_inset,4)))
    ax_right.scatter(values_of_k2,label="inset for k closest to "+str(np.round(k2_inset,4)))
    ax_right.set_xticks(np.arange(N_spectra), labels=ensemble_ids, rotation=40)
    ax_right.legend()

    plt.savefig(case_title+".png")

def save_args_to_file(frame, filepath="settings.json"):
    args, _, _, values = inspect.getargvalues(frame)
    settings = {arg: values[arg] for arg in args}
    with open(filepath, "w") as f:
        json.dump(settings, f, indent=2, default=str)

def power_comparison_plots(redo_window_calc=False, redo_box_calc=False,
              mode="pathfinder", nu_ctr=800, epsxy=0.1,
              frac_tol_conv=0.1, N_sph=256,categ="PA", # categ is manual/PA/CST, beam_type is either Gaussian (for PA) or manual (for CST)
              N_fidu_types=1, N_pert_types=0, 
              N_pbws_pert=0, per_channel_systematic=None,
              PA_dist="random", f_types_prefacs=None, plot_qty="P",
              Nkpar_box=None,Nkperp_box=None, 
                  
              wedge_cut=False, layer_foregrounds=False,
                  
              freq_bin_width=0.1953125, # kHz

              CST_lo=None,CST_hi=None,CST_deltanu=None,
              beam_sim_directory=None,f_mid1=")_[1]",f_mid2=")_[2]",f_tail="_efield.txt",
              CST_f_head_fidu="farfield_(f=",CST_f_head_real="farfield_(f=",CST_f_head_thgt="farfield_(f=",
              
              from_incomplete_MC=False,
              contaminant_or_window=None, k_idx_for_window=0,
              isolated=False,seed=None,
              per_chan_syst_facs=[]): # the default chromaticity systematic
    save_args_to_file(inspect.currentframe())

    ############################## other survey management factors ########################################################################################################################
    nu_ctr_Hz=nu_ctr*1e6
    wl_ctr_m=c/nu_ctr_Hz

    ############################## baselines and beams ########################################################################################################################
    b_NS_CHORD=8.5 # m
    N_NS_CHORD=24
    b_EW_CHORD=6.3 # m
    N_EW_CHORD=22
    bminCHORD=np.min([b_NS_CHORD,b_EW_CHORD])

    if (mode=="pathfinder"): # 10x7=70 antennas (64 w/ receiver hut gaps), 123 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*10)**2+(b_EW_CHORD*7)**2) # pathfinder (as per the CHORD-all telecon on May 26th, but without holes)
        N_ant=64
    elif mode=="full": # 24x22=528 antennas (512 w/ receiver hut gaps), 1010 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2)
        N_ant=512
    else:
        raise ValueError("unknown array mode (not pathfinder or full)")

    if categ=="PA":
        print("PA mode currently only supports a Gaussian beam")
    hpbw_x= 1.029*wl_ctr_m/D*pi/180. # rad; lambda/D estimate
    hpbw_y= 0.75*hpbw_x # simulations show this is characteristic of the UWB feeds

    ############################## pipeline administration ########################################################################################################################
    if contaminant_or_window is not None:
        c_or_w="wind"
    else:
        c_or_w="cont"
    per_chan_syst_string="none"
    per_chan_syst_name=""
    if per_channel_systematic=="D3A_like":
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
    elif PA_dist=="rowcol":
        PA_dist_string="rwcl"
    elif PA_dist!="random":
        raise ValueError("unknown PA_dist")

    # setup for the new regime 
    if type(N_fidu_types)==int:
        N_fidu_types=[N_fidu_types]
        N_pert_types=[N_pert_types]

    complexity_types=np.union1d(N_fidu_types,N_pert_types)
    complexity_cases=list(permutations(complexity_types,2))
    complexity_ids=[str(case) for case in complexity_cases]
    power_quantities_all=[]
    for i,complexity_type in enumerate(complexity_cases):
        N_fidu_types_i,N_pert_types_i=complexity_type
        f_types_prefacs_i=f_types_prefacs[i]
        ioname=mode+"_"+c_or_w+"_"+categ+"_"\
           ""+per_chan_syst_string+"_"+per_chan_syst_name+"_"\
           ""+str(int(nu_ctr))+"MHz__"\
           "cosmicvar_"+str(round(frac_tol_conv,2))+"__"\
           "Nreal_"+str(N_fidu_types_i)+"__"\
           "Npert_"+str(N_pert_types_i)+"_"+str(N_pbws_pert)+"__"\
           "dist_"+PA_dist_string+"__"\
           "epsxy_"+str(epsxy)+"__"\
           "realprefacs_"+str(f_types_prefacs)+"__"\
           "layer_"+str(layer_foregrounds)+"__"\
           "wedge_"+str(wedge_cut)+"__"\
           "seed_"+str(seed)
        
        print("complexity case",complexity_ids[i])
        if (N_fidu_types_i!=4 and PA_dist=="corner"):
            continue

        # PIPELINE ADMIN FOR THIS PA SYSTEMATIC PERMUTATION
        bundled_non_manual_primary_aux=np.array([hpbw_x,hpbw_y])
        bundled_non_manual_primary_uncs=np.array([epsxy,epsxy])
        if categ=="PA":
            windowed_survey=beam_effects(# SCIENCE
                                            # the observation
                                            bminCHORD,bmaxCHORD,                                                             # extreme baselines of the array
                                            nu_ctr,freq_bin_width,                                                       # for the survey of interest
                                            evol_restriction_threshold=def_evol_restriction_threshold,             # how close to coeval is close enough?
                                                
                                            # beam generalities
                                            primary_beam_categ=categ,primary_beam_type="Gaussian",                 # modelling choices
                                            primary_beam_aux=bundled_non_manual_primary_aux,
                                            primary_beam_uncs=bundled_non_manual_primary_uncs,                          # helper arguments
                                            manual_primary_beam_modes=None,                                        # config space pts at which a pre–discretely sampled primary beam is known

                                            # additional considerations for per-antenna systematics
                                            PA_N_pert_types=N_pert_types_i,PA_N_pbws_pert=N_pbws_pert,PA_N_fidu_types=N_fidu_types_i,
                                            PA_fidu_types_prefactors=f_types_prefacs_i,PA_ioname=ioname,PA_distribution=PA_dist,mode=mode,
                                            per_channel_systematic=per_channel_systematic,per_chan_syst_facs=per_chan_syst_facs,

                                            # FORECASTING
                                            P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,
                                            wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds,

                                            # NUMERICAL 
                                            n_sph_modes=N_sph,                                            # conditioning the CAMB/etc. call
                                            init_and_box_tol=0.05,CAMB_tol=0.05,                                   # considerations for k-modes at different steps
                                            Nkpar_box=Nkpar_box,Nkperp_box=Nkperp_box,frac_tol_conv=frac_tol_conv,                          # considerations for cyl binned power spectra from boxes
                                            seed=seed,                                            # enforce zero-mean in realization boxes?
                                            ftol_deriv=1e-16,maxiter=5,                                            # subtract off monopole moment to give zero-mean box?
                                            PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,            # pixels per side of gridded uv plane, uv binning chunk snapshot tightness
                                            radial_taper=kaiser,image_taper=None,

                                            # CONVENIENCE
                                            heavy_beam_recalc=redo_box_calc                                                        # save time by not repeating per-antenna calculations? 
                                            
                                            )

        elif categ=="CST":
            windowed_survey=beam_effects(# SCIENCE
                                        # the observation
                                        bminCHORD,bmaxCHORD,                                                             # extreme baselines of the array
                                        nu_ctr,freq_bin_width,                                                       # for the survey of interest
                                        evol_restriction_threshold=def_evol_restriction_threshold,             # how close to coeval is close enough?
                                            
                                        # beam generalities
                                        primary_beam_categ=categ,primary_beam_type="Gaussian",                 # modelling choices
                                        primary_beam_aux=bundled_non_manual_primary_aux,
                                        primary_beam_uncs=bundled_non_manual_primary_uncs,                          # helper arguments
                                        manual_primary_beam_modes=None,                                       # config space pts at which a pre–discretely sampled primary beam is known

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
                                        wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds,

                                        # NUMERICAL 
                                        n_sph_modes=N_sph,                                             # conditioning the CAMB/etc. call
                                        init_and_box_tol=0.05,CAMB_tol=0.05,                                   # considerations for k-modes at different steps
                                        Nkpar_box=Nkpar_box,Nkperp_box=Nkperp_box,frac_tol_conv=frac_tol_conv,                          # considerations for cyl binned power spectra from boxes
                                        seed=seed,                                            # enforce zero-mean in realization boxes?
                                        ftol_deriv=1e-16,maxiter=5,                                            # subtract off monopole moment to give zero-mean box?
                                        radial_taper=kaiser,image_taper=None,

                                        # CONVENIENCE
                                        heavy_beam_recalc=redo_box_calc                                                        # save time by not repeating per-antenna calculations? 
                                        
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
                np.save("kpar_internal_"+ioname+".npy",kpar_internal)
                np.save("kperp_internal_"+ioname+".npy",kperp_internal)
                if isolated is not False: # break early if you just calculate one windowed power spectrum at a time
                    return None
            else:
                Prealthought=np.load("Prealthought_"+ioname+".npy")
                Pfiducial=np.load("Pfiducial_cyl_"+ioname+".npy")
                Pnotheory=np.load("Pnotheory_"+ioname+".npy")
                Ptheory=np.load("Ptheory_"+ioname+".npy")
                N_per_realization=np.load("N_per_realization_"+ioname+".npy")
                kpar_internal=np.load("kpar_internal_"+ioname+".npy")
                kperp_internal=np.load("kperp_internal_"+ioname+".npy")
        else:
            Prealthought=np.load("P_rt_unconverged.npy")
            Pfiducial=np.load("P_fi_unconverged.npy")
            Pnotheory=np.load("P_sf_unconverged.npy")
            N_per_realization=np.load("N_per_realization_"+ioname+".npy")
            kpar_internal=np.load("kpar_internal_"+ioname+".npy")
            kperp_internal=np.load("kperp_internal_"+ioname+".npy")

        print("Prealthought.shape=",Prealthought.shape)
        print("Pfiducial.shape=",Pfiducial.shape)
        print("Ptheory.shape=",Ptheory.shape)
        print("Pnotheory.shape=",Pnotheory.shape)
        Presidual= Prealthought-Pfiducial
        Pratio=    Pnotheory/Ptheory

        power_quantities_this_complexity=np.array([Pnotheory, Pfiducial, Prealthought, Presidual, Pratio]) # 5 x Nkperp x Nkpar
        power_quantities_all.append(power_quantities_this_complexity) # N_complexity_cases x 5 x Nkperp x Nkpar
        print("handled complexity",complexity_ids[i])

    power_quantities_all=np.asarray(power_quantities_all)
    N_plots=5 # hard-coded for this generation of plots where I can look at the same feasibility analysis for different systematics families
    print("power_quantities_all.shape=",power_quantities_all.shape," EXPECTED shape (",len(power_quantities_all),",N_plots,",len(kperp_internal),",",len(kpar_internal),")") # 
    abs_map=cmasher.cosmic # also consider eclipse, amber, dusk, rainforest, fall, ...others
    rel_map=cmasher.prinsenvlag # also consider viola, ...others
    plot_version_names = ["fidu beam + syst + meas errs + fg", "theory + fidu beam + fg", "theory + fidu beam + syst + meas errs + fg", 
                          "(theory + fidu beam + syst + meas errs + fg) \n- (theory + fidu beam + fg)", "(fidu beam + syst + meas err + fg) \n/ theory"]
    plot_cmaps=    [abs_map, abs_map, abs_map,
                    rel_map, rel_map]
    

    for i in range(N_plots): # iterate over plot cases
        power_quantity_this_plot_case=power_quantities_all[:,i,:,:] # [:,i,:,:] = all complexity cases, ith power spectrum quantity, all kperps, all kpars
        memo_ii_plotter(power_quantity_this_plot_case, complexity_ids, plot_cmaps[i], 
                        kperp_internal, kpar_internal, plot_version_names[i], 
                        plot_qty=plot_qty) # memo_ii_plotter(ensemble_of_spectra, ensemble_ids, plot_cmaps, k_perp, k_par, case_title