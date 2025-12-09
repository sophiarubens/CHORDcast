import numpy as np
import camb
from camb import model
from scipy.signal import convolve
from scipy.signal.windows import kaiser
from scipy.interpolate import interpn,interp1d,RegularGridInterpolator
from scipy.special import j1
from numpy.fft import fftshift,ifftshift,fftn,irfftn,fftfreq,ifft2
from cosmo_distances import *
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
import time

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
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps
parnames_Planck18=                ['H_0',       'Omega_b h**2',      'Omega_c h**2',      '10**9 * A_S',        'n_s'       ]
scale=1e-9
dpar_default=1e-3*np.ones(len(pars_Planck18))
dpar_default[3]*=scale

# physical
nu_HI_z0=1420.405751768 # MHz
c=2.998e8

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
# nearly_zero=(1./maxfloat)**2
nearly_zero=1e-30
symbols=["o", # circle
         "*", # star
         "v", # equilateral triangle (vertex down)
         "s", # square (edge up)
         "H", # hexagon (edge at top)
        "d", # diamond
         "1", # thirds-division, point down
         "8", # octagon
         "p", # pentagon
         "P", # filled plus
         "h", # hexagon (vertex at top)
         "+", # fine plus
         "X", # filled x
         "D", # square (vertex up)
         ".", # point
         "^", # equilateral triangle (vertex up)
         "<", # equilateral triangle (vertex left)
         "x", # fine x
         ">", # equilaterial triangle (vertex right)
         "2", # thirds-division, point up
         "3", # thirds-division, point left
         "_", # horizontal line
         "4"  # thirds-division, point right
         ]

# numerical
scale=1e-9
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # preposterous number of sig figs from Mathematica (past the double-precision threshold or whatever)
eps=1e-15

# CHORD
N_NS_full=24
N_EW_full=22
b_NS=8.5
b_EW=6.3
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6. # m
def_observing_dec=pi/60.
def_offset_deg=1.75*pi/180. # for this placeholder state where I build up the CHORD layout using rotation matrices instead of actual measurements (does Richard know more? see if he gets back to me from when I asked on 30/10/25)
def_pbw_pert_frac=1e-2
def_N_timesteps=15
def_evol_restriction_threshold=1./15.
img_bin_tol=1.75
def_PA_N_timesteps=15
def_PA_N_grid_pix=256
N_fid_beam_types=1

# warnings 
class NotYetImplementedError(Exception):
    pass
class NumericalDeltaError(Exception):
    pass
class ResolutionError(Exception):
    pass
class UnsupportedBinningMode(Exception):
    pass
class NotEnoughInfoError(Exception):
    pass
class PathologicalError(Exception):
    pass
class ConflictingInfoError(Exception):
    pass
class SurveyOutOfBoundsError(Exception): # make these inherit from each other (or something) to avoid repetitive code
    pass
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None

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

# beams
def UAA_Gaussian(X,Y,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    return np.exp(-ln2*((X/fwhm_x)**2+(Y/fwhm_y)**2)/r0**2)
def UAA_Airy(X,Y,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    thetaX=X/r0
    argX=thetaX*BasicAiryHWHM/fwhm_x
    thetaY=Y/r0
    argY=thetaY*BasicAiryHWHM/fwhm_y
    perp=((j1(argX+eps)*j1(argY+eps))/((argX+eps)*(argY+eps)))**2
    return perp
def PA_Gaussian(u,v,ctr,fwhm):
    u0,v0=ctr
    fwhmx,fwhmy=fwhm
    evaled=((pi*ln2)/(fwhmx*fwhmy))*np.exp(-pi**2*((u-u0)**2*fwhmx**2+(v-v0)**2*fwhmy**2)/np.log(2))
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
                 bmin,bmax,                                                             # extreme baselines of the array
                 nu_ctr,delta_nu,                                                       # for the survey of interest
                 evol_restriction_threshold=def_evol_restriction_threshold,             # how close to coeval is close enough?
                 
                 # beam generalities
                 primary_beam_categ="UAA",primary_beam_type="Gaussian",                 # modelling choices
                 primary_beam_aux=None,primary_beam_uncs=None,                          # helper arguments
                 manual_primary_beam_modes=None,                                        # config space pts at which a pre–discretely sampled primary beam is known

                 # additional considerations for per-antenna systematics
                 PA_N_pert_types=0,PA_N_pbws_pert=0,                                    # numbers of perturbation types, primary beam widths to perturb
                 PA_N_fidu_types=N_fid_beam_types,PA_fidu_types_prefactors=None,        # how many kinds of fiducial beams and how to set them apart
                 PA_N_timesteps=def_PA_N_timesteps,PA_ioname="placeholder",             # numbers of timesteps to put in rotation synthesis, in/output file name
                 PA_distribution="random",mode="full",per_channel_systematic=None,
                 per_chan_syst_facs=[1.05,0.9,1.25],

                 # FORECASTING
                 pars_set_cosmo=pars_Planck18,pars_forecast=pars_Planck18,              # implement soon: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                 uncs=None,frac_unc=0.1,                                                # for Fisher-type calcs
                 pars_forecast_names=None,                                              # for verbose output
                 P_fid_for_cont_pwr=None, k_idx_for_window=0,                         # examine contaminant power or window functions?

                 # NUMERICAL 
                 n_sph_modes=256,dpar=None,                                             # conditioning the CAMB/etc. call
                 init_and_box_tol=0.05,CAMB_tol=0.05,                                   # considerations for k-modes at different steps
                 Nkpar_box=15,Nkperp_box=18,frac_tol_conv=0.1,                          # considerations for cyl binned power spectra from boxes
                 no_monopole=True,                                                      # enforce zero-mean in realization boxes?
                 ftol_deriv=1e-16,maxiter=5,                                            # subtract off monopole moment to give zero-mean box?
                 PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,            # pixels per side of gridded uv plane, uv binning chunk snapshot tightness
                 radial_taper=None,

                 # CONVENIENCE
                 ceil=0,                                                                # avoid any high kpars to speed eval? (for speedy testing, not science) 
                 PA_recalc=True                                                        # save time by not repeating per-antenna calculations? 
                 
                 ):                                                                                                                                                     
                
        """
        bmin,bmax                  :: floats                       :: max and min baselines of the array       :: m
        ceil                       :: int                          :: # high-kpar channels to ignore           :: ---
        primary_beam_categ         :: str                          :: * UAA = uniform across the array         :: ---
                                                                      * PA  = per-antenna
                                                                      * manual = pathological from elsewhere
        primary_beam_type          :: str                          :: * UAA: Gaussian, Airy                    :: ---
                                                                      * PA: Gaussian [MORE IN PROGRESS]
                                                                      * manual: None
        primary_beam_aux           :: (N_args,) of floats          :: * UAA:pass FWHMs; r0 appended internally :: r0:           Mpc
                                                                      * manual: primary beams evaluated on the    fwhms:        rad
                                                                        grid of interest, a list ordered as       evaled beams: ---
                                                                        [fidu,pert]
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
        uncs                       :: (Nkpar_surv,Nkperp_surv) of  :: unc in power spec @each cyl survey mode  :: K^2 Mpc^3 (same as power)
                                      floats
        frac_unc                   :: float                        :: if init w/ uncs=None, set uncs as        :: ---
                                                                      frac_unc*(fiducial power @survey modes) 
        Nkpar_box,Nkperp_box       :: ints                         :: # modes to put along cyl axes in power   :: ---
                                                                      spec calcs from boxes
        frac_tol_conv              :: float                        :: how much the Poisson noise must fall off :: ---
        pars_forecast_names        :: (N_pars_forecast,) or equiv. :: names of the pars being forecast         :: ---
                                      of strs
        manual_primary_beam_modes  :: x,y,z coordinate axes        :: domain of a discrete sampling            :: Mpc
                                      (if primary_beam !callable)
        no_monopole                :: bool                         :: y/n enforce mean-0 in box realizations   :: ---
        PA_N_pert_types            :: int                          :: # classes of PB (per-antenna only)       :: ---
        PA_N_pbws_pert             :: int                          :: # antennas w/ pertn PBs (per-ant only)   :: ---
        PA_N_timesteps             :: int                          :: # time steps in rotation synthesis (per- :: ---
                                                                      antenna only)
        PA_N_grid_pix              :: int                          :: # bins per side for uv plane gridding    :: ---
                                                                      (per-antenna only)
        PA_img_bin_tol             :: float                        :: # how much padding (to avoid ringing) to :: ---
                                                                      put in uv-plane gridding (per-ant only)
        PA_ioname                  :: str                          :: fname to save/load stacked per-ant boxes :: ---
        PA_recalc                  :: bool                         :: recalculate per-antenna beamed boxes?    :: ---
        PA_distribution            :: str                          :: how to distribute perturbation types     :: ---
        PA_N_fidu_types   :: int
        PA_fidu_types_prefactors   :: (PA_N_fidu_types,)  :: initial inroads into making the dif fidu :: ---
                                      of floats                       beam classes actually dif (multiplic.
                                                                      prefactor compared to lambda/D)
        mode                       :: str                          :: full, PF, or intermed states tbd later   :: ---

        short-term extensions:
        * the flexibility to introduce per-channel chromaticity systematics for each fiducial beam class
        """
        # primary beam considerations
        self.primary_beam_categ=primary_beam_categ
        if (primary_beam_categ.lower()!="manual"):
            self.fwhm_x,self.fwhm_y=primary_beam_aux
            self.primary_beam_uncs= primary_beam_uncs
            self.epsx,self.epsy=    self.primary_beam_uncs

        if (primary_beam_categ.lower()=="uaa"):
            if (primary_beam_type.lower()!="gaussian" and primary_beam_type.lower()!="airy"):
                raise NotYetImplementedError
        elif (primary_beam_categ.lower()=="pa" or primary_beam_categ.lower()=="manual"):
            if (primary_beam_categ.lower()=="pa"):
                self.per_chan_syst_facs=per_chan_syst_facs
                if mode=="full":
                        N_ant=512
                elif mode=="pathfinder":
                    N_ant=64
                self.N_ant=N_ant
                self.N_bl=int(N_ant*(N_ant-1)/2)
                if PA_recalc:
                    self.PA_N_pert_types=          PA_N_pert_types
                    self.PA_N_pbws_pert=           PA_N_pbws_pert
                    if (self.PA_N_pbws_pert>N_ant): # need to bring the mode full/pf stuff up here
                        print("WARNING: as called, more antennas would be perturbed than present in this array configuration")
                        print("resetting with merely all antennas perturbed...")
                        PA_N_pbws_pert=N_ant
                        self.PA_N_pbws_pert=PA_N_pbws_pert
                    self.PA_N_timesteps=           PA_N_timesteps
                    self.PA_N_grid_pix=            PA_N_grid_pix
                    self.img_bin_tol=              PA_img_bin_tol
                    self.PA_distribution=          PA_distribution
                    self.PA_N_fidu_types= PA_N_fidu_types
                    self.PA_fidu_types_prefactors= PA_fidu_types_prefactors
                    fwhm=primary_beam_aux # now with two polarizations!
                    self.eps=primary_beam_uncs 

                    real=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=0,
                                    pbw_pert_frac=[0.,0.],
                                    N_timesteps=self.PA_N_timesteps,
                                    N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                    distribution=self.PA_distribution,
                                    N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                    outname=PA_ioname,
                                    per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
                    real.stack_to_box()
                    print("constructed real-beamed box")
                    real_box=real.box
                    xy_vec=real.xy_vec
                    z_vec=real.z_vec
                    thgt=per_antenna(mode=mode,pbw_fidu=fwhm,N_pert_types=self.PA_N_pert_types,
                                    pbw_pert_frac=self.primary_beam_uncs,
                                    N_timesteps=self.PA_N_timesteps,
                                    N_pbws_pert=PA_N_pbws_pert,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                    distribution=self.PA_distribution,
                                    N_fiducial_beam_types=PA_N_fidu_types,fidu_types_prefactors=PA_fidu_types_prefactors,
                                    outname=PA_ioname,
                                    per_channel_systematic=per_channel_systematic,per_chan_syst_facs=self.per_chan_syst_facs)
                    thgt.stack_to_box()
                    print("constructed perturbed-beamed box")
                    thgt_box=thgt.box
                    per_chan_syst_name=thgt.per_chan_syst_name
                    self.per_chan_syst_name=per_chan_syst_name

                    # np.save("fidu_box_"+PA_ioname+".npy",fidu_box)
                    np.save("real_box_"+PA_ioname+".npy",real_box)
                    np.save("thgt_box_"+PA_ioname+".npy",thgt_box)
                    np.save("xy_vec_"+  PA_ioname+".npy",xy_vec)
                    np.save("z_vec_"+   PA_ioname+".npy",z_vec)
                else:
                    # fidu_box=np.load("fidu_box_"+PA_ioname+".npy")
                    real_box=np.load("real_box_"+PA_ioname+".npy")
                    thgt_box=np.load("thgt_box_"+PA_ioname+".npy")
                    xy_vec=  np.load("xy_vec_"+  PA_ioname+".npy")
                    z_vec=   np.load("z_vec_"+   PA_ioname+".npy")

                primary_beam_aux=[real_box,thgt_box]
                manual_primary_beam_modes=(xy_vec,xy_vec,z_vec)
            
            # now do the manual-y things
            if (manual_primary_beam_modes is None):
                raise NotEnoughInfoError
            else:
                self.manual_primary_beam_modes=manual_primary_beam_modes
            try:
                self.manual_primary_real,self.manual_primary_thgt=primary_beam_aux # assumed to be sampled at the same config space points
            except: # primary beam samplings not unpackable the way they need to be
                raise NotEnoughInfoError
        else:
            raise PathologicalError # as far as primary power beam perturbations go, they can all pretty much be described as being applied UAA, PA, or in some externally-implemented custom way

        self.primary_beam_type=primary_beam_type
        self.primary_beam_aux=primary_beam_aux
        self.primary_beam_uncs=primary_beam_uncs
        
        # forecasting considerations
        self.pars_set_cosmo=pars_set_cosmo
        self.N_pars_set_cosmo=len(pars_set_cosmo)
        self.pars_forecast=pars_forecast
        self.N_pars_forecast=len(pars_forecast)
        self.n_sph_modes=n_sph_modes
        self.dpar=dpar
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
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
            self.perturbed_primary_beam_aux=(self.fwhm_x*(1-self.epsx),self.fwhm_y*(1-self.epsy))
            self.primary_beam_aux=np.array([self.fwhm_x,self.fwhm_y,self.r0]) 
            self.perturbed_primary_beam_aux=np.append(self.perturbed_primary_beam_aux,self.r0)
        elif (primary_beam_type.lower()=="manual"):
            pass
        else:
            raise NotYetImplementedError
        self.P_fid_for_cont_pwr=P_fid_for_cont_pwr
        self.k_idx_for_window=k_idx_for_window

        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        self.ceil=ceil
        self.kpar_surv=kpar_surv
        if self.ceil>0:
            self.kpar_surv=self.kpar_surv[:-self.ceil]
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        self.bmax=bmax
        self.kperp_surv=kperp(self.nu_ctr,self.N_bl,self.bmin,self.bmax)
        self.Nkperp_surv=len(self.kperp_surv)

        self.kmin_surv=np.min((self.kpar_surv[ 0],self.kperp_surv[ 0]))
        self.kmax_surv=np.sqrt(self.kpar_surv[-1]**2+self.kperp_surv[-1]**2)

        self.Lsurv_box_xy=twopi/self.kperp_surv[0]
        self.Nvox_box_xy=int(self.Lsurv_box_xy*self.kperp_surv[-1]/pi)
        self.Lsurv_box_z=twopi/self.kpar_surv[0]
        self.Nvox_box_z=int(self.Lsurv_box_z*self.kpar_surv[-1]/pi)
        print("Nxy,Nz for generated box realizations=",self.Nvox_box_xy,self.Nvox_box_z)

        # self.NvoxPracticalityWarning()

        # numerical protections for assorted k-ranges
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init*np.sqrt(3) # factor of sqrt(3) from pythag theorem for box to prevent the need for extrap
        self.ksph,self.Ptruesph=self.get_mps(self.pars_set_cosmo,kmin_CAMB,kmax_CAMB)
        self.Deltabox_xy=self.Lsurv_box_xy/self.Nvox_box_xy
        self.Deltabox_z= self.Lsurv_box_z/ self.Nvox_box_z
        if mode=="UAA":
            if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
                self.all_sigmas=self.r0*np.array([self.fwhm_x,self.fwhm_y])/np.sqrt(2*np.log(2))
                print("self.all_sigmas,self.Deltabox_xy,self.Deltabox_z=",self.all_sigmas,self.Deltabox_xy,self.Deltabox_z)
                if (np.any(self.all_sigmas<self.Deltabox_xy) or np.any(self.all_sigmas<self.Deltabox_z)):
                    raise NumericalDeltaError
            elif (primary_beam_type.lower()=="manual"):
                print("WARNING: unable to do a robust numerical delta error check when a manual beam is passed")
            else:
                raise NotYetImplementedError
        self.radial_taper=radial_taper

        # considerations for power spectra binned to survey k-modes
        _,_,self.Pcyl=self.unbin_to_Pcyl(self.pars_set_cosmo)
        self.frac_unc=frac_unc
        if (uncs==None):
            uncs=self.frac_unc*self.Pcyl
            uncs[uncs==0]=huge
            self.uncs=uncs
        else:
            self.uncs=uncs

        # precision control for numerical derivatives
        self.ftol_deriv=ftol_deriv
        self.eps=eps
        self.maxiter=maxiter

        # considerations for power spectrum binning directly from the box
        self.Nkpar_box=Nkpar_box
        self.Nkperp_box=Nkperp_box
        self.frac_tol_conv=frac_tol_conv
        self.no_monopole=no_monopole
        
        # considerations for printing the calculated bias results
        self.pars_forecast_names=pars_forecast_names
        assert (len(pars_forecast)==len(pars_forecast_names))

        # holder for numerical derivatives of a cylindrically binned power spectrum (sampled at the survey modes) wrt the params being forecast
        self.cyl_partials=np.zeros((self.N_pars_forecast,self.Nkpar_surv,self.Nkperp_surv))

        with open("settings.txt", "w") as file:
            file.write("primary beam width systematics category           = "+str(primary_beam_categ)+"\n")
            file.write("                               type (if UAA mode) = "+str(primary_beam_type)+"\n")
            file.write("                               distribution       = "+str(PA_distribution)+"\n")
            file.write("central frequency of survey                       = "+str(nu_ctr)+"\n")
            file.write("observing setup                                   = "+str(mode)+"\n")
            file.write("number of high-kparallel channels truncated       = "+str(ceil)+"\n")
            file.write("Poisson noise convergence threshold               = "+str(self.frac_tol_conv)+"\n")
            file.write("per-channel systematic                            = "+str(per_channel_systematic)+"\n")
            file.write("number of fiducial beam types (if applicable)     = "+str(PA_N_fidu_types)+"\n")
            file.write("number of perturbed beam types (if applicable)    = "+str(PA_N_pert_types)+"\n")
            file.write("number of perturbed primary beams (if applicable) = "+str(PA_N_pbws_pert)+"\n")

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
    
    def unbin_to_Pcyl(self,pars_to_use):
        """
        interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        """
        k,Psph_use=self.get_mps(pars_to_use,minkh=self.kmin_surv,maxkh=self.kmax_surv)
        Psph_use=Psph_use.reshape((Psph_use.shape[1],))
        self.Psph=Psph_use
        kpar_grid,kperp_grid=np.meshgrid(self.kpar_surv,self.kperp_surv,indexing="ij")
        kmag_grid=np.sqrt(kpar_grid**2+kperp_grid**2)

        kmag_grid_flat=np.reshape(kmag_grid,(self.Nkpar_surv*self.Nkperp_surv,))
        Psph_interpolator=interp1d(k,Psph_use,kind="cubic",bounds_error=False,fill_value="extrapolate")
        P_interp_flat=Psph_interpolator(kmag_grid_flat)
        Pcyl=np.reshape(P_interp_flat,(self.Nkpar_surv,self.Nkperp_surv))
        return kpar_grid,kperp_grid,Pcyl
    
    def NvoxPracticalityWarning(self,threshold_lo=75,threshold_hi=200):
        prefix="WARNING: the specified survey requires"
        voxel_names=["Nxy_box","Nz_box"]
        for i,voxel_number in enumerate([self.Nvox_box_xy,self.Nvox_box_z]):
            if voxel_number>threshold_hi:
                print(prefix+" "+str(voxel_names[i])+"= {:4}, which may cause slow eval".format(voxel_number))
            elif voxel_number<threshold_lo:
                print(prefix+" "+str(voxel_names[i])+"= {:4}, which is suspiciously coarse".format(voxel_number))

    def calc_power_contamination(self, isolated=False):
        """
        calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
        contaminant power, calculated as [see memo] useful combinations of three different instrument responses
        """
        if self.P_fid_for_cont_pwr is None:
            P_fid=np.reshape(self.Ptruesph,(self.n_sph_modes))
        elif self.P_fid_for_cont_pwr=="window": # make the fiducial power spectrum a numerical top hat
            P_fid=np.zeros(self.n_sph_modes)
            P_fid[self.k_idx_for_window]=1.
        else:
            raise NotYetImplementedError

        print("self.primary_beam_categ=",self.primary_beam_categ)
        print("self.primary_beam_type=",self.primary_beam_type)
        if self.primary_beam_categ!="manual":
            if (self.primary_beam_type=="Gaussian"):
                pb_here=UAA_Gaussian
            elif (self.primary_beam_type=="Airy"):
                pb_here=UAA_Airy
            else:
                raise NotYetImplementedError
            fi=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           P_fid=P_fid,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph, no_monopole=self.no_monopole,
                           radial_taper=self.radial_taper)
        else: 
            raise NotYetImplementedError
        if self.primary_beam_categ=="UAA":
            rt=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           P_fid=P_fid,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                           primary_beam_num=pb_here,primary_beam_aux_num=self.perturbed_primary_beam_aux,primary_beam_type_num=self.primary_beam_type,
                           primary_beam_den=pb_here,primary_beam_aux_den=self.perturbed_primary_beam_aux,primary_beam_type_den=self.primary_beam_type,
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph, no_monopole=self.no_monopole,
                           radial_taper=self.radial_taper)
        elif self.primary_beam_categ=="PA":
            rt=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           P_fid=P_fid,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                           primary_beam_num=self.manual_primary_real,primary_beam_type_num="manual", # THIS SHOULD HAVE BEEN THE "REAL," NOT "THOUGHT" BEAM ALL ALONG
                           primary_beam_den=self.manual_primary_thgt,primary_beam_type_den="manual",
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph,
                           manual_primary_beam_modes=self.manual_primary_beam_modes, no_monopole=self.no_monopole,
                           radial_taper=self.radial_taper)
        else:
            raise NotYetImplementedError # need to re-implement fully manual windowing
        
        recalc_fi=False
        recalc_rt=False
        if isolated==False:
            recalc_fi=True
            recalc_rt=True
        if isolated=="thought":
            recalc_rt=True
        if isolated=="fiue":
            recalc_fi=True

        if recalc_fi:
            fi.avg_realizations(interfix="fi")
            self.N_cumul=fi.N_cumul
            self.Pfiducial_cyl=fi.P_converged
            interp_holder=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,P_fid=self.Pfiducial_cyl,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                    Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                    k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                                    no_monopole=self.no_monopole)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Pfiducial_cyl_surv=interp_holder.P_interp
        if recalc_rt:
            rt.avg_realizations(interfix="rt")
            if not recalc_fi:
                self.N_cumul=rt.N_cumul
            self.Prealthought_cyl=rt.P_converged
            interp_holder=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,P_fid=self.Prealthought_cyl,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                        Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                        k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                                        no_monopole=self.no_monopole)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Prealthought_cyl_surv=interp_holder.P_interp
        if isolated==False:
            self.Pcont_cyl_surv=self.Pfiducial_cyl_surv-self.Prealthought_cyl_surv
        
        interp_holder=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,P_fid=self.N_cumul,Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                    Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                    k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                                    no_monopole=self.no_monopole)
        interp_holder.interpolate_P(use_P_fid=True)
        self.N_cumul_surv=interp_holder.P_interp

    def cyl_partial(self,n):  
        """        
        cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
        """
        dparn=self.dpar[n]
        pcopy=self.pars_set_cosmo.copy()
        pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

        P0=np.mean(np.abs(self.Pcyl))+self.eps
        tol=self.ftol_deriv*P0 # generalizes tol=ftol*f0 from PHYS512

        pcopy[n]=pcopy[n]+2*dparn # don't need to generate a fresh copy immediately before b/c the initial copy hasn't been modified yet
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
            self.cyl_partials[n,:,:]=estimate
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
                self.cyl_partials[n,:,:]=fallback

    def build_cyl_partials(self):
        """
        builds a (N_pars_forecast,Nkpar,Nkperp) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
        """
        for n in range(self.N_pars_set_cosmo):
            self.iter=0 # bc starting a new partial deriv calc.
            self.cyl_partial(n)
        
    def bias(self):
        """
        collect and stitch together the ingredients of the parameter bias calculation
        """
        self.build_cyl_partials()
        print("built partials")
        self.calc_power_contamination()
        print("computed Pcont")
        V=0.*self.cyl_partials
        for i in range(self.N_pars_forecast):
            V[i,:,:]=self.cyl_partials[i,:,:]/self.uncs # elementwise division for an nkpar x nkperp slice
        V_completely_transposed=np.transpose(V,axes=(2,1,0))
        F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("computed F")
        Pcont_div_sigma=self.Pcont_cyl_surv/self.uncs
        B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
        print("computed B")
        self.biases=(np.linalg.inv(F)@B).reshape((self.N_pars_forecast,))
        print("computed b")

    def print_survey_characteristics(self):
        print("survey properties.......................................................................")
        print("........................................................................................")
        print("survey centred at.......................................................................\n    nu ={:>7.4}     MHz \n    z  = {:>9.4} \n    Dc = {:>9.4f}  Mpc\n".format(float(self.nu_ctr),self.z_ctr,self.r0))
        print("survey spans............................................................................\n    nu =  {:>5.4}    -  {:>5.4}    MHz (deltanu = {:>6.4}    MHz) \n    z =  {:>9.4} - {:>9.4}     (deltaz  = {:>9.4}    ) \n    Dc = {:>9.4f} - {:>9.4f} Mpc (deltaDc = {:>9.4f} Mpc)\n".format(self.nu_lo,self.nu_hi,self.bw,self.z_hi,self.z_lo,self.z_hi-self.z_lo,self.Dc_hi,self.Dc_lo,self.Dc_hi-self.Dc_lo))
        if (self.primary_beam_type.lower()!="manual"):
            print("characteristic instrument response widths...............................................\n    beamFWHM0 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_x,self.epsx))
            print("specific to the cylindrically asymmetric beam...........................................\n    beamFWHM1 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_y,self.epsy))
        print("cylindrically binned wavenumbers of the survey..........................................\n    kparallel {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} channels of width {:>7.4}  Mpc**(-1)) \n    kperp     {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} bins of width {:>8.4} Mpc**(-1))\n".format(self.kpar_surv[0],self.kpar_surv[-1],self.Nkpar_surv,self.kpar_surv[-1]-self.kpar_surv[-2],   self.kperp_surv[0],self.kperp_surv[-1],self.Nkperp_surv,self.kperp_surv[-1]-self.kperp_surv[-2]))
        print("cylindrically binned k-bin sensitivity..................................................\n    fraction of Pcyl amplitude = {:>7.4}".format(self.frac_unc))

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
                 Nk0=10,Nk1=0,binning_mode="lin",                                                   # binning considerations for power spec realizations (log mode not fully tested yet b/c not impt. for current pipeline)
                 frac_tol=0.1,                                                                      # max number of realizations
                 k0bins_interp=None,k1bins_interp=None,                                             # bins where it would be nice to know about P_converged
                 P_realizations=None,P_converged=None,                                              # power spectra related to averaging over those from dif box realizations
                 verbose=False,                                                                     # status updates for averaging over realizations
                 k_fid=None,kind="nearest",avoid_extrapolation=False,                                 # helper vars for converting a 1d fid power spec to a box sampling
                 no_monopole=True,                                                                  # consideration when generating boxes
                 manual_primary_beam_modes=None,                                                    # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 radial_taper=None):                                                                     # implement soon: quick way to use an Airy beam in per-antenna mode
        """
        Lxy,Lz                    :: float                       :: side length of cosmo box          :: Mpc
        T_pristine                :: (Nvox,Nvox,Nvox) of floats  :: cosmo box (just physics/no beam)  :: K
        T_primary                 :: (Nvox,Nvox,Nvox) of floats  :: cosmo box * primary beam          :: K
        P_fid                     :: (Nk0_fid,) of floats        :: sph binned fiducial power spec    :: K^2 Mpc^3
        Nvox,Nvoxz                :: float                       :: cosmobox#vox/side,z-ax can differ :: ---
        primary_beam              :: callable (or, if            :: power beam in Cartesian coords    :: ---
                                     primary_beam_type=="manual" 
                                     a 3D array)          
        primary_beam_aux         :: tuple of floats             :: Gaussian, Airy: μ, σ      :: Gaussian: r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_type         :: str                         :: for now: Gaussian / Airy          :: ---
        Nk0, Nk1                  :: int                         :: # power spec bins for axis 0/1    :: ---
        binning_mode              :: str                         :: lin/log sp. P_realizations bins   :: ---
        frac_tol                  :: float                       :: max fractional amount by which    :: ---
                                                                    the p.s. avg can change w/ the 
                                                                    addition of the latest realiz. 
                                                                    and the ensemble average is 
                                                                    considered converged
        k0bins_interp,            :: (Nk0_interp,) of floats     :: bins to which to interpolate the  :: 1/Mpc
        k1bins_interp                (Nk1_interp,) of floats        converged power spec (prob set
                                                                    by survey considerations)
        P_realizations            :: if Nk1==0: (Nk0,)    floats :: sph/cyl power specs for dif       :: K^2 Mpc^3
                                     if Nk1>0:  (Nk0,Nk1) floats    realizations of a cosmo box 
        P_converged               :: same as that of P_fid       :: average of P_realizations         :: K^2 Mpc^3
        verbose                   :: bool                        :: every 10% of realization_ceil     :: ---
        k_fid                     :: (Nk0_fid,) of floats        :: modes where P_fid is sampled      :: 1/Mpc
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
        else:            # the kind of rectangular prism box I care about for dirty image stacking (+probably other extension)
            self.Lz=Lz
            self.Lxy=Lxy
        self.P_fid=P_fid
        self.T_primary=T_primary
        self.T_pristine=T_pristine
        self.no_monopole=no_monopole
        if ((T_primary is None) and (T_pristine is None) and (P_fid is None)): # require either a box or a fiducial power spec (il faut some way of determining #voxels/side; passing just Nvox is not good enough)
            raise NotEnoughInfoError
        else:                                                                  # there is possibly enough info to proceed, but still need to check for conflicts and gaps
            if ((T_pristine is not None) and (T_primary is not None)):
                print("WARNING: T_pristine and T_primary both passed; T_primary will be temporarily ignored and then internally overwritten to ensure consistency with primary_beam")
                if (T_pristine.shape!=T_primary.shape):
                    raise ConflictingInfoError
                else:                                                          # use box shape to set cubic/ rectangular prism box attributes
                    self.Nvox,_,self.Nvoxz=T_primary.shape
            if ((Nvox is not None) and (T_pristine is not None)):              # possible conflict: if both Nvox and a box are passed, 
                T_pristine_shape0,_,T_pristine_shape2=T_pristine.shape
                if (Nvox!=T_pristine.shape[0]):                                # but Nvox and the box shape disagree,
                    raise ConflictingInfoError                                 # estamos en problemas
                else:
                    self.Nvox= T_pristine_shape0                               # otherwise, initialize the Nvox attributes
                    self.Nvoxz=T_pristine_shape2
            elif (Nvox is not None):                                           # if Nvox was passed but T was not, use Nvox to initialize the Nvox attributes
                self.Nvox=Nvox 
                if (Nvoxz is None):                                            # if no Nvoxz was provided, make the box cubic
                    self.Nvoxz=Nvox
                else:
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
                        self.fid_Nk0,self.fid_Nk1=Pfidshape
                        if primary_beam_num is not None: 
                            raise ConflictingInfoError # primary beam 1 needs to be the fiducial one; doesn't make sense to claim you have a perturbed but not fiducial pb
                    else:
                        try: # see if the power spec is a CAMB-esque (1,npts) array
                            self.P_fid=np.reshape(P_fid,(Pfidshape[-1],)) # make the CAMB MPS shape amenable to the calcs internal to this class
                        except: # barring that...
                            pass # treat the power spectrum as being truly cylindrically binned
                elif (Pfiddims==1):
                    self.fid_Nk0=Pfidshape[0] # already checked that P_fid is 1d, so no info is lost by extracting the int in this one-element tuple, and fid_Nk0 being an integer makes things work the way they should down the line
                    self.fid_Nk1=0
                else:
                    raise UnsupportedBinningMode
        
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
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the generate_box direction)
        self.kmag_grid_centre=fftshift(self.kmag_grid_corner) 
        self.kmag_grid_corner_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**2*self.Nvoxz,))

        # if P_fid was passed, establish its values on the k grid (helpful when generating a box)
        self.k_fid=k_fid
        self.kind=kind
        self.avoid_extrapolation=avoid_extrapolation
        if (self.P_fid is not None and self.k_fid is not None):
            if (len(self.P_fid.shape)==1): # truly 1d fiducial power spec (by this point, even CAMB-like shapes have been reshuffled)
                self.P_fid_interp_1d_to_3d()
            elif (len(self.P_fid.shape)==2):
                self.k_fid0,self.kfid1=self.k_fid # fiducial k-modes should be unpackable, since P_fid has been verified to be truly 2d
                self.P_fid_interp_2d_to_3d()
            else: # so far, I do not anticipate working with "truly three dimensional"/ unbinned power spectra
                raise NotYetImplementedError
        
        # binning considerations
        self.binning_mode=binning_mode
        self.Nk0=Nk0 # the number of bins to put in power spec realizations you construct (ok if not the same as the number of bins in the fiducial power spec)
        self.Nk1=Nk1
        self.kmax_box_xy= pi/self.Deltaxy
        self.kmax_box_z=  pi/self.Deltaz
        self.kmin_box_xy= twopi/self.Lxy
        self.kmin_box_z=  twopi/self.Lz
        self.k0bins,self.limiting_spacing_0=self.calc_bins(self.Nk0,self.Nvoxz,self.kmin_box_z,self.kmax_box_z)
        if self.limiting_spacing_0<self.Deltakz: # trying to bin more finely than the box can tell you about (guaranteed to have >=1 empty bin)
            raise ResolutionError
        
        if (self.Nk1>0):
            self.k1bins,self.limiting_spacing_1=self.calc_bins(self.Nk1,self.Nvox,self.kmin_box_xy,self.kmax_box_xy)
            if (self.limiting_spacing_1<self.Deltakxy): # idem ^
                raise ResolutionError
        else:
            self.k1bins=None
        
            # voxel grids for sph binning        
        self.sph_bin_indices_centre=      np.digitize(self.kmag_grid_centre,self.k0bins)
        self.sph_bin_indices_1d_centre=   np.reshape(self.sph_bin_indices_centre, (self.Nvox**2*self.Nvoxz,))

            # voxel grids for cyl binning
        if (self.Nk1>0):
            self.kpar_column_centre= np.abs(fftshift(self.kz_vec_for_box_corner))                                      # magnitudes of kpar for a representative column along the line of sight (z-like)
            self.kperp_slice_centre= np.sqrt(fftshift(self.kx_grid_corner)**2+fftshift(self.ky_grid_corner)**2)[:,:,0] # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
            self.perpbin_indices_slice_centre=    np.digitize(self.kperp_slice_centre,self.k1bins)                     # cyl kperp bin that each voxel falls into
            self.perpbin_indices_slice_1d_centre= np.reshape(self.perpbin_indices_slice_centre,(self.Nvox**2,))        # 1d version of ^ (compatible with np.bincount)
            self.parbin_indices_column_centre=    np.digitize(self.kpar_column_centre,self.k0bins)                     # cyl kpar bin that each voxel falls into

        # tapering/apodization
        if radial_taper is not None:
            beta=6 # starting point recommended in the documentation is 14
            taper_x=np.ones(Nvox)
            taper_y=np.ones(Nvox)
            taper_z=radial_taper(self.Nvoxz,beta)/self.Nvoxz
            taper_xxx,taper_yyy,taper_zzz=np.meshgrid(taper_x,taper_y,taper_z,indexing="ij")
            taper_xyz=taper_xxx*taper_yyy*taper_zzz/(self.Nvox**2*self.Nvoxz)
            self.taper_xyz=taper_xyz
            self.taper_sum=np.sum(taper_xyz**2)
        else:
            self.taper_xyz=1. # this is a box-shaped array in the other branch, but there's no point in lugging around an array of ones when it's equally valid to scalar-multiply a numpy array
            self.taper_sum=1.
        print("self.taper_sum=",self.taper_sum)

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
                    raise ConflictingInfoError 
                if self.manual_primary_beam_modes is None:
                    raise NotEnoughInfoError

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
                evaled_primary_num=interpn(manual_primary_beam_modes,
                                         self.primary_beam_num,
                                         (self.xx_grid,self.yy_grid,self.zz_grid),
                                         method=self.kind,bounds_error=self.avoid_extrapolation,fill_value=None)
            else:
                raise NotYetImplementedError
            
        self.primary_beam_den=primary_beam_den
        self.primary_beam_aux_den=primary_beam_aux_den
        self.primary_beam_type_den=primary_beam_type_den
        self.manual_primary_beam_modes=manual_primary_beam_modes # _fi and _rt assumed to be sampled at the same modes, if this is the case
        if (self.primary_beam_den is not None): # non-identity PERTURBED primary beam
            if (self.primary_beam_type_den=="Gaussian" or self.primary_beam_type_den=="Airy"):
                self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_aux_den
                evaled_primary_den=  self.primary_beam_den(self.xx_grid,self.yy_grid,self.fwhm_x,  self.fwhm_y,  self.r0)                
            elif (self.primary_beam_type_den=="manual"):
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable
                    primary_beam_den.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ConflictingInfoError 
                if self.manual_primary_beam_modes is None:
                    raise NotEnoughInfoError

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
                evaled_primary_den=interpn(manual_primary_beam_modes,
                                         self.primary_beam_den,
                                         (self.xx_grid,self.yy_grid,self.zz_grid),
                                         method=self.kind,bounds_error=self.avoid_extrapolation,fill_value=None)
            else:
                evaled_primary_den=None    

            if evaled_primary_den is not None:
                evaled_primary_use=evaled_primary_den
            else:
                evaled_primary_use=evaled_primary_num

            evaled_primary_for_div=np.copy(evaled_primary_den)
            evaled_primary_for_mul=np.copy(evaled_primary_num)
            evaled_primary_for_div[evaled_primary_for_div<nearly_zero]=maxfloat # protect against division-by-zero errors
            self.evaled_primary_for_div=evaled_primary_for_div
            self.evaled_primary_for_mul=evaled_primary_for_mul
            self.effective_volume=np.sum(evaled_primary_use**2*self.d3r) # should be **2 but this is causing a gross underestimate. I also revisited previous typos of *2 and *1 (multiplication, not exponentials) and, of course, those were no good either. the problem must be elsewhere.
            fft_beam_for_map=fftshift(fftn(ifftshift(evaled_primary_den/np.sum(evaled_primary_den))))
            print("using beam-modulated volume")
        else:                               # identity primary beam
            self.effective_volume=self.Lxy**2*self.Lz
            self.evaled_primary_for_div=np.ones((self.Nvox,self.Nvox,self.Nvoxz))
            self.evaled_primary_for_mul=np.copy(self.evaled_primary_for_div)
            print("using physical volume")
        print("self.effective_volume,self.Nvox**2*self.Nvoxz=",self.effective_volume,self.Nvox**2*self.Nvoxz)
        if (self.T_pristine is not None):
            self.T_primary=self.T_pristine*self.evaled_primary_num # APPLY THE FIDUCIAL BEAM
        
        # strictness control for realization averaging
        self.frac_tol=frac_tol
        self.realization_ceiling=int(np.round(self.frac_tol**-2))
        self.verbose=verbose

        # P_converged interpolation bins
        self.k0bins_interp=k0bins_interp
        self.k1bins_interp=k1bins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        if (P_realizations is not None):       # maybe you want to import realizations from a prev run and just add more? (unclear why you'd have left the
            self.P_realizations=P_realizations # prev run w/o a converged average, unless, maybe, you want to re-run with a stricter convergence threshold?)
        else:
            self.P_realizations=[] 
        if (P_converged is not None):          # maybe you have a converged power spec average from a previous calc and just want to interpolate or generate more box realizations?
            self.P_converged=P_converged
        else:
            self.P_converged=None
        self.P_interp=None                     # can't init with this because, if you had one, there'd be no point of using cosmo_stats b/c the job is already done (at best, you can provide a P_fid)
        self.not_converged=None

    def calc_bins(self,Nki,Nvox_to_use,kmin_to_use,kmax_to_use):
        """
        generate a set of bins spaced according to the desired scheme with max and min
        """
        if (self.binning_mode=="log"):
            kbins=np.logspace(np.log10(kmin_to_use),np.log10(kmax_to_use),num=Nki)
            limiting_spacing=twopi*(10.**(kmax_to_use)-10.**(kmax_to_use-(np.log10(Nvox_to_use)/Nki))) # twopi*(10.**(self.kmax_box)-10.**(self.kmax_box-(np.log10(self.Nvox)/Nki)))
        elif (self.binning_mode=="lin"):
            kbins=np.linspace(kmin_to_use,kmax_to_use,Nki)
            limiting_spacing=twopi*(0.5*Nvox_to_use-1)/(Nki) # version for a kmax that is "aware that" there are +/- k-coordinates in the box
        else:
            raise UnsupportedBinningMode
        return kbins,limiting_spacing # kbins            -> floors of the bins to which the power spectrum will be binned (along one axis)
                                      # limiting_spacing -> smallest spacing between adjacent bins (uniform if linear; otherwise, depends on the binning strategy)
    
    def P_fid_interp_1d_to_3d(self):
        """
        * interpolate a "physics-only" (spherically symmetric) power spectrum (e.g. from CAMB) to a 3D cosmological box.
        * for now, I don't have a solution better than overwriting the k=0 term after the fact (because extrapolation to this term based on a reasonable CAMB call leads to negative power there)
        """
        P_fid_interpolator=interp1d(self.k_fid,self.P_fid,kind=self.kind,bounds_error=self.avoid_extrapolation,fill_value="extrapolate")
        P_interp_flat=P_fid_interpolator(self.kmag_grid_corner_flat)
        self.P_fid_box=np.reshape(P_interp_flat,(self.Nvox,self.Nvox,self.Nvoxz))
            
    def generate_P(self,send_to_P_fid=False,T_use=None):
        """
        philosophy: 
        * compute the power spectrum of a known cosmological box and bin it spherically or cylindrically
        * append to the list of reconstructed P realizations (self.P_realizations)
        """
        if T_use is None:
            T_use=self.T_pristine
        else:
            T_use=self.T_primary
        # T_use=np.ones_like(self.T_primary) # PLACEHOLDER FOR TESTING
        if (self.T_pristine is None):    # power spec has to come from a box
            self.generate_box() # populates/overwrites self.T_pristine and self.T_primary
        T_tilde=fftshift(fftn((ifftshift(T_use*self.taper_xyz)*self.d3r)))
        modsq_T_tilde=(T_tilde*np.conjugate(T_tilde)).real
        modsq_T_tilde[:,:,self.Nvoxz//2]*=2 # fix pos/neg duplication issue at the origin
        denom=self.effective_volume*self.taper_sum
        # print("self.effective_volume=",self.effective_volume)
        # print("self.taper_sum=",self.taper_sum)
        unbinned_power=modsq_T_tilde/denom
        if (self.Nk1==0):   # bin to sph
            unbinned_power_1d= np.reshape(unbinned_power,    (self.Nvox**2*self.Nvoxz,))

            sum_unbinned_power= np.bincount(self.sph_bin_indices_1d_centre, 
                                           weights=unbinned_power_1d, 
                                           minlength=self.Nk0)       # for the ensemble avg: sum    of unbinned_power values in each bin
            N_unbinned_power=   np.bincount(self.sph_bin_indices_1d_centre,
                                           minlength=self.Nk0)       # for the ensemble avg: number of unbinned_power values in each bin
            sum_unbinned_power_truncated=sum_unbinned_power[:-1]       # excise sneaky corner modes: I devised my binning to only tell me about voxels w/ k<=(the largest sphere fully enclosed by the box), and my bin edges are floors. But, the highest floor corresponds to the point of intersection of the box and this largest sphere. To stick to my self-imposed "the stats are not good enough in the corners" philosophy, I must explicitly set aside the voxels that fall into the "catchall" uppermost bin. 
            N_unbinned_power_truncated=  N_unbinned_power[:-1]         # idem ^
            final_shape=(self.Nk0,)
        elif (self.Nk0!=0): # bin to cyl
            sum_unbinned_power= np.zeros((self.Nk0+1,self.Nk1+1)) # for the ensemble avg: sum    of unbinned_power values in each bin  ...upon each access, update the kparBIN row of interest, but all Nkperp columns
            N_unbinned_power=   np.zeros((self.Nk0+1,self.Nk1+1)) # for the ensemble avg: number of unbinned_power values in each bin
            for i in range(self.Nvoxz): # iterate over the kpar axis of the box to capture all LoS slices
                if (i==0): # stats for the representative "bull's eye" slice transverse to the LoS
                    slice_bin_counts=np.bincount(self.perpbin_indices_slice_1d_centre, minlength=self.Nk1)
                unbinned_power_slice= unbinned_power[:,:,i]                    # take the slice of interest of the preprocessed box values !!kpar is z-like
                unbinned_power_slice_1d= np.reshape(unbinned_power_slice, 
                                                   (self.Nvox**2,))          # 1d for bincount compatibility
                current_binsums= np.bincount(self.perpbin_indices_slice_1d_centre,
                                             weights=unbinned_power_slice_1d, 
                                             minlength=self.Nk1)             # this slice's update to the numerator of the ensemble average
                current_par_bin=self.parbin_indices_column_centre[i]

                sum_unbinned_power[current_par_bin,:]+= current_binsums  # update the numerator   of the ensemble avg
                N_unbinned_power[  current_par_bin,:]+= slice_bin_counts # update the denominator of the ensemble avg
            
            sum_unbinned_power_truncated= sum_unbinned_power[:-1,:-1] # excise sneaky corner modes (see the analogous operation in the sph branch for an explanation)
            N_unbinned_power_truncated=   N_unbinned_power[  :-1,:-1] # idem ^
            final_shape=(self.Nk0,self.Nk1)

        N_unbinned_power_truncated[N_unbinned_power_truncated==0]=maxint # avoid division-by-zero errors during the division the estimator demands
        self.N_modes_per_bin=N_unbinned_power_truncated
        self.N_cumul=self.N_modes_per_bin*self.realization_ceiling

        avg_unbinned_power=sum_unbinned_power_truncated/(N_unbinned_power_truncated) # actual estimator math

        P=np.array(avg_unbinned_power)
        P.reshape(final_shape)
        if send_to_P_fid: # if generate_P was called speficially to have a spec from which all future box realizations will be generated
            self.P_fid=P
            self.P_fid_interp_1d_to_3d() # generate interpolated values of the newly established 1D P_fid over the k-magnitudes of the box
        else:             # the "normal" case where you're just accumulating a realization
            self.P_realizations.append([P])
        self.unbinned_P=unbinned_power # box-shaped, but calculated according to the power spectrum estimator equation

    def generate_box(self):
        """
        philosophy: 
        * generate a box that comprises a random realization of a known power spectrum
        * this always generates a "pristine" box and stores it in self.T_pristine
        * if primary_beam is not None, self.T_beamed is also populated
        """
        assert(self.Nvox>=self.Nk0), PathologicalError
        if (self.P_fid is None):
            try:
                self.generate_P(store_as_P_fid=True) # T->P_fid is deterministic, so, even if you start with a random realization, it'll be helpful to have a power spec summary stat to generate future realizations
            except: # something goes wrong in the P_fid calculation
                raise NotEnoughInfoError
        # not checking for 2D-ness of P_fid here since I've already done that during init (yes, support for cyl binned P_fid is still functionality I want to add eventually)
        # not warning abt potentially overwriting T -> the only case where info would be lost is where self.P_fid is None, and I already have a separate warning for that
        
        assert(self.P_fid_box is not None)
        if (self.effective_volume<=0):
            raise PathologicalError
        if (np.min(self.P_fid_box)<0):
            self.P_fid_box[self.P_fid_box<0]=0 # hackily overwriting error from having to extrapolate at the origin
        sigmas=np.sqrt(self.effective_volume*self.P_fid_box/2.) # from inverting the estimator equation and turning variances into std devs
        T_tilde_Re,T_tilde_Im=np.random.normal(loc=0.*sigmas,scale=sigmas,size=np.insert(sigmas.shape,0,2))
        
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        T=fftshift(irfftn(T_tilde*self.d3k,s=(self.Nvox,self.Nvox,self.Nvoxz),axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: fftshiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
        if self.no_monopole:
            T-=np.mean(T) # subtract monopole moment to make things more akin to what powerbox does
        self.T_pristine=T
        self.T_primary=T*self.evaled_primary_for_mul

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
        for i in range(self.realization_ceiling):
            self.generate_box()
            self.generate_P(T_use="primary")
            if self.verbose:
                if (i%(self.realization_ceiling//10)==0):
                    print("realization",i)
            ti=time.time()
            if ((ti-t0)>3600): # actually save the realizations every hour
                np.save("P_"+interfix+"_unconverged.npy",np.mean(self.P_realizations,axis=0))
                t0=time.time()

        arr_realiz_holder=np.array(self.P_realizations)
        self.P_realizations=arr_realiz_holder
        if (arr_realiz_holder.shape[0]>1):
            P_converged=np.mean(arr_realiz_holder,axis=0)
        else:
            P_converged=arr_realiz_holder

        if (self.Nk1>0):
            self.P_converged=np.reshape(P_converged,(self.Nk0,self.Nk1))
        else:
            self.P_converged=np.reshape(P_converged,(self.Nk0,))

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
            if (self.k0bins_interp is None):
                raise NotEnoughInfoError

        if (self.k1bins_interp is not None):
            kpar_have_lo=  self.k0bins[0]
            kpar_have_hi=  self.k0bins[-1]
            kperp_have_lo= self.k1bins[0]
            kperp_have_hi= self.k1bins[-1]

            kpar_want_lo=  self.k0bins_interp[0]
            kpar_want_hi=  self.k0bins_interp[-1]
            kperp_want_lo= self.k1bins_interp[0]
            kperp_want_hi= self.k1bins_interp[-1]

            if (kpar_want_lo<kpar_have_lo):
                extrapolation_warning("low kpar",   kpar_want_lo,  kpar_have_lo)
            if (kpar_want_hi>kpar_have_hi):
                extrapolation_warning("high kpar",  kpar_want_hi,  kpar_have_hi)
            if (kperp_want_lo<kperp_have_lo):
                extrapolation_warning("low kperp",  kperp_want_lo, kperp_have_lo)
            if (kperp_want_hi>kperp_have_hi):
                extrapolation_warning("high kperp", kperp_want_hi, kperp_have_hi)
            self.k0_interp_grid,self.k1_interp_grid=np.meshgrid(self.k0bins_interp,self.k1bins_interp,indexing="ij")
            self.P_interp=interpn((self.k0bins,self.k1bins),self.P_converged,(self.k0_interp_grid,self.k1_interp_grid),method=self.kind,bounds_error=self.avoid_extrapolation,fill_value=None)
        else:
            k_have_lo=self.k0bins[0]
            k_have_hi=self.k0bins[-1]
            k_want_lo=self.k0bins_interp[0]
            k_want_hi=self.k0bins_interp[-1]
            if (k_want_lo<k_have_lo):
                extrapolation_warning("low k",k_want_lo,k_have_lo)
            if (k_want_hi>k_have_hi):
                extrapolation_warning("high k",k_want_hi,k_have_hi)
            P_interpolator=interp1d(self.k0bins,self.P_converged,kind=self.kind,bounds_error=self.avoid_extrapolation,fill_value="extrapolate")
            self.P_interp=P_interpolator(self.k0bins_interp)
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
                 N_timesteps=def_N_timesteps,nu_ctr=nu_HI_z0,
                 pbw_fidu=None,N_grid_pix=def_PA_N_grid_pix,Delta_nu=0.1953125,
                 distribution="random",fidu_types_prefactors=None,
                 outname=None,per_channel_systematic=None,per_chan_syst_facs=None
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
        self.fidu_types_prefactors=fidu_types_prefactors
        self.Delta_nu=Delta_nu
        self.N_NS=N_NS_full
        self.N_EW=N_EW_full
        self.DRAO_lat=DRAO_lat
        if (mode=="pathfinder"):
            self.N_NS=self.N_NS//2
            self.N_EW=self.N_EW//2
        self.bmax=np.sqrt(self.N_NS*b_NS**2+self.N_EW*b_EW**2)
        self.N_ant=self.N_NS*self.N_EW
        self.N_bl=self.N_ant*(self.N_ant-1)//2
        self.observing_dec=observing_dec
        self.nu_ctr_MHz=nu_ctr
        self.nu_ctr_Hz=nu_ctr*1e6
        self.N_hrs=synthesized_beam_crossing_time(self.nu_ctr_Hz,bmax=self.bmax,dec=self.observing_dec) # freq needs to be in Hz
        self.lambda_obs=c/self.nu_ctr_Hz
        if (pbw_fidu is None):
            pbw_fidu=self.lambda_obs/D
            pbw_fidu=[pbw_fidu,pbw_fidu]
        self.pbw_fidu=np.array(pbw_fidu) # NEEDS TO BE UNPACKABLE AS X,Y ... but pointless to re-cast to np array here bc I've already done so in the calling routine
        
        # antenna positions xyz
        antennas_EN=np.zeros((self.N_ant,2))
        for i in range(self.N_NS):
            for j in range(self.N_EW):
                antennas_EN[i*self.N_EW+j,:]=[j*b_EW,i*b_NS]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        offset=offset_deg*pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset.
        offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(self.N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (self.N_ant,1)) # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T
        self.antennas_xyz=antennas_xyz

        N_beam_types=(self.N_pert_types+1)*self.N_fiducial_beam_types 
        self.N_beam_types=N_beam_types

        # array layout, organized and indexed by fiducial beam type
        if fidu_types_prefactors is None:
            fidu_types_prefactors=np.ones(N_fiducial_beam_types)
        self.fidu_types_prefactors=fidu_types_prefactors
        pbw_fidu_types=np.zeros((self.N_ant,))
        if self.distribution=="random":
            pbw_fidu_types=np.random.randint(0,self.N_fiducial_beam_types,size=(self.N_ant,))
            np.savetxt("pbw_fidu_types.txt",pbw_fidu_types)
        elif self.distribution=="corner":
            if self.N_fiducial_beam_types!=4:
                raise ConflictingInfoError # in order to use corner mode, you need four fiducial beam types
            pbw_fidu_types=np.zeros((self.N_NS,self.N_EW))
            half_NS=self.N_NS//2
            half_EW=self.N_EW//2
            pbw_fidu_types[:half_NS,half_EW:]=1
            pbw_fidu_types[half_NS:,:half_EW]=2
            pbw_fidu_types[half_NS:,half_EW:]=3 # the quarter of the array with no explicit overwriting keeps its idx=0 (as necessary)
            pbw_fidu_types=np.reshape(pbw_fidu_types,(self.N_ant,))
        elif self.distribution=="diagonal":
            raise NotYetImplementedError
        elif self.distribution=="rowcol":
            raise NotYetImplementedError
        elif self.distribution=="ring":
            raise NotYetImplementedError
        else:
            raise NotYetImplementedError
        
        # seed the systematics (still doing this randomly throughout the array)
        pbw_pert_types=np.zeros((self.N_ant,))
        epsilons=np.zeros(N_pert_types+1)
        if (self.N_pbws_pert>0):
            if (self.N_pert_types>1):
                epsilons[1:]=self.pbw_pert_frac*np.random.uniform(size=np.insert(self.N_pert_types,0,1))
            else: 
                epsilons=self.pbw_pert_frac
            indices_of_ants_w_pert_pbws=np.random.randint(0,self.N_ant,size=self.N_pbws_pert) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
            pbw_pert_types[indices_of_ants_w_pert_pbws]=np.random.randint(1,high=(self.N_pert_types+1),size=np.insert(self.N_pbws_pert,0,1)) # leaves as zero the indices associated with unperturbed antennas
            np.savetxt("pbw_pert_types.txt",pbw_pert_types)
        else:
            indices_of_ants_w_pert_pbws=None
        self.pbw_pert_types=pbw_pert_types
        self.indices_of_ants_w_pert_pbws=indices_of_ants_w_pert_pbws
        self.epsilons=epsilons
        self.per_chan_syst_facs=per_chan_syst_facs
        
        # ungridded instantaneous uv-coverage (baselines in xyz)        
        uvw_inst=np.zeros((self.N_bl,3))
        indices_of_constituent_ant_pb_fidu_types=np.zeros((self.N_bl,2))
        indices_of_constituent_ant_pb_pert_types=np.zeros((self.N_bl,2))
        k=0
        for i in range(self.N_ant):
            for j in range(i+1,self.N_ant):
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
                    plt.scatter(antennas_xyz[keep,0],antennas_xyz[keep,1],c="C"+str(j),marker=symbols[i],label=str(j)+str(i),lw=0.3,s=20) # change j and i to permute
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("CHORD "+str(self.nu_ctr_MHz)+" MHz pointing dec="+str(round(self.observing_dec,5))+" rad \n"
                      "projected antenna positions by primary beam status\n"
                      "[antenna fiducial status][antenna perturbation status]=")
            fig.legend(loc="outside right upper")
            plt.savefig("layout_"+outname+".png",dpi=350)
        
            ant_a_pert_type,ant_b_pert_type=indices_of_constituent_ant_pb_pert_types.T
            ant_a_fidu_type,ant_b_fidu_type=indices_of_constituent_ant_pb_fidu_types.T
            Nrow=9 # make this less hard-coded
            Ncol=int(np.ceil(N_beam_types**2/Nrow))
            fig,axs=plt.subplots(Nrow,Ncol,figsize=(self.N_beam_types*2.25,self.N_beam_types*2.25))
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
            plt.savefig("inst_uv_"+outname+".png",dpi=250)

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*self.N_hrs/12
        hour_angles=np.linspace(0,hour_angle_ceiling,self.N_timesteps)
        thetas=hour_angles*15*np.pi/180
        
        zenith=np.array([np.cos(self.observing_dec),0,np.sin(self.observing_dec)]) # Jon math redux
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        project_to_dec=np.vstack([east,north])

        uv_synth=np.zeros((2*self.N_bl,2,self.N_timesteps))
        for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
            accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                        [-np.sin(theta),np.cos(theta),0],
                                        [ 0,            0,            1]])
            uvw_rotated=uvw_inst@accumulate_rotation
            uvw_projected=uvw_rotated@project_to_dec.T
            uv_synth[:,:,i]=uvw_projected/self.lambda_obs
        self.uv_synth=uv_synth
        print("synthesized rotation")

    def calc_dirty_image(self, Npix=1024, pbw_fidu_use=None,tol=img_bin_tol):
        if pbw_fidu_use is None: # otherwise, use the one that was passed
            pbw_fidu_use=self.pbw_fidu
        uvmin=np.min([np.min(self.uv_synth[:,0,:]),np.min(self.uv_synth[:,1,:])])
        uvmax=np.max([np.max(self.uv_synth[:,0,:]),np.max(self.uv_synth[:,1,:])])
        uvbins=np.linspace(tol*uvmin,tol*uvmax,Npix)
        self.uvmin=uvmin
        self.uvmax=uvmax
        Npix=uvbins.shape[0]
        self.Npix=Npix
        uvmagmin=np.sort(np.abs(uvbins))[1]
        thetamax=1/uvmagmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
        self.thetamax=thetamax
        d2u=uvbins[1]-uvbins[0]
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

        uvplane/=(self.N_beam_types**2*np.sum(uvplane)) # divide out the artifact of there having been multiple convolutions
        self.uvplane=uvplane
        dirty_image=np.abs(fftshift(ifft2(ifftshift(uvplane)*d2u,norm="forward")))
        dirty_image/=np.sum(dirty_image) # also account for renormalization in image space
        uv_bin_edges=[uvbins,uvbins]
        self.dirty_image=dirty_image
        self.uv_bin_edges=uv_bin_edges
        return dirty_image,uv_bin_edges,thetamax

    def stack_to_box(self,evol_restriction_threshold=def_evol_restriction_threshold, tol=img_bin_tol):
        if (self.nu_ctr_MHz<(350/(1-evol_restriction_threshold/2)) or self.nu_ctr_MHz>(nu_HI_z0/(1+evol_restriction_threshold/2))):
            raise SurveyOutOfBoundsError
        self.img_bin_tol=tol
        N_grid_pix=self.N_grid_pix
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/self.Delta_nu)
        self.nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        self.nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(self.nu_hi,self.nu_lo,N_chan) # decr.
        surv_channels_Hz=1e6*surv_channels_MHz
        surv_wavelengths=c/surv_channels_Hz # incr.
        self.surv_channels=surv_channels_Hz
        self.z_channels=nu_HI_z0/surv_channels_MHz-1.
        self.comoving_distances_channels=np.asarray([comoving_distance(chan) for chan in self.z_channels]) # incr.
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]
        surv_beam_widths=1.029*surv_wavelengths/D # incr.
        plt.figure()
        plt.plot(surv_channels_MHz,surv_beam_widths,label="diffraction-limited Airy FWHM")    
        per_chan_syst_name="None"        
        if self.per_channel_systematic=="D3A_like":
            surv_beam_widths=(surv_beam_widths)**1.2 # keep things dimensionless, but use a steeper decay
            noise_bound_lo=0.75
            noise_bound_hi=1.25
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
            raise NotYetImplementedError
        if self.per_channel_systematic is not None:
            plt.plot(surv_channels_MHz,surv_beam_widths,label="chromaticity systematic–laden")
        plt.xlabel("frequency (MHz)")
        plt.ylabel("beam FWHM (rad)")
        plt.title("reference beam widths by frequency bin")
        plt.legend()
        plt.savefig("beam_chromaticity_slice_"+str(self.nu_ctr_MHz)+"_MHz_"+per_chan_syst_name+".png")
        self.per_chan_syst_name=per_chan_syst_name

        # rescale chromatic beam widths by whatever was passed
        xy_beam_widths=np.array((surv_beam_widths,surv_beam_widths)).T
        ctr_chan_beam_width=(c/(self.nu_ctr_Hz*D))
        xy_beam_widths[:,0]*=(self.pbw_fidu[0]/ctr_chan_beam_width)
        xy_beam_widths[:,1]*=(self.pbw_fidu[1]/ctr_chan_beam_width)

        box=np.zeros((N_grid_pix,N_grid_pix,N_chan))
        xy_beam_widths_desc=np.flip(xy_beam_widths,axis=0)
        for i,xy_beam_width in enumerate(xy_beam_widths_desc): # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=surv_wavelengths[i] # update the observing frequency for next time

            # compute the dirty image
            chan_dirty_image,chan_uv_bin_edges,thetamax=self.calc_dirty_image(Npix=N_grid_pix, pbw_fidu_use=xy_beam_width, tol=self.img_bin_tol)
            
            # interpolate to store in stack
            if i==0:
                uv_bin_edges_0=chan_uv_bin_edges[0]
                uu_bin_edges_0,vv_bin_edges_0=np.meshgrid(uv_bin_edges_0,uv_bin_edges_0,indexing="ij")
                theta_max_box=thetamax
                interpolated_slice=chan_dirty_image
            else: # chunk excision and mode interpolation in one step
                interpolated_slice=interpn(chan_uv_bin_edges,
                                           chan_dirty_image,
                                           (uu_bin_edges_0,vv_bin_edges_0),
                                           bounds_error=False, fill_value=None) # extrap necessary because the smallest u and v you have at a given slice-needing-extrapolation will be larger than the min u and v mags to extrapolate to
            box[:,:,i]=interpolated_slice
            if ((i%(N_chan//4))==0):
                print("{:7.1f} pct complete".format(i/N_chan*100))
        box/=np.sum(box)
        self.box=box 
        self.theta_max_box=theta_max_box

        # generate a box of r-values (necessary for interpolation to survey modes in the manual beam mode of cosmo_stats as called by beam_effects)
        thetas=np.linspace(-self.theta_max_box,self.theta_max_box,N_grid_pix)
        xy_vec=self.ctr_chan_comov_dist*thetas # making the coeval approximation
        z_vec=self.comoving_distances_channels-self.ctr_chan_comov_dist 
        self.xy_vec=xy_vec
        self.z_vec=z_vec

def cyl_sph_plots(redo_window_calc, redo_box_calc,
              mode, nu_ctr, epsxy,
              ceil, frac_tol_conv, N_sph,
              categ, beam_type, # categ is manual/UAA/PA, beam_type is Airy/Gaussian
              N_fidu_types, N_pert_types, 
              N_pbws_pert, per_channel_systematic,
              PA_dist, f_types_prefacs, plot_qty, 
                  
              pars=None, parnames=None, dpar=None, 
                  
              b_NS_CHORD=b_NS,N_NS_CHORD=N_NS_full,
              b_EW_CHORD=b_EW,N_EW_CHORD=N_EW_full,
              freq_bin_width=0.1953125, # kHz
              
              from_saved_power_spectra=False,
              contaminant_or_window=None, k_idx_for_window=0,
              isolated=False,
              per_chan_syst_facs=[]): # the default chromaticity systematic

    ############################## bundling and preparing Planck18 cosmo params of interest here ########################################################################################################################
    if pars is None:
        scale=1e-9
        pars_Planck18=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,           ns_Planck18])
        parnames=                ['H_0',       'Omega_b h**2',      'Omega_c h**2',      '10**9 * A_S',        'n_s'       ]
        pars_Planck18[3]/=scale # A_s management (avoid numerical conditioning–related issues)
        nprm=len(pars_Planck18)
        dpar=1e-3*np.ones(nprm) # starting point (numerical derivatives have adaptive step size)
        dpar[3]*=scale

        pars=pars_Planck18

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
    elif mode=="full": # 24x22=528 antennas (512 w/ receiver hut gaps), 1010 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2)

    if categ=="PA":
        print("PA mode currently only supports a Gaussian beam")
        beam_type="Gaussian"
    hpbw_x= 1.029*wl_ctr_m/D *  pi/180. # rad; lambda/D estimate (actually physically realistic)
    hpbw_y= 0.75 * hpbw_x         # we know this tends to be a little narrower, based on measurements (...from D3A ...so far)

    ############################## pipeline administration ########################################################################################################################


    if contaminant_or_window is not None:
        qty_title_prefix="Window function "
        c_or_w="wind"
    else:
        qty_title_prefix=""
        c_or_w="cont"
    per_chan_syst_string="none"
    per_chan_syst_name=""
    if per_channel_systematic=="D3A_like":
        per_chan_syst_string="D3AL"
    elif per_channel_systematic=="sporadic":
        per_chan_syst_string="spor"
        for fac in per_chan_syst_facs:
            per_chan_syst_name=per_chan_syst_name+str(fac)+"_"
    PA_dist_string="rand"
    if PA_dist=="corner":
        PA_dist_string="corn"


    ioname=mode+"_"+c_or_w+"_"+categ+"_"\
           ""+per_chan_syst_string+"_"+per_chan_syst_name+"_"\
           ""+str(int(nu_ctr))+"MHz__"\
           "ceil_"+str(ceil)+"__"\
           "cosmicvar_"+str(round(frac_tol_conv,2))+"__"\
           "Nreal_"+str(N_fidu_types)+"__"\
           "Npert_"+str(N_pert_types)+"_"+str(N_pbws_pert)+"__"\
           "dist_"+PA_dist_string+"__"\
           "epsxy_"+str(epsxy)+"__"\
           "realprefacs_"+str(f_types_prefacs)

    if plot_qty=="P":
        qty_title=qty_title_prefix+"Power"
        y_label="P (K$^2$ Mpc$^3$)"
    elif plot_qty=="Delta2":
        qty_title=qty_title_prefix+"Dimensionless power"
        y_label="Δ$^2$ (log(K$^2$/K$^2$)"

    ############################## run the pipeline or load results ########################################################################################################################
    if categ!="manual":
        bundled_non_manual_primary_aux=np.array([hpbw_x,hpbw_y])
        bundled_non_manual_primary_uncs=np.array([epsxy,epsxy])
        if categ=="UAA":
            windowed_survey=beam_effects(
                                            # SCIENCE
                                            # the observation
                                            bminCHORD,bmaxCHORD,                                
                                            nu_ctr,freq_bin_width,                             
                                            evol_restriction_threshold=def_evol_restriction_threshold,    
                                                
                                            # beam generalities
                                            primary_beam_categ=categ,primary_beam_type=beam_type,       
                                            primary_beam_aux=bundled_non_manual_primary_aux,
                                            primary_beam_uncs=bundled_non_manual_primary_uncs,

                                            # FORECASTING
                                            pars_set_cosmo=pars,pars_forecast=pars,        
                                            pars_forecast_names=parnames,
                                            P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,                           

                                            # NUMERICAL 
                                            n_sph_modes=N_sph,dpar=dpar,                                   
                                            init_and_box_tol=0.05,CAMB_tol=0.05,                           
                                            Nkpar_box=15,Nkperp_box=18,frac_tol_conv=frac_tol_conv,                      
                                            no_monopole=False,                                                    
                                            ftol_deriv=1e-16,maxiter=5,                                      
                                            PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,        
                                                
                                            # CONVENIENCE
                                            ceil=ceil                                                                                                       
                                            )

            pert_title="primary beam widths perturbed uniformly across the array"
            categ_title=pert_title
        elif categ=="PA":
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
                                            PA_N_pert_types=N_pert_types,PA_N_pbws_pert=N_pbws_pert,
                                            PA_N_fidu_types=N_fidu_types,
                                            PA_fidu_types_prefactors=f_types_prefacs,
                                            PA_N_timesteps=def_PA_N_timesteps,PA_ioname=ioname,             # numbers of timesteps to put in rotation synthesis, in/output file name
                                            PA_distribution=PA_dist,mode=mode,
                                            per_channel_systematic=per_channel_systematic,
                                            per_chan_syst_facs=per_chan_syst_facs,

                                            # FORECASTING
                                            pars_set_cosmo=pars,pars_forecast=pars,              # implement soon: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                                            uncs=None,frac_unc=0.1,                                                # for Fisher-type calcs
                                            pars_forecast_names=parnames,                                              # for verbose output
                                            P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,

                                            # NUMERICAL 
                                            n_sph_modes=N_sph,dpar=dpar,                                             # conditioning the CAMB/etc. call
                                            init_and_box_tol=0.05,CAMB_tol=0.05,                                   # considerations for k-modes at different steps
                                            Nkpar_box=15,Nkperp_box=18,frac_tol_conv=frac_tol_conv,                          # considerations for cyl binned power spectra from boxes
                                            no_monopole=False,                                                      # enforce zero-mean in realization boxes?
                                            ftol_deriv=1e-16,maxiter=5,                                            # subtract off monopole moment to give zero-mean box?
                                            PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,            # pixels per side of gridded uv plane, uv binning chunk snapshot tightness
                                            
                                            # CONVENIENCE
                                            ceil=ceil,                                                                # avoid any high kpars to speed eval? (for speedy testing, not science) 
                                            PA_recalc=redo_box_calc                                                        # save time by not repeating per-antenna calculations? 
                                            
                                            )

            if PA_dist=="random":
                PA_title=" randomly throughout the array"
            elif PA_dist=="corner":
                PA_title=" in separate corners"
            PA_title
            pert_title=str(N_pbws_pert)+" primary beam widths perturbed randomly throughout the array"
            categ_title="real beams arranged "+PA_title
    else:
        head="placeholder_fname_manual_"
        xy_vec=np.load(head+"_xy_vec.npy")
        z_vec=np.load(head+"_z_vec.npy")
        fidu=np.load(head+"_fidu.npy")
        pert=np.load(head+"_pert.npy")

        manual_primary_aux=[fidu,pert]
        windowed_survey=beam_effects(bminCHORD,bmaxCHORD,nu_ctr,
                                        freq_bin_width,
                                        categ,None,manual_primary_aux,None,
                                        pars_Planck18,pars_Planck18,
                                        N_sph,dpar,
                                        nu_ctr,freq_bin_width,
                                        frac_tol_conv=frac_tol_conv,
                                        pars_forecast_names=parnames, no_monopole=False,
                                        manual_primary_beam_modes=(xy_vec,xy_vec,z_vec))
    
    handle_fi=False
    handle_rt=False
    if isolated==False:
        handle_fi=True
        handle_rt=True
    if isolated=="thought":
        handle_rt=True
    if isolated=="true":
        handle_fi=True

    windowed_survey.print_survey_characteristics()
    if not from_saved_power_spectra:
        if redo_window_calc:
            t0=time.time()
            windowed_survey.calc_power_contamination(isolated=isolated)
            t1=time.time()
            print("Pcont calculation time was",t1-t0)

            if handle_fi:
                Pfiducial_cyl_surv=windowed_survey.Pfiducial_cyl_surv # self.Pfiducial_cyl_surv-self.Prealthought_cyl_surv
                np.save("Pfiducial_cyl_"+ioname+".npy",Pfiducial_cyl_surv)
            if handle_rt:
                Prealthought_cyl_surv=windowed_survey.Prealthought_cyl_surv
                np.save("Prealthought_cyl_surv_"+ioname+".npy",Prealthought_cyl_surv)
            N_cumul_surv=windowed_survey.N_cumul_surv
            np.save("N_cumul_surv_"+ioname+".npy",N_cumul_surv)
            if isolated is not False: # break early if you just calculate one windowed power spectrum at a time
                return None
        else:
            Prealthought_cyl_surv=np.load("Prealthought_cyl_surv_"+ioname+".npy")
            Pfiducial_cyl_surv=np.load("Pfiducial_cyl_"+ioname+".npy")
            N_cumul_surv=np.load("N_cumul_surv_"+ioname+".npy")
    else:
        Prealthought_cyl_surv=np.load("P_rt_unconverged.npy")
        Pfiducial_cyl_surv=np.load("P_fi_unconverged.npy")
        N_cumul_surv=np.load("N_cumul_surv_"+ioname+".npy")

    Pcont_cyl_surv=Prealthought_cyl_surv-Pfiducial_cyl_surv
    Pfidu_sph=windowed_survey.Ptruesph
    kfidu_sph=windowed_survey.ksph
    kmin_surv=windowed_survey.kmin_surv
    kmax_surv=windowed_survey.kmax_surv
    kpar=windowed_survey.kpar_surv
    kperp=windowed_survey.kperp_surv
    kpar_grid,kperp_grid=np.meshgrid(kpar,kperp,indexing="ij")

    super_title_string="{:5} MHz CHORD {} survey \n" \
                        "{}\n" \
                        "{}\n" \
                        "{} HPBW {:5.3}+\-{:5.3}% (x) and {:5.3}+/-{:5.3}% (y)\n" \
                        "systematic-laden and fiducially beamed {} (multiplicative offsets {})\n" \
                        "{} fiducial beam types; {} beam perturbation types\n" \
                        "per-channel systematics: {}\n" \
                        "numerical convenience factors: {} high k-parallel channels truncated and cosmic variance mitigated to {}%" \
                        "".format(nu_ctr,mode,
                                pert_title,
                                categ_title,
                                beam_type,hpbw_x,100*epsxy,hpbw_y,100*epsxy,
                                qty_title,f_types_prefacs,
                                N_fidu_types,N_pert_types,
                                per_channel_systematic,
                                ceil, int(frac_tol_conv*100))
    if contaminant_or_window=="window":
        super_title_string="WINDOW FUNCTIONS FOR\n"+super_title_string

    for_diagnostics=plt.cm.PRGn
    for_spectra=plt.cm.cividis
    fig,axs=plt.subplots(1,4,figsize=(12,6),layout="constrained")
    for i in range(4):
        axs[i].set_ylabel("k$_{||}$ (1/Mpc)")
        axs[i].set_xlabel("k$_{\perp} (1/Mpc)$")
    title_quantities=["P$_{fiducial}$",
                        "P$_{cont}$=P$_{real / thought}$-P$_{fiducial}$",
                        "P$_{real / thought}$",
                        "P$_{real / thought}$/P$_{fiducial}$"]
    plot_quantities=[Pfiducial_cyl_surv,
                        Pcont_cyl_surv,
                        Prealthought_cyl_surv,
                        Prealthought_cyl_surv/Pfiducial_cyl_surv]
    cmaps=[for_spectra,
           for_diagnostics,
           for_spectra,
           for_diagnostics]
    vcentres=[None,0,None,1]
    order=[0,2,1,3]
    edge=0.1
    for i,num in enumerate(order):
        vcentre=vcentres[num]
        plot_qty_here=plot_quantities[num]
        if vcentre is not None:
            norm=CenteredNorm(vcenter=vcentres[num],halfrange=0.5*(np.percentile(plot_qty_here,100-edge)-np.percentile(plot_qty_here,edge)))
        else: 
            norm=None
        im=axs[i].pcolor(kperp_grid.T,kpar_grid.T,plot_qty_here.T,cmap=cmaps[num],norm=norm)
        axs[i].set_title(title_quantities[num])
        axs[i].set_aspect('equal')
        if contaminant_or_window=="window":
            desired_xlims=axs[i].get_xlim()
            desired_ylims=axs[i].get_ylim()
            thetas=np.linspace(0,pi/2)
            r=kfidu_sph[k_idx_for_window]
            x=r*np.cos(thetas)
            y=r*np.sin(thetas)
            axs[i].plot(x,y,c="tab:orange")
            axs[i].set_xlim(desired_xlims)
            axs[i].set_ylim(desired_ylims)
        plt.colorbar(im,ax=axs[i]) # ,shrink=0.xx

    fig.suptitle(super_title_string)
    # fig.tight_layout()
    fig.savefig("CYL_"+ioname+".png",dpi=200)

    ############################## SPHERICAL PLOT ########################################################################################################################
    fig,axs=plt.subplots(1,2,figsize=(12,8))
    for i in range(2):
        axs[i].set_xlabel("k (1/Mpc)")
        axs[i].set_ylabel(y_label)
    axs[0].set_title("side-by-side")
    axs[1].set_title("fractional difference")
    Pfidu_sph=np.reshape(Pfidu_sph,(Pfidu_sph.shape[-1],))

    kcyl_mags_for_interp_grid=np.sqrt(kpar_grid**2+kperp_grid**2)
    N_cyl_k=len(kpar)*len(kperp)
    kcyl_mags_for_interp_flat=np.reshape(kcyl_mags_for_interp_grid,(N_cyl_k,))
    Pthought_cyl_surv_flat=np.reshape(Prealthought_cyl_surv,(N_cyl_k,))
    Ptrue_cyl_surv_flat=np.reshape(Pfiducial_cyl_surv,(N_cyl_k,))
    N_cumul_surv_flat=np.reshape(N_cumul_surv,(N_cyl_k,))

    sort_arr=np.argsort(kcyl_mags_for_interp_flat)
    k_interpolated=np.linspace(kmin_surv,kmax_surv,int(N_sph/10))
    kcyl_mags_for_interp_flat_sorted=kcyl_mags_for_interp_flat[sort_arr]
    Pthought_cyl_surv_flat_sorted=Pthought_cyl_surv_flat[sort_arr]
    Ptrue_cyl_surv_flat_sorted=Ptrue_cyl_surv_flat[sort_arr]
    N_cumul_surv_flat_sorted=N_cumul_surv_flat[sort_arr]
    Ptr_interpolated=np.interp(k_interpolated,kcyl_mags_for_interp_flat_sorted,Ptrue_cyl_surv_flat_sorted)
    Pth_interpolated=np.interp(k_interpolated,kcyl_mags_for_interp_flat_sorted,Pthought_cyl_surv_flat_sorted)
    N_cumul_interpolated=np.interp(k_interpolated,kcyl_mags_for_interp_flat[sort_arr],N_cumul_surv_flat_sorted)

    Poisson_term=np.sqrt(2/N_cumul_interpolated)
    lo_fac=(1-Poisson_term)
    hi_fac=(1+Poisson_term)

    Pthought_lo=Pth_interpolated*lo_fac
    Pthought_hi=Pth_interpolated*hi_fac
    Ptrue_lo=Ptr_interpolated*lo_fac
    Ptrue_hi=Ptr_interpolated*hi_fac

    Delta2_fac_interpolated=k_interpolated**3/(twopi**2)
    Delta2_fidu=Pfidu_sph*kfidu_sph**3/(twopi**2)
    Delta2_rt=Pth_interpolated*Delta2_fac_interpolated
    Delta2_fi=Ptr_interpolated*Delta2_fac_interpolated
    Delta2_rt_lo=Pthought_lo*Delta2_fac_interpolated
    Delta2_rt_hi=Pthought_hi*Delta2_fac_interpolated
    Delta2_fi_lo=Ptrue_lo*Delta2_fac_interpolated
    Delta2_fi_hi=Ptrue_hi*Delta2_fac_interpolated

    fidu=[Pfidu_sph,Delta2_fidu]
    thought=[Pth_interpolated,Delta2_rt]
    true=[Ptr_interpolated,Delta2_fi]
    thought_lo=[Pthought_lo,Delta2_rt_lo]
    thought_hi=[Pthought_hi,Delta2_rt_hi]
    true_lo=[Ptrue_lo,Delta2_fi_lo]
    true_hi=[Ptrue_hi,Delta2_fi_hi]

    if plot_qty=="P":
        k=0
    elif plot_qty=="Delta2":
        k=1

    axs[0].semilogy(k_interpolated,true[k],c="C1",label="fiducial/fiducial windowing")
    axs[0].semilogy(k_interpolated,thought[k],c="C0",label="real/knowable windowing")
    axs[0].fill_between(k_interpolated,true_lo[k],true_hi[k],color="C1",alpha=0.5)
    axs[0].fill_between(k_interpolated,thought_lo[k],thought_hi[k],color="C0",alpha=0.5)

    frac_dif=(true[k]-thought[k])/true[k]
    axs[1].plot(k_interpolated,frac_dif,c="C2")
    if contaminant_or_window=="window":
        for i in range(2):
            axs[i].axvline(kfidu_sph[k_idx_for_window],c="C3")

    desired_xlims_0=axs[0].get_xlim()
    axs[0].plot(kfidu_sph,Pfidu_sph,c="C2",label="unwindowed")
    axs[0].set_xlim(desired_xlims_0)
    axs[0].legend()
    fig.suptitle(super_title_string)
    fig.tight_layout()
    fig.savefig("SPH_"+ioname+".png",dpi=225)