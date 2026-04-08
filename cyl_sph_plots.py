from cosmo_distances import *
from forecasting_pipeline import *

redo_window_calc=True
redo_box_calc=True

contaminant_or_window=None # None (calculates contaminant power), "window"
mode="pathfinder" # "pathfinder", "full"
nu_ctr=600. 
frac_tol_conv=0.01 # 0.005 for Fir runs

N_fidu_types=1
N_pert_types=1
N_pbws_pert=64
per_channel_systematic=None # None, "D3A_like", "sporadic"
PA_dist="random" # "random", "corner", "rowcol"
f_types_prefacs=[1.] # np.linspace(0.95,1.05,N_fidu_types) example nontrivial; [1.] trivial 
plot_qty="P"
epsxy=0.1

power_comparison_plots(redo_window_calc=redo_window_calc,
                      redo_box_calc=redo_box_calc,
                      mode=mode, nu_ctr=nu_ctr, epsxy=epsxy,
                      frac_tol_conv=frac_tol_conv,
                      categ="CST", beam_type="Gaussian", # PA mode is the only one to incorporate realistic chromaticity, but it uses a Gaussian beam for closed-form convolution Fourier dual speedup. make this less hacky eventually, but por ahora es lo que hay.
                      N_fidu_types=N_fidu_types, N_pert_types=N_pert_types, 
                      N_pbws_pert=N_pbws_pert, per_channel_systematic=per_channel_systematic,
                      PA_dist=PA_dist, f_types_prefacs=f_types_prefacs, plot_qty=plot_qty,
                      contaminant_or_window=contaminant_or_window, k_idx_for_window=136,
                      isolated=False, per_chan_syst_facs=[1.],wedge_cut=False,layer_foregrounds=False,
                      
                      CST_lo=0.58,CST_hi=0.62,CST_deltanu=0.0002,
                      beam_sim_directory="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams_Aditya/CHORD_CST_600/",
                      CST_f_head_fidu="fiducial/farfield_(f=",CST_f_head_real="fiducial/farfield_(f=",CST_f_head_thgt="feed_tilt/farfield_(f=" # stick with defaults for now
                      ) 