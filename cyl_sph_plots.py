from cosmo_distances import *
from forecasting_pipeline import *

redo_window_calc=True
redo_box_calc=True

contaminant_or_window=None # None (calculates contaminant power), "window"
mode="pathfinder" # "pathfinder", "full"
nu_ctr=900. 
frac_tol_conv=0.005

N_fidu_types=1
N_pert_types=1
N_pbws_pert=0
per_channel_systematic=None # None, "D3A_like", "sporadic"
PA_dist="random" # "random", "corner"
f_types_prefacs=[1.] # np.linspace(0.85,1.15,N_fidu_types) example nontrivial; [1.] trivial 
plot_qty="P"
epsxy=0.

cyl_sph_plots(redo_window_calc,
              redo_box_calc,
              mode, nu_ctr, epsxy,
              0, frac_tol_conv, 256, # ceil=0, N_sph=256 works for now.
              "PA", "Airy", # PA mode is the only one to incorporate realistic chromaticity, but it uses a Gaussian beam for closed-form convolution Fourier dual speedup. make this less hacky eventually, but por ahora es lo que hay.
              N_fidu_types, N_pert_types, 
              N_pbws_pert, per_channel_systematic,
              PA_dist, f_types_prefacs, plot_qty,
              contaminant_or_window=contaminant_or_window, k_idx_for_window=136,
              isolated=False, per_chan_syst_facs=[0.8]) # [1.05,0.89,1.01,1.03]