from cosmo_distances import *
from forecasting_pipeline import *

Nft=[1,3]
Npt=[1,2]
power_comparison_plots(redo_window_calc=True, redo_box_calc=False,
                       mode="pathfinder", nu_ctr=600, epsxy=0.1,
                       frac_tol_conv=0.05, categ="PA", # categ is manual/PA/CST, beam_type is either Gaussian (for PA) or manual (for CST) # frac_tol_conv=0.005 for Fir
                       N_fidu_types=Nft, N_pert_types=Npt, 
                       N_pbws_pert=64, per_channel_systematic=None,
                       PA_dist="rowcol", plot_qty="P",
                       Nkpar_box=None,Nkperp_box=None, 
                            
                       layer_foregrounds=True, isolated=False)