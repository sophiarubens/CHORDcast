from forecasting_pipeline import *

Nft=np.arange(1,4)
Npt=np.arange(0,3)
power_comparison_plots(redo_window_calc=False, redo_box_calc=False,
                       mode="pathfinder", nu_ctr=600.*u.MHz, epsxy=0.1,
                       frac_tol_conv=0.05, categ="PA", # categ is manual/PA/CST, beam_type is either Gaussian (for PA) or manual (for CST) # frac_tol_conv=0.005 for Fir
                       N_fidu_types=Nft, N_pert_types=Npt, 
                       N_pbws_pert=30, per_channel_systematic=None,
                       PA_dist="column", plot_qty="P",
                            
                       layer_foregrounds=True, pointing_error=[12.,0.,-30.])