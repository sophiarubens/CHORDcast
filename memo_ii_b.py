from forecasting_pipeline import *

CST_dir="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams/CHORD_CST_600/"
ff="farfield_(f="

# per-antenna Gaussian: Nft has all elements >=1; Npt has all elements >=0 [array-like of options]
# per-antenna CST: N_PA_CST_types >=1 [!!scalar. makes more sense to handle this way, even if the philosophy is different from ^]

power_comparison_plots(redo_window_calc=True, redo_box_calc=True,
                       mode="pathfinder", nu_ctr=600.*u.MHz, layer_foregrounds=True,
                       frac_tol_conv=0.05, categ="PA",                                # categ is manual/PA/CST, beam_type is either Gaussian (for PA) or manual (for CST) # frac_tol_conv=0.005 for Fir
                       N_PA_CST_types=3, pointing_error=[1.5,-0.7,0.4],
                       CST_lo=0.58*u.GHz,CST_hi=0.62*u.GHz,CST_deltanu=2e-4*u.GHz,    # 580-620 MHz = 0.58-0.62 GHz; 200 kHz = 0.2 MHz = 2e-4 GHz
                       beam_sim_directory=CST_dir,
                       CST_f_head_fidu="fiducial/"+ff,CST_f_head_real="fiducial/"+ff,CST_f_head_thgt="feed_tilt/"+ff)