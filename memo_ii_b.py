from forecasting_pipeline import *

CST_dir="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams/CHORD_CST_600/" # local
# CST_dir="/home/sophiaru/scratch/pipeline/CST/" # Fir
ff="farfield_(f="
pointingerrs=[ [1.5,-0.7,0.4],   [0.,2.6,-1.5],   [-0.9,0.1,1.2],  [0.9,-0.3,1.5], [2.7,-0.6,0.],
               [-0.1,0.1,0.2],   [-1.1,0.9,-1.2], [0.2,-0.2,-0.3], [-0.1,0.,0.1],  [5.,-0.1,1.8],
               [-0.3,-0.2,-0.4], [0.,0.,-0.3],    [2.1,0.1,-0.5],  [3.2,-0.1,2.3], [4.4,-2.1,-0.7] ]

power_comparison_plots(redo_window_calc=False, redo_box_calc=False, alr_imp_CST=True,
                       mode="pathfinder", nu_ctr=600.*u.MHz, layer_foregrounds=True,
                       frac_tol_conv=0.1, categ="PACST", PA_dist="frame",             # categ is manual/PA/CST/PACST, beam_type is either Gaussian (for PA) or manual (for CST) # frac_tol_conv=0.005 for Fir
                       N_PA_CST_types=6, pointing_errors=pointingerrs[:6],
                       CST_lo=0.58*u.GHz,CST_hi=0.62*u.GHz,CST_deltanu=2e-4*u.GHz,    # 580-620 MHz = 0.58-0.62 GHz; 200 kHz = 0.2 MHz = 2e-4 GHz
                       beam_sim_directory=CST_dir,
                       CST_f_head_fidu="fiducial/"+ff,CST_f_head_real="fiducial/"+ff,CST_f_head_thgt="feed_tilt/"+ff) # local
                    #    CST_f_head_fidu="fidu/"+ff,CST_f_head_real="fidu/"+ff,CST_f_head_thgt="syst/"+ff) # Fir