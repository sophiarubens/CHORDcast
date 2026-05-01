from forecasting_pipeline import *

CST_dir="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams/CHORD_CST_600/" # local
# CST_dir="/home/sophiaru/scratch/pipeline/CST/" # Fir
ff="farfield_(f="

base_pointing_error=[1.2,-0.7,0.4]
base_seed=270426
pointingerrs=[pointing_family(base_pointing_error,2,seed=base_seed),
              pointing_family(base_pointing_error,1,seed=base_seed+1),
              pointing_family(base_pointing_error,3,seed=base_seed+2),
              pointing_family(base_pointing_error,4,seed=base_seed+3),
              pointing_family(base_pointing_error,5,seed=base_seed+4),]
with open("ptg_err.json", "w") as f:
   json.dump(pointingerrs, f, indent=2, default=str)

N_points_use=2
fiduname="fiducial/"+ff  # local
systname="feed_tilt/"+ff # local
# fiduname="fidu/"+ff # Fir
# systname="syst/"+ff # Fir
all_syst_dirs=[systname for _ in range(N_points_use)]

power_comparison_plots(redo_window_calc=True, redo_box_calc=False, alr_imp_CST=True,
                       mode="pathfinder", nu_ctr=600.*u.MHz, frac_tol_conv=0.5, which_power="P",
                       categ="PA-CST", PA_dist="frame", pointing_errors=pointingerrs[:N_points_use],
                       CST_lo=0.58*u.GHz,CST_hi=0.62*u.GHz,CST_deltanu=2e-4*u.GHz,
                       beam_sim_directory=CST_dir, CST_f_head_fidu="fiducial/"+ff, CST_f_head_syst=all_syst_dirs)