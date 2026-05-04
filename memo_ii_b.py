from forecasting_pipeline import *

# >>>>> where to look for CST <<<<<
CST_dir="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams/CHORD_feed_tilts_integ_dom_600/farfield_" # local
# CST_fir="/home/sophiaru/scratch/CHORD_CST/farfield_" # Fir

# >>>>> details of fidu and syst CST cases <<<<< 
fiduname="fiducial/"
systnames = [ "+0.0+0.0+0.5", "+0.0+0.0+1.0", "+0.0+0.0+1.5", 
              "+0.0+0.5+0.0", "+0.0+1.0+0.0", "+0.0+1.5+0.0", 
              "+0.0+0.0-0.5", "+0.0+0.0-1.0", "+0.0+0.0-1.5", 
              "+0.0-0.5+0.0", "+0.0-1.0+0.0", "+0.0-1.5+0.0" ]
all_syst_dirs=[sn+"_deg" for sn in systnames]

N_systs_use=len(systnames) # exhaustive case
N_systs_use=3 # pared-down case for debugging

# configure pointing errors
base_pointing_error=[1.2,-0.7,0.4]
base_seed=270426
meta_rng=np.random.default_rng(base_seed-1)
N_ptg_errs_per_CST=meta_rng.integers(low=0,high=6, size=N_systs_use,endpoint=False)
pointingerrs=[pointing_family(base_pointing_error,Ni,seed=base_seed+i) for i,Ni in enumerate(N_ptg_errs_per_CST)]
print("pointingerrs=",pointingerrs)

with open("ptg_err.json", "w") as f:
   json.dump(pointingerrs, f, indent=2, default=str)

power_comparison_plots(redo_window_calc=True, redo_box_calc=False, alr_imp_CST=True,
                       mode="pathfinder", nu_ctr=600.*u.MHz, frac_tol_conv=0.5, which_power="P",
                       categ="PA-CST", PA_dist="frame", pointing_errors=pointingerrs[:N_systs_use],
                       CST_lo=0.58*u.GHz,CST_hi=0.62*u.GHz,CST_deltanu=2e-4*u.GHz,
                       beam_sim_directory=CST_dir, CST_f_head_fidu=fiduname, CST_f_head_syst=all_syst_dirs)