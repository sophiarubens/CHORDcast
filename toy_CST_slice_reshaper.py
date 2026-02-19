import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata as gd

theta=np.load("CST_theta.npy")
phi=np.load("CST_phi.npy")
uninterp_slice_pol1=np.load("CST_power.npy") # just to validate my projection strategy I'll form a power beam from two copies of one polarization and then, once that's squared away, go back to my main script and do things fully

print("theta.shape=",theta.shape)
print("phi.shape=",phi.shape)
print("uninterp_slice_pol1.shape=",uninterp_slice_pol1.shape)

# alphamax= 7561.531854117401 # borrowed from the final slice from the other script
alphamax=1
Npix=256
alpha_vec=np.linspace(-alphamax,alphamax,Npix)
alpha_grid_x,alpha_grid_y=np.meshgrid(alpha_vec,alpha_vec,indexing="ij")
alpha_grid_points=np.array([alpha_grid_x.ravel(),alpha_grid_y.ravel()]).T

safe_1_over_cos_phi=np.zeros_like(phi)
nonzero_angles=np.nonzero(phi!=0)
safe_1_over_cos_phi[nonzero_angles]=1/np.cos(phi[nonzero_angles])
safe_1_over_sin_phi=np.zeros_like(phi)
safe_1_over_sin_phi[nonzero_angles]=1/np.sin(phi[nonzero_angles])

sky_angle_x=np.sin(theta)*np.cos(phi)
sky_angle_y=np.sin(theta)*np.sin(phi)

sky_angle_points=np.array([sky_angle_x,sky_angle_y]).T

pol1_interpolated=gd(sky_angle_points,uninterp_slice_pol1,
                             alpha_grid_points,method="nearest")

pseudo_power=pol1_interpolated**2
pseudo_power=np.reshape(pseudo_power,(Npix,Npix))

plt.figure(layout="constrained")
plt.imshow(pseudo_power,origin="lower") #,extent=[alpha_vec[0],alpha_vec[-1],0,Nfreqs-1])
plt.xlabel("x index ")
plt.ylabel("y index")
plt.title("final slice xy cut")
plt.colorbar()
plt.savefig("CST_beam_one_slice.png")