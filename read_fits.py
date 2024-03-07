from astropy.table import Table
import numpy as np

t_sizes = Table.read('../../Downloads/gkvScienceCatv02.fits')
t_masses = Table.read('../../Downloads/StellarMassesGKVv24.fits')

R50 = t_sizes['R50']
logmstar = t_masses['logmstar'] #dex(Msun)
dellogmstar = t_masses['dellogmstar'] #dex(Msun)

uberID_size= t_sizes['uberID']
uberID_mass = t_masses['uberID']

Z_mass = t_masses['Z']
Z_size = t_sizes['Z']
ind1=np.where(Z_mass<0.0001)
ind2 = np.where(Z_size<0.0001)
mass = Z_mass[ind1]
size=Z_size[ind2]
ind=np.where(mass>0)
inds=np.where(size>0)
Zm = mass[ind]
Zs = size[inds]
#print(Zm)
#print(Zs)
#print(Zm.max(), Zm.min())
#print(Zs.max(), Zs.min())

uberID_mass=uberID_mass[ind]
uberID_size=uberID_size[inds]

logmstar = logmstar[ind]
dellogmstar = dellogmstar[ind]
R50 = R50[inds]

#print(logmstar)
#print(R50)

#exit()

r = []
m = []
ID = []

t = Table(names=('uberID_size','uberID_mass','Z_spectroscopic','Z_heliocentric', 'R50(arcsec)', 'logmstar', 'dellogmstar'))
#ind = [np.where(uberID_size==id) for id in uberID_mass]
#print(ind)
#r.append(R50[ind])
#exit()

for id in uberID_mass:
    ind = np.where(uberID_size==id)
    ind_mass = np.where(uberID_mass==id)
    if np.shape(ind)==(1,1):
        #print(logmstar[ind])
        t.add_row((uberID_size[ind], uberID_mass[ind_mass], Zs[ind],Zm[ind_mass], R50[ind], logmstar[ind_mass], dellogmstar[ind_mass]))
    #r.append(R50[ind])
    #m.append(logmstar[ind])
    #ID.append(uberID_size[ind])

#print(len(r), len(m), len(ID))


from astropy.cosmology import Planck15
redshifts = t['Z_spectroscopic']
R50 = t['R50(arcsec)']

# Calculate the comoving distance in Mpc
comoving_distances = Planck15.comoving_distance(redshifts).value

# Convert angular measurements from arcseconds to radians
angular_measurements_rad = np.deg2rad(R50 / 3600.0)

# Use the small-angle approximation to calculate physical distances
# Physical distance = comoving distance * angular measurement (in radians)
physical_distances_Mpc = comoving_distances * angular_measurements_rad

# Convert from Mpc to kpc
physical_distances_kpc = physical_distances_Mpc * 1e3

t.add_column(physical_distances_kpc, name="R50(kpc)")

t.write('size_SM.fits', overwrite=True)

