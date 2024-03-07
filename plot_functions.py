##### "Halo mass function"(lhm) and "Stellar mass function"(lm) from a .hdf5 file. #####

### FILES ###
file_shark = '/home/olivia/galform_calib/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5' # redshift 0
file_galform = '/home/olivia/galform_calib/galform_output/galaxies.hdf5'

### COSMOLOGY ###
#cosmo_shark = h5ls galaxies.hdf5/cosmology

# Histograms

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import style

plt.style.use(style.style1)

#f = h5py.File(file_shark,'r')
#run_info = f['run_info']
#volume = run_info['lbox'][()]**3


def plot_lm(model, dm,z=[0,1,2]):
    """Plot the lm for the different models and compare them all with the observations in the same plot for different redshifts.

    Args:
        model (array of str): name of the models
        dm (float): bin of mass for the model's histograms
        z (array of float or int): the different redshifts. Defaults to [0,1,2].
    """
    h = 0.6774
    for iz in z:
        fig,ax=plt.subplots()
        for mod in model:
            if mod == 'shark':
                if iz == 0:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5'
                if iz == 1:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/97/multiple_batches/galaxies.hdf5'
                if iz == 2:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/78/multiple_batches/galaxies.hdf5'

            if mod == 'galform':
                filename = '/home/olivia/galform_calib/galform_output/galaxies.hdf5'

            with h5py.File(filename,"r") as f:
                if mod == 'shark':
                    volume = 1.56250E+07 #f['run_info/lbox'][()]**3 # [Mpc/h]³ #! es 1e9 el 1.56250E+07 es el volumen de 1 subvolumen, 1e9/64=1.56250E+07
                    #mres = f['run_info/particle_mass'][()] # [Msun/h] #! Es 0, mirar por qué
                    mres = 9.97e9 # [Msun/h]
                    mvir = f['galaxies/mvir_hosthalo'][:] # Dark matter mass of the host halo in which this galaxy resides [Msun/h]
                    mstars = f['galaxies/mstars_bulge'][:] + f['galaxies/mstars_disk'][:] #[Msun/h]
                    #print(len(mstars))

                if mod == 'galform':
                    volume = 1.56250E+07 #1e09 # [Mpc/h]³ # more used-parameters
                    mres = 9.97e9 # [Msun/h]

                    mstars = f['Output001/mstars_bulge'][:] + f['Output001/mstars_disk'][:] #[Msun/h]
                    #print(len(mstars)) #! Hay muy pocas, porque tengo galform corrido en 1 solo subvolumen

                lm = np.log10(mstars)
                #print(np.shape(lm))
                ind = np.where(mvir>(mres*20)) # All of them are bigger
                #print(mres*20)
                lm = lm[ind]
                #print(mvir.min(), mvir.max())
                #print(np.shape(lm))
                #exit()
                #mmin = np.log10(mres*10)
                mmin = lm.min() #mres = galaxies.hdf5/run_info/particle_mass
                mmax = lm.max()
                edges = np.array(np.arange(mmin, mmax+dm, dm)) #from mmin to mmax with a dex bin
                hist = edges[1:]-0.5*dm
                nm = np.zeros(shape=len(hist))

                iline=0

                for mass in lm: # in log

                    for ie, edge in enumerate(edges[:-1]):
                        if mass >= edge and mass<edges[ie+1]:
                            nm[ie] = nm[ie] + 1

                    iline+=1

                #print(sum(nm))

                f.close

            yhist = np.log10(nm) - np.log10(dm) - np.log10(volume)

            ax.plot(hist, yhist, label = mod)
            plt.title('Stellar Mass Function z = ' + str(iz))

            plt.ylabel('log$_{10}$ (dn/dlog (M$_{*}$)/h$^{3}$ Mpc$^{-3}$)')
            plt.xlabel('log$_{10}$(M$_{*}$/h$^{-1}$ M$_{\\odot}$)')


        ### OBSERVATIONS ###
        # henriques: [h^-2Msun] If one wants to plot quantities without h in the units, then both observations and simulations should be divided by the h^2 of the simulation.
        # baldry: [Msun]
        if iz==0:
            # Observations henriques
            fileobs='../Obs_Data/smf/henriques_2014_z0_cha.txt'
            mass_low = np.loadtxt(fileobs, skiprows=5, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=5, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.log10(np.loadtxt(fileobs, skiprows=5, usecols=(2), unpack=True))#*dm)
            yhist_err = np.abs(np.log10(np.loadtxt(fileobs, skiprows=5, usecols=(3), unpack=True)))#*dm)

            plt.plot(hist_obs, yhist_obs, 'o', label = 'henriques_2014_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'orange')

            # Observations Baldry
            fileobs = '../Obs_Data/smf/baldry_2012_z0_cha.txt'
            mass = np.loadtxt(fileobs,skiprows = 3, usecols = (0), unpack = True) + np.log10(h)
            phi = np.loadtxt(fileobs,skiprows=3, usecols = (1), unpack = True)
            yhist_obs = np.log10(phi)-3*np.log10(h)-3
            yhist_err = np.abs(np.log10(np.loadtxt(fileobs,skiprows=5, usecols = (3), unpack = True))-3*np.log10(h)-3)

            plt.plot(mass, yhist_obs, 'o', label = 'baldry_2012_z' + str(iz))
            #plt.errorbar(mass, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'limegreen')


            # Observations Moustakas
            fileobs = '../Obs_Data/smf/moustakas_z0.01_z0.20.smf'
            mass_low = np.loadtxt(fileobs, skiprows=5, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=5, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.log10(np.loadtxt(fileobs,skiprows=5, usecols = (2), unpack = True))
            yhist_err = np.abs(np.log10(np.loadtxt(fileobs,skiprows=5, usecols = (3), unpack = True)))

            plt.plot(hist_obs, yhist_obs, 'o', label = 'moustakas_2013_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'red')

            # Observations Ilbert
            fileobs = '../Obs_Data/smf/muzzin_ilbert_z0.2_z0.5.smf'
            mass_low = np.loadtxt(fileobs, skiprows=4, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=4, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.loadtxt(fileobs,skiprows=4, usecols = (2), unpack = True)
            yhist_err = np.loadtxt(fileobs,skiprows=4, usecols = (3), unpack = True)

            plt.plot(hist_obs, yhist_obs, 'o', label = 'ilbert_2013_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'purple')


        if iz==1:
            # Observations Moustakas
            fileobs = '../Obs_Data/smf/moustakas_z0.80_z1.00.smf'
            mass_low = np.loadtxt(fileobs, skiprows=4, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=4, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.loadtxt(fileobs,skiprows=4, usecols = (2), unpack = True)
            yhist_err = np.abs(np.log10(np.loadtxt(fileobs,skiprows=5, usecols = (3), unpack = True)))
            plt.plot(hist_obs, yhist_obs, 'o', label = 'moustakas_2013_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'orange')

            # Observations Ilbert
            fileobs = '../Obs_Data/smf/muzzin_ilbert_z0.5_z1.1.smf'
            mass_low = np.loadtxt(fileobs, skiprows=4, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=4, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.loadtxt(fileobs,skiprows=4, usecols = (2), unpack = True)
            yhist_err = np.loadtxt(fileobs,skiprows=4, usecols = (3), unpack = True)

            plt.plot(hist_obs, yhist_obs, 'o', label = 'ilbert_2013_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'limegreen')


        if iz==2:
            # Observations Henriques
            fileobs='../Obs_Data/smf/henriques_2014_z2_cha.txt'
            mass_low = np.loadtxt(fileobs, skiprows=5, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=5, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.log10(np.loadtxt(fileobs, skiprows=5, usecols=(2), unpack=True))#*dm)
            yhist_err = np.abs(np.log10(np.loadtxt(fileobs, skiprows=5, usecols=(3), unpack=True)))#*dm)

            plt.plot(hist_obs, yhist_obs, 'o', label = 'henriques_2014_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'orange')


            # Observations Ilbert
            fileobs = '../Obs_Data/smf/muzzin_ilbert_z2.0_z2.5.smf'
            mass_low = np.loadtxt(fileobs, skiprows=4, usecols=(0), unpack=True) - np.log10(h)
            mass_high = np.loadtxt(fileobs, skiprows=4, usecols=(1), unpack=True) - np.log10(h)

            dm_obs = mass_high[1]-mass_low[1] # 0.25
            #print(dm_obs) # = 0.25
            hist_obs = mass_high - 0.5 * dm_obs
            yhist_obs = np.loadtxt(fileobs,skiprows=4, usecols = (2), unpack = True)
            yhist_err = np.loadtxt(fileobs,skiprows=4, usecols = (3), unpack = True)

            plt.plot(hist_obs, yhist_obs, 'o', label = 'ilbert_2013_z' + str(iz))
            #plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'limegreen')

        plt.legend()
        plt.savefig('/home/olivia/calibration/calibration/plots/lm_z'+ str(iz)+'.pdf')
        plt.show()

plot_lm(['shark'],0.25, z=[0,1,2])
#exit()

def plot_sfr(model, dsfr,z=[0,1,2]):
    """Plot the sfrf for the different models and compare them all with the observations in the same plot for different redshifts.

    Args:
        model (array of str): name of the models
        dsfr (float): bin of sfr for the model's histograms
        z (array of float or int): the different redshifts. Defaults to [0,1,2].
    """
    h = 0.6774
    hobs = 0.71

    for iz in z:
        fig,ax=plt.subplots()
        for mod in model:
            if mod == 'shark':
                if iz == 0:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5'
                if iz == 1:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/97/multiple_batches/galaxies.hdf5'
                if iz == 2:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/78/multiple_batches/galaxies.hdf5'

            if mod == 'galform':
                filename = '/home/olivia/calibration/calibration/galform_output/galaxies.hdf5'

            with h5py.File(filename,"r") as f:
                if mod == 'shark':
                    volume = 1.56250E+07 #f['run_info/lbox'][()]**3 # [Mpc/h]³
                    #mres = f['run_info/particle_mass'][()] # [Msun/h] #! Es 0, mirar por qué
                    mres = 9.97e9 # [Msun/h]
                    mvir = f['galaxies/mvir_hosthalo'][:] # Dark matter mass of the host halo in which this galaxy resides [Msun/h]
                    sfr = f['galaxies/sfr_burst'][:] + f['galaxies/sfr_disk'][:] #[Msun/Gyr/h]
                    mstars = f['galaxies/mstars_bulge'][:] + f['galaxies/mstars_disk'][:] #[Msun/h]

                    #print(len(sfr))

                if mod == 'galform':
                    volume = 1.56250E+07 #1e9 # [Mpc/h]³ # more used-parameters
                    mres = 9.97e9 # [Msun/h]

                    sfr = f['Output001/mstardot'][:] + f['Output001/mstardot_burst'][:] # disk and bulge respectively [Msun/Gyr/h]
                    mstars = f['Output001/mstars_bulge'][:] + f['Output001/mstars_disk'][:]  #[Msun/h]

                    #print('works', len(sfr))

                ind = np.where(mvir>mres*20)
                lsfr = np.log10(sfr[ind]) - np.log10(h) - 9 #[Msun/yr]

                sfrmin = lsfr.min()
                sfrmax = lsfr.max()
                edges = np.array(np.arange(sfrmin, sfrmax+dsfr, dsfr)) #from sfrmin to sfrmax with a dex bin
                hist = edges[1:]-0.5*dsfr
                nsfr = np.zeros(shape=len(hist))

                iline=0

                for starfr in lsfr: # in log

                    for ie, edge in enumerate(edges[:-1]):
                        if starfr >= edge and starfr<edges[ie+1]:
                            nsfr[ie] = nsfr[ie] + 1

                    iline+=1

                #print(sum(nm))

                f.close

            yhist = np.log10(nsfr) - np.log10(dsfr) - np.log10(volume) + 3*np.log10(h)

            ax.plot(hist, yhist, label = mod)
            plt.title('Star Formation Rate Function z = ' + str(iz))
            plt.ylabel('log$_{10}$ ($\\Phi$ [Mpc$^{-3}$ dex$^{-1}$])')
            plt.xlabel('log$_{10}$(SFR[M$_{\\odot}$ yr$^{-1}$ ])')

        ### OBSERVATIONS ###
        # [h^-2Msun] If one wants to plot quantities without h in the units, then both observations and simulations should be divided by the h^2 of the simulation.
        if iz==0:
            fileobs='../Obs_Data/sfrf/gruppioni_2015_z0.0-0.3_cha.txt'

        if iz==1:
            fileobs='../Obs_Data/sfrf/gruppioni_2015_z0.8-1.0_cha.txt'

        if iz==2:
            fileobs='../Obs_Data/sfrf/gruppioni_2015_z2.0-2.5_cha.txt'
        sfr_low = np.loadtxt(fileobs, skiprows=3, usecols=(0), unpack=True) # log(SFR/(Msun/yr))
        sfr_high = np.loadtxt(fileobs, skiprows=3, usecols=(1), unpack=True) # log(SFR/(Msun/yr))

        dsfr_obs = sfr_high[1]-sfr_low[1]
        #print(dm_obs) # = 0.25
        hist_obs = sfr_high - 0.5 * dsfr_obs
        yhist_obs = np.loadtxt(fileobs, skiprows=3, usecols=(2), unpack=True)
        yhist_err = np.loadtxt(fileobs, skiprows=3, usecols=(3), unpack=True)

        plt.plot(hist_obs, yhist_obs, 'o', label = 'gruppioni_2015_z~' + str(iz))
        plt.errorbar(hist_obs, yhist_obs, yhist_err,fmt='none', elinewidth=None, ecolor = 'orange')

        plt.xlim(-2,5)
        plt.legend()
        plt.savefig('/home/olivia/calibration/calibration/plots/sfr_z'+ str(iz)+'.pdf')
        plt.show()

plot_sfr(['shark',], 0.25, z = [0,1,2])
exit()


def plot_BHSM(model, dbulge, dbh, z=0):
    """Plot the BHSM relation for the different models and compare them all with the observations in the same plot for different redshifts.

    Args:
        model (array of str): name of the models
        z (array of float or int): the different redshifts. Defaults to [0,1,2].
    """
    h = 0.6774

    fig,ax=plt.subplots()
    for mod in model:
        if mod == 'shark':
            filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5'
        if mod == 'galform':
            filename = '/home/olivia/calibration/calibration/galform_output/galaxies.hdf5'

        with h5py.File(filename,"r") as f:
            if mod == 'shark':
                volume = 1.56250E+07 #f['run_info/lbox'][()]**3 # [Mpc/h]³
                #mres = f['run_info/particle_mass'][()] # [Msun/h] #! Es 0, mirar por qué
                mres = 9.97e9 # [Msun/h]
                mvir = f['galaxies/mvir_hosthalo'][:] # Dark matter mass of the host halo in which this galaxy resides [Msun/h]

                mbh = f['galaxies/m_bh'][:] #[Msun/h]
                m_bulge = f['galaxies/mstars_bulge'][:]

                #print(len(sfr))

            if mod == 'galform':
                volume = 1.56250E+07 #1e9 # [Mpc/h]³ # more used-parameters
                mres = 9.97e9 # [Msun/h]
                print('no data yet')
                exit()

                #print('works', len(sfr))

            #mmin = np.log10(mres)
            #indmass = np.where(np.log10(mbh)>mmin)

            #lmbh = np.log10(mbh[indmass])
            #lm_bulge = np.log10(m_bulge[indmass])
            ind = np.where(mvir>mres*20)
            lm_bulge = np.log10(m_bulge[ind]) - np.log10(h) #log[Msun]
            lmbh = np.log10(mbh[ind]) - np.log10(h) #log[Msun]

            f.close
        ind = np.where(lm_bulge!=float('-inf'))
        #print(ind)
        #exit()
        lm_bulge = lm_bulge[ind]
        lmbh = lmbh[ind]
        bulge_min = lm_bulge.min()
        bulge_max = lm_bulge.max()
        print(bulge_min,bulge_max)

        bulge_edges = np.array(np.arange(bulge_min, bulge_max+dbulge, dbulge)) #from sfrmin to sfrmax with a dex bin
        
        bh_min = lmbh.min()
        bh_max = lmbh.max()
        bh_edges = np.array(np.arange(bh_min, bh_max+dbh, dbh)) #from sfrmin to sfrmax with a dex bin
        print(bh_min, bh_max)
        H, xedges, yedges = np.histogram2d(lm_bulge, lmbh, bins=([bulge_edges,bh_edges]))
        ind = np.where(H>0)
        median = np.median(H[ind])
        percentile_25 = np.percentile(H[ind], 25)
        percentile_75 = np.percentile(H[ind], 75)
        percentiles = np.percentile(H[ind], [25,50,75])
        print(percentiles)
        #exit()
        #ax.scatter(lm_bulge, lmbh, label = mod)

        #plt.contourf(xedges[:-1], yedges[:-1], H.T, levels=20, cmap='viridis')
        # Plot contours for median and percentiles
        #contour = plt.contour(xedges[:-1], yedges[:-1], H.T, levels=percentiles, colors='black')
        #plt.clabel(contour, fmt='%1.2f%%', inline=True)
        #X, Y = np.meshgrid(xedges, yedges)
        #ax.pcolormesh(X, Y, H)
        #plt.contourf(xedges[:-1], yedges[:-1], H.T, levels=20, cmap='viridis')
        plt.contour(xedges[:-1], yedges[:-1], H.T, levels=[median], colors='r', linestyles='solid', linewidths=2)
        plt.contour(xedges[:-1], yedges[:-1], H.T, levels=[percentile_25, percentile_75], colors='b', linestyles='dashed', linewidths=2)

        plt.title('BH-bulge mass relation z = 0')
        plt.ylabel('log$_{10}$ (M$_{BH}$/M$_{\\odot}$)')
        plt.xlabel('log$_{10}$ (M$_{bulge}$/M$_{\\odot}$)')

        ### OBSERVATIONS ###
        # McConnel
        fileobs='../Obs_Data/bhsm/McConnell_Ma_2013_ascii.txt'

        mbh_obs = np.log10(np.loadtxt(fileobs, skiprows=17, usecols=(2), unpack=True)) # [Msun]
        m_bulge_obs = np.log10(np.loadtxt(fileobs, skiprows=17, usecols=(11), unpack=True)) # [Msun]

        plt.plot(m_bulge_obs, mbh_obs, 'o', label = 'McConnel+2013')

        plt.ylim(5,11)
        plt.xlim(8,13)
        plt.legend()
        plt.savefig('/home/olivia/calibration/calibration/plots/BHSM_z0.pdf')
        plt.show()

#plot_BHSM(['shark'], 0.25, 0.25)
#exit()
def plot_sizes(model,dmbulge, drbulge, dmdisk, drdisk,dmtotal, z=[0]):
    """Plot the size-stellar mass relation for disks and bulges for the different models and compare them all with the observations in the same plot for different redshifts.

    Args:
        model (array of str): name of the models
        z (array of float or int): the different redshifts. Defaults to [0,1,2].
    """

    h = 0.6774
    hobs = 0.71


    for iz in z:
        fig_bulge,ax_bulge=plt.subplots()
        plt.title('Size-stellar mass relation for bulges z = ' + str(iz))
        plt.ylabel('log$_{10}$ (r$_{*, bulge}$/ckpc)')
        plt.xlabel('log$_{10}$ (M$_{*, bulge}$/M$_{\\odot}$)')
        plt.xlim(8,12)
        plt.ylim(-0.5,2)
        fig_disk,ax_disk=plt.subplots()
        plt.title('Size-stellar mass relation for disks z = ' + str(iz))
        plt.ylabel('log$_{10}$ (r$_{*, disk}$/ckpc)')
        plt.xlabel('log$_{10}$ (M$_{*, disk}$/M$_{\\odot}$)')
        plt.xlim(8,12)
        plt.ylim(-0.5,2)
        
        fig_total,ax_total=plt.subplots()
        plt.title('Size-stellar mass relation z = ' + str(iz))
        plt.ylabel('log$_{10}$ (r$_{*, disk}$/ckpc)')
        plt.xlabel('log$_{10}$ (M$_{*}$/M$_{\\odot}$)')
        plt.xlim(8,12)
        plt.ylim(-0.2,1.6)
        for mod in model:
            if mod == 'shark':
                if iz == 0:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5'
                if iz == 1:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/97/multiple_batches/galaxies.hdf5'
                if iz == 2:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/78/multiple_batches/galaxies.hdf5'

            if mod == 'galform':
                filename = '/home/olivia/calibration/calibration/galform_output/galaxies.hdf5'

            with h5py.File(filename,"r") as f:
                if mod == 'shark':
                    volume = 1.56250E+07 #f['run_info/lbox'][()]**3 # [Mpc/h]³
                    #mres = f['run_info/particle_mass'][()] # [Msun/h] #! Es 0, mirar por qué
                    mres = 9.97e9 # [Msun/h]
                    mvir = f['galaxies/mvir_hosthalo'][:] # Dark matter mass of the host halo in which this galaxy resides [Msun/h]

                    rstar_bulge = f['galaxies/rstar_bulge'][:] #[cMpc/h]
                    rstar_disk = f['galaxies/rstar_disk'][:] #[cMpc/h]
                    mstars_bulge = f['galaxies/mstars_bulge'][:] #[Msun/h]
                    mstars_disk = f['galaxies/mstars_disk'][:] #[Msun/h]

                    #print(len(sfr))

                if mod == 'galform':
                    volume = 1.56250E+07 #1e9 # [Mpc/h]³ # more used-parameters
                    mres = 9.97e9 # [Msun/h]
                    print('no data yet')
                    exit()

                    #print('works', len(sfr))


                ind = np.where(mvir>mres*20)

                lrstar_bulge = np.log10(rstar_bulge[ind]) + 3 - np.log10(h) #[ckpc]
                lrstar_disk = np.log10(rstar_disk[ind]) + 3 - np.log10(h) #[ckpc]
                lmstars_bulge = np.log10(mstars_bulge[ind]) - np.log10(h) #[Msun]
                lmstars_disk = np.log10(mstars_disk[ind]) - np.log10(h) #[Msun]
                f.close

            ## BULGE

            ind = np.where(lrstar_bulge!=float('-inf'))
            #print(ind)
            #exit()
            lrstar_bulge = lrstar_bulge[ind]
            lmstar_bulge = lmstars_bulge[ind]

            # Mass
            mbulge_min = lmstar_bulge.min()
            mbulge_max = lmstar_bulge.max()

            mbulge_edges = np.array(np.arange(mbulge_min, mbulge_max+dmbulge, dmbulge))
            #print(mbulge_min,mbulge_max, mbulge_edges)

            # Radius
            rbulge_min = lrstar_bulge.min()
            rbulge_max = lrstar_bulge.max()
            rbulge_edges = np.array(np.arange(rbulge_min, rbulge_max+drbulge, drbulge))
            #print(rbulge_min, rbulge_max, rbulge_edges)


            H, xedges, yedges = np.histogram2d(lmstar_bulge, lrstar_bulge, bins=([mbulge_edges,rbulge_edges]))

            ind = np.where(H>0)
            median = np.median(H[ind])
            percentile_25 = np.percentile(H[ind], 25)
            percentile_75 = np.percentile(H[ind], 75)
            percentiles = np.percentile(H[ind], [25,50,75])
            #print(percentiles)

            ax_bulge.contour(xedges[:-1], yedges[:-1], H.T, levels=[median], colors='r', linestyles='solid', linewidths=2)
            ax_bulge.contour(xedges[:-1], yedges[:-1], H.T, levels=[percentile_25, percentile_75], colors='b', linestyles='dashed', linewidths=2)


            ## DISK

            ind = np.where(lrstar_disk!=float('-inf'))
            #print(ind)
            #exit()
            lrstar_disk = lrstar_disk[ind]
            lmstar_disk = lmstars_disk[ind]

            # Mass
            mdisk_min = lmstar_disk.min()
            mdisk_max = lmstar_disk.max()

            mdisk_edges = np.array(np.arange(mdisk_min, mdisk_max+dmdisk, dmdisk))
            #print(mbulge_min,mbulge_max, mbulge_edges)

            # Radius
            rdisk_min = lrstar_disk.min()
            rdisk_max = lrstar_disk.max()
            rdisk_edges = np.array(np.arange(rdisk_min, rdisk_max+drdisk, drdisk))
            #print(rbulge_min, rbulge_max, rbulge_edges)


            H, xedges, yedges = np.histogram2d(lmstar_disk, lrstar_disk, bins=([mdisk_edges,rdisk_edges]))

            ind = np.where(H>0)
            median = np.median(H[ind])
            percentile_25 = np.percentile(H[ind], 25)
            percentile_75 = np.percentile(H[ind], 75)
            percentiles = np.percentile(H[ind], [25,50,75])
            #print(percentiles)

            ax_disk.contour(xedges[:-1], yedges[:-1], H.T, levels=[median], colors='r', linestyles='solid', linewidths=2)
            ax_disk.contour(xedges[:-1], yedges[:-1], H.T, levels=[percentile_25, percentile_75], colors='b', linestyles='dashed', linewidths=2)

            ## TOTAL

            ind = np.where(lrstar_disk!=float('-inf'))
            #print(ind)
            #exit()
            lrstar_disk = lrstar_disk[ind]
            lmstar = lmstars_disk[ind] + lmstars_bulge[ind]
            ind = np.where(lmstar!=float('-inf'))
            lmstar = lmstar[ind]
            lrstar_disk = lrstar_disk[ind]

            # Mass
            m_min = lmstar.min()
            m_max = lmstar.max()

            m_edges = np.array(np.arange(m_min, m_max+dmtotal, dmtotal))
            #print(mbulge_min,mbulge_max, mbulge_edges)

            # Radius
            rdisk_min = lrstar_disk.min()
            rdisk_max = lrstar_disk.max()
            rdisk_edges = np.array(np.arange(rdisk_min, rdisk_max+drdisk, drdisk))
            #print(rbulge_min, rbulge_max, rbulge_edges)


            H, xedges, yedges = np.histogram2d(lmstar, lrstar_disk, bins=([m_edges,rdisk_edges]))

            ind = np.where(H>0)
            median = np.median(H[ind])
            percentile_25 = np.percentile(H[ind], 25)
            percentile_75 = np.percentile(H[ind], 75)
            percentiles = np.percentile(H[ind], [25,50,75])
            #print(percentiles)

            ax_total.contour(xedges[:-1], yedges[:-1], H.T, levels=[median], colors='r', linestyles='solid', linewidths=2)
            ax_total.contour(xedges[:-1], yedges[:-1], H.T, levels=[percentile_25, percentile_75], colors='b', linestyles='dashed', linewidths=2)

            #ax1.scatter(lmstars_bulge, lrstar_bulge, label = mod)
            #ax2.scatter(lmstars_disk, lrstar_disk, label = mod)



        """
        ### OBSERVATIONS ###
        if iz==0:
            fileobs='data/gruppioni_2015_z0.0-0.3_cha.txt'

        if iz==1:
            fileobs='data/gruppioni_2015_z0.8-1.0_cha.txt'

        if iz==2:
            fileobs='data/gruppioni_2015_z2.0-2.5_cha.txt'
        sfr_low = np.loadtxt(fileobs, skiprows=3, usecols=(0), unpack=True) # log(SFR/(Msun/yr))
        sfr_high = np.loadtxt(fileobs, skiprows=3, usecols=(1), unpack=True) # log(SFR/(Msun/yr))

        plt.plot(hist_obs, yhist_obs, 'o', label = 'gruppioni_2015_z~' + str(iz))
        """

        fig_bulge.legend()
        fig_bulge.legend()
        fig_disk.savefig('/home/olivia/calibration/calibration/plots/sizes_bulge_z'+ str(iz)+'.pdf')
        fig_disk.savefig('/home/olivia/calibration/calibration/plots/sizes_disk_z'+ str(iz)+'.pdf')
        plt.show()

plot_sizes(['shark'], 0.25, 0.25, 0.25, 0.25, 0.35)
exit()

def plot_himf(model, dgas,z=[0]):
    """Plot the himf for the different models and compare them all with the observations in the same plot for different redshifts.

    Args:
        model (array of str): name of the models
        dgas (float): bin of gas mass for the model's histograms
        z (array of float or int): the different redshifts. Defaults to [0,1,2].
    """
    h = 0.6774
    for iz in z:
        fig,ax=plt.subplots()
        for mod in model:
            if mod == 'shark':
                if iz == 0:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5'
                if iz == 1:
                    filename = '/home/olivia/calibration/calibration//shark_output/UNIT-PNG/subf+subl+dhalos/97/multiple_batches/galaxies.hdf5'
                if iz == 2:
                    filename = '/home/olivia/calibration/calibration/shark_output/UNIT-PNG/subf+subl+dhalos/78/multiple_batches/galaxies.hdf5'

            if mod == 'galform':
                filename = '/home/olivia/calibration/calibration/galform_output/galaxies.hdf5'

            with h5py.File(filename,"r") as f:
                if mod == 'shark':
                    volume = 1.56250E+07 #f['run_info/lbox'][()]**3 # [Mpc/h]³ #! es 1e9 el 1.56250E+07 es el volumen de 1 subvolumen, 1e9/64=1.56250E+07
                    #mres = f['run_info/particle_mass'][()] # [Msun/h] #! Es 0, mirar por qué
                    mres = 9.97e9 # [Msun/h]
                    mvir = f['galaxies/mvir_hosthalo'][:] # Dark matter mass of the host halo in which this galaxy resides [Msun/h]

                    mstars = f['galaxies/mstars_bulge'][:] + f['galaxies/mstars_disk'][:] #[Msun/h]
                    #print(len(mstars))
                    mgas = f['galaxies/matom_bulge'][:] + f['galaxies/matom_disk'][:] #[Msun/h]


                if mod == 'galform':
                    volume = 1.56250E+07 #1e09 # [Mpc/h]³ # more used-parameters
                    mres = 9.97e9 # [Msun/h]

                    mstars = f['Output001/mstars_bulge'][:] + f['Output001/mstars_disk'][:] #[Msun/h]
                    #print(len(mstars)) #! Hay muy pocas, porque tengo galform corrido en 1 solo subvolumen


                ind = np.where(mvir>mres*20)

                lgas = np.log10(mgas[ind])

                gasmin = lgas.min()
                gasmax = lgas.max()


                edges = np.array(np.arange(gasmin, gasmax+dgas, dgas)) #from mmin to mmax with a dex bin
                hist = edges[1:]-0.5*dgas
                ngas = np.zeros(shape=len(hist))

                iline=0

                for hi in lgas: # in log

                    for ie, edge in enumerate(edges[:-1]):
                        if hi >= edge and hi<edges[ie+1]:
                            ngas[ie] = ngas[ie] + 1

                    iline+=1

                #print(sum(nm))

                f.close

            yhist = np.log10(ngas) - np.log10(dgas) - np.log10(volume)

            ax.plot(hist, yhist, label = mod)
            plt.title('HI Mass Function z = ' + str(iz))

            plt.ylabel('log$_{10}$ (dn/dlog (M$_{*}$)/h$^{3}$ Mpc$^{-3}$)')
            plt.xlabel('log$_{10}$(M$_{*}$/h$^{-1}$ M$_{\\odot}$)')


        ### OBSERVATIONS ###
        # Not yet
        plt.legend()
        plt.savefig('/home/olivia/calibration/calibration/plots/himf_z'+ str(iz)+'.pdf')
        plt.show()

plot_himf(['shark'], 0.25)



def plot_sfr_M(model, dm, dsfr, z=[0,1,2]): #, obsSFR, obsGSM, colsSFR,colsGSM,labelObs, outplot, verbose=False):

    '''
        Given log10(Mstar) and log10(sSFR)
        get the plots to compare log10(SFR) vs log10(Mstar).
        Get the GSMF and the SFRF plots.

        Given the observations, compare the plots with the observations too.

        Parameters
        ----------

        obsSFR : string
        Name of the input file for the SFR data observed.
        Expected histogram mode:
        with a column with the low value of the bin,
        a column with the high value of the bin,
        a column with the frequency in the bin,
        and a column with the error.
        These columns must be specify in the colsSFR variable.

        In text files (*.dat, *txt, *.cat), columns separated by ' '.
        In csv files (*.csv), columns separated by ','.

        obsGSM : string
        Name of the input file for the GSM data observed.

        Expected histogram mode:
        with a column with the low value of the bin,
        a column with the high value of the bin,
        a column with the frequency in the bin,
        and a column with the error.
        These columns must be specify in the colsGSM variable.

        In text files (*.dat, *txt, *.cat), columns separated by ' '.
        In csv files (*.csv), columns separated by ','.

        colsSFR : list
        Columns with the data required to do the observational histogram of the SFR.
        Expected: [ind_column1, ind_column2, ind_column3, ind_column4]
        where: column1 is the column with the low values of the bins, in Msun/yr,
                column2 with the high values of the bins, in Msun/yr,
                column3 with the frequency, in Mpc^-3 dex^-1
                column4 with the error, in Mpc^-3 dex^-1
        colsGSM :
        Columns with the data required to do the observational histogram of the GSM.
        Expected: [ind_column1, ind_column2, ind_column3, ind_column4]
        where: column1 is the column with the low values of the bins, in h^-2Msun,
                column2 with the high values of the bins, in h^-2Msun,
                column3 with the frequency, in h^-3 Mpc^-3,
                column4 with the error, in h^-3 Mpc^-3.

        labelObs : list of strings
        For the legend, add the name to cite the observational data source.
        ['GSM observed', 'SFR observed']

        outplot : string
        Name of the output file.
         Image-type files (*.pdf, *.jpg, ...)

        h0 : float
        If not None: value of h, H0=100h km/s/Mpc.
        volume : float
        Carlton model default value = 542.16^3 Mpc^3/h^3.
        table 1: https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4922B/abstract
        If not 542.16**3. : value of the simulation volume in Mpc^3/h^3
        verbose : boolean
        Yes = print out messages

        Returns
        -------
        plot(log10(SFR),log10(Mstar)), plot GSMF and plot SFRF,
        all three in one grid.
        Save it in the outplot path.
    '''

    h = 0.6774
    hobs = 0.71

    for iz in z:
        fig,ax=plt.subplots()
        for mod in model:
            if mod == 'shark':
                if iz == 0:
                    filename = '/home/olivia/galform_calib/shark_output/UNIT-PNG/subf+subl+dhalos/128/multiple_batches/galaxies.hdf5'
                if iz == 1:
                    filename = '/home/olivia/galform_calib/shark_output/UNIT-PNG/subf+subl+dhalos/97/multiple_batches/galaxies.hdf5'
                if iz == 2:
                    filename = '/home/olivia/galform_calib/shark_output/UNIT-PNG/subf+subl+dhalos/78/multiple_batches/galaxies.hdf5'

            if mod == 'galform':
                filename = '/home/olivia/galform_calib/galform_output/galaxies.hdf5'

            with h5py.File(filename,"r") as f:
                if mod == 'shark':
                    volume = 1.56250E+07 #f['run_info/lbox'][()]**3 # [Mpc/h]³
                    #mres = f['run_info/particle_mass'][()] # [Msun/h] #! Es 0, mirar por qué
                    mres = 9.97e9 # [Msun/h]
                    mvir = f['galaxies/mvir_hosthalo'][:] # Dark matter mass of the host halo in which this galaxy resides [Msun/h]
                    sfr = f['galaxies/sfr_burst'][:] + f['galaxies/sfr_disk'][:] #[Msun/Gyr/h]
                    mstars = f['galaxies/mstars_bulge'][:] + f['galaxies/mstars_disk'][:] #[Msun/h]

                    #print(len(sfr))

                if mod == 'galform':
                    volume = 1.56250E+07 #1e9 # [Mpc/h]³ # more used-parameters
                    mres = 9.97e9 # [Msun/h]

                    sfr = f['Output001/mstardot'][:] + f['Output001/mstardot_burst'][:] # disk and bulge respectively [Msun/Gyr/h]
                    mstars = f['Output001/mstars_bulge'][:] + f['Output001/mstars_disk'][:]  #[Msun/h]

                    #print('works', len(sfr))

                f.close()
            #Prepare the plot
            lsty = ['-',(0,(2,3))] # Line form

            nds = np.array([-2., -3., -4.]) # Contours values
            al = np.sort(nds)

            #!SFR = ['LC', 'avSFR']
            #!labels = ['average SFR', 'SFR from LC photons']

            cm = plt.get_cmap('tab10')  # Colour map to draw colours from
            color = []
            for ii in range(0, 10):
                col = cm(ii)
                color.append(col)  # col change for each iteration

            # Limit in mass:

            ind = np.where(mvir>(mres*20)) # All of them are bigger

            # Initialize GSMF (Galaxy Cosmological Mass Function)
            #!mmin = 10.3 # mass resolution 2.12 * 10**9 h0 M_sun (Baugh 2019)
            #!mmax = 15. #15.
            #!dm = 0.1
            #!mbins = np.arange(mmin, mmax, dm)
            #!mhist = mbins + dm * 0.5
            #!gsmf = np.zeros((len(mhist)))

            lm = np.log10(mstars)
            lm = lm[ind]

            mmin = lm.min() #mres = galaxies.hdf5/run_info/particle_mass
            mmax = lm.max()
            medges = np.array(np.arange(mmin, mmax+dm, dm)) #from mmin to mmax with a dex bin
            mhist = medges[1:]-0.5*dm
            nm = np.zeros(shape=len(mhist))

            # Initialize SFRF
            #!smin = -6.
            #!smax = 3.5
            #!ds = 0.1
            #!sbins = np.arange(smin, smax, ds)
            #!shist = sbins + ds * 0.5
            #!sfrf = np.zeros((len(shist)))
            lsfr = np.log10(sfr[ind]) - np.log10(h) - 9 #[Msun/yr]

            sfrmin = lsfr.min()
            sfrmax = lsfr.max()
            sedges = np.array(np.arange(sfrmin, sfrmax+dsfr, dsfr)) #from sfrmin to sfrmax with a dex bin
            shist = sedges[1:]-0.5*dsfr
            nsfr = np.zeros(shape=len(shist))

            # Initialize SFR vs M function
            lenm = len(mhist)
            lens = len(shist)
            smf = np.zeros((lens,lenm))
            #print(len(nm))
            #print(len(nsfr))
            #print(smf[61][40]) # [nsfr][nm]
            #exit()

            # Plots limits and style
            fig = plt.figure(figsize=(8.5, 9.))
            gs = gridspec.GridSpec(3, 3)
            gs.update(wspace=0., hspace=0.)
            ax = plt.subplot(gs[1:, :-1])

            # Fig. sSFR vs M
            xtit = "log$_{10}(\\rm M_{*}/M_{\\odot})$" #TODO: see if the units are ok
            ytit = "log$_{10}(\\rm SFR/M_{\\odot}yr^{-1})$"
            xmin = mmin; xmax = 11.6; ymin = sfrmin;  ymax = sfrmax
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_xlabel(xtit); ax.set_ylabel(ytit)

            # GSMF
            axm = plt.subplot(gs[0, :-1],sharex=ax)
            ytit="log$_{10}(\\Phi(M_*))$" ; axm.set_ylabel(ytit)
            axm.set_autoscale_on(False) ;  axm.minorticks_on()
            axm.set_ylim(-4.5,-2.)
            plt.setp(axm.get_xticklabels(), visible=False)

            # SFRF
            axs = plt.subplot(gs[1:, 2], sharey=ax)
            xtit = "log$_{10}(\\Phi(SFR))$"; axs.set_xlabel(xtit)
            axs.set_autoscale_on(False); axs.minorticks_on()
            axs.set_xlim(-6.4, 0.0)
            start, end = axs.get_xlim()
            axs.xaxis.set_ticks(np.arange(-6., end, 1.))
            plt.setp(axs.get_yticklabels(), visible=False)

            # Data Observations

            # SFR observed
            """
            ih = get_nheader(obsSFR)

            dataSFR = [0]*len(colsSFR)

            for ii, col in enumerate(colsSFR):
                #print(ii,col,colsSFR[ii])
                data = np.loadtxt(obsSFR,skiprows=ih, usecols=col, unpack=True)
                dataSFR[ii] = np.array(data)

            dex = dataSFR[1]-dataSFR[0]
            histSFR = dataSFR[1]-0.5*dex

            # GSM observed

            ih = get_nheader(obsGSM)

            dataGSM = [0]*len(colsGSM)

            for ii, col in enumerate(colsGSM):
                data = np.loadtxt(obsGSM,skiprows=ih, usecols=col, unpack=True)
                dataGSM[ii] = np.array(data)

            dex = dataGSM[1] - dataGSM[0]

            # Change the units from h^-2 Msun to Msun.
            histGSM = dataGSM[1] + 2*np.log10(h0) - 0.5*dex

            # Change the units from h^3 Mpc^-3 to Mpc^-3
            freqGSM = np.log10((dataGSM[2]))- 3 * np.log10(h0)

            """
            #! for ii, sfr in enumerate(SFR):
            #!tempfile = r"example_data/tmp_"+sfr+".dat"
            #!if not os.path.isfile(tempfile): continue

            #!ih = get_nheader(tempfile) # Number of lines in header

            # Jump the header and read the provided columns
            #!lms = np.loadtxt(tempfile, skiprows=ih, usecols=(0), unpack=True)
            #!lsfr = np.loadtxt(tempfile, skiprows=ih, usecols=(3), unpack=True)
            # loh12 = np.loadtxt(tempfile, skiprows=ih, usecols=(6), unpack=True). Not necessary in this plot


            # Make the histograms

            #!H, bins_edges = np.histogram(lm, bins=np.append(medges, mmax))
            #!gsmf = H / volume / dm  # In Mpc^3/h^3


            iline=0
            for mass in lm: # in log
                for ie, edge in enumerate(medges[:-1]):
                    if mass >= edge and mass<medges[ie+1]:
                        nm[ie] = nm[ie] + 1
                iline+=1


            gsmf = np.log10(nm) - np.log10(dm) - np.log10(volume)

            #! ax.plot(hist, yhist, label = mod)

            #!H, bins_edges = np.histogram(lsfr, bins=np.append(sedges, sfrmax))
            #!sfrf = H / volume / dsfr

            iline=0
            for starfr in lsfr: # in log
                for ie, edge in enumerate(sedges[:-1]):
                        if starfr >= edge and starfr<sedges[ie+1]:
                            nsfr[ie] = nsfr[ie] + 1
                iline+=1

            sfrf = np.log10(nsfr) - np.log10(dsfr) - np.log10(volume) + 3*np.log10(h)


            #!H, xedges, yedges = np.histogram2d(lsfr, lm, bins=([np.append(sedges, sfrmax),np.append(medges, mmax)]))
            H, xedges, yedges = np.histogram2d(lsfr, lm, bins=([sedges,medges]))

            #!smf = H / volume / dm / dsfr
            smf = np.log10(H) - np.log10(dsfr) - np.log10(volume) + 3*np.log10(h) - np.log10(dm) #TODO: verificar unidades

            """
            for starfr in lsfr: # in log
                for mass in lm:
                    for ise, sedge in enumerate(sedges[:-1]):
                        for ime, medge in enumerate(medges[:-1]):
                            if starfr >= sedge and starfr<sedges[ise+1]:
                                if mass>=medge and mass<medges[ime+1]:
                                    smf[ise][ime] = smf[ise][ime] + 1
                iline+=1
                print(iline)
            print(smf)
            exit()
            smf = np.log10(nsfr) - np.log10(dsfr) - np.log10(volume) + 3*np.log10(h) - np.log10(dm)
            """
            # Plot SMF vs SFR

            matplotlib.rcParams['contour.negative_linestyle'] = lsty[0]
            zz = np.zeros(shape=(len(shist), len(mhist))); zz.fill(-999)
            ind = np.where(smf > 0.)
            zz[ind] = np.log10(smf[ind])

            ind = np.where(zz > -999)

            if (np.shape(ind)[1] > 1):

                # Contours
                xx, yy = np.meshgrid(medges, sedges)
                # Here: How to find the levels of the data?
                cs = ax.contour(xx, yy, zz, levels=al, colors=color[0])
                ax.clabel(cs, inline=1, fontsize=10)
                #for i in range(len(labels)):
                #    cs.collections[i].set_label(labels[i])

            # Plot GSMF
            py = gsmf; ind = np.where(py > 0.)
            x = mhist[ind]; y = py[ind]#y = np.log10(py[ind])
            ind = np.where(y < 0.)

            axm.plot(x[ind], y[ind], color=color[0],
                    linestyle=lsty[0], label='GSMF')

            #! Plot observations GSMF
            #! axm.plot(histGSM, freqGSM, 'o', color=color[ii + 1])


            # Plot SFRF
            px = sfrf; ind = np.where(px > 0.)
            y = shist[ind]; x = np.log10(px[ind])
            ind = np.where(x < 0.)
            axs.plot(x[ind], y[ind], color=color[0],
                    linestyle=lsty[0], label='SFRF')

            #! Plot observations SFRF
            #! axs.plot(dataSFR[2], histSFR, 'o', color=color[ii + 2],
            #!     label=''+ labelObs[ii] +'')

            leg = axs.legend(bbox_to_anchor=(1.5, 1.4), fontsize='small',
                            handlelength=1.2, handletextpad=0.4)
            # for item in leg.legendHandles:
            # item.set_visible(True)
            leg.get_texts()
            leg.draw_frame(False)

            # for col,text in zip(color,leg.get_texts()):
            #   text.set_color(color)
            #  leg.draw_frame(False)

            plotf = '/home/olivia/galform_calib/plots/sfr_m_z'+ str(iz)+'.pdf'

            # Save figures
            print('Plot: {}'.format(plotf))
            fig.savefig(plotf)

            # os.remove(r"example_data/tmp_LC.dat")
            # os.remove(r"example_data/tmp_avSFR.dat")

#plot_sfr_M(['shark'], 0.25, 0.25, z = [0])
#exit()
