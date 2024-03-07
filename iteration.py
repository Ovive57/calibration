#### Basics ####
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
np.random.seed(42)

#### Parameter space ####
from skopt.space import Space
from skopt.sampler import Hammersly

#### Gaussian Processes ####
# SKLEARN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

#GPY
import GPy
#import gpry

#### Likelihood minimisation ####
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

# Time
import time

#### Pellejero-Ibáñez notes ####

# Let us assume there is a set of values {x} for which we have evaluated the
# function f ≡ log L(x), and another set, { y}, for which we would like to
# estimate the values f ∗ ≈ log L(y) without evaluating the likelihood at
# those points.

# It is customary to assume μ = μ∗ = 0 since the flexibility provided by
# the covariances is enough to model f ∗ arbitrarily well. It is also common
# to assume that the form of the covariances is the same.

# NOTE: kernel function, or covariance, K !!!!

# in certain regions of parameter space, the probability will be narrow and the
# predictions accurate, whereas in other regions (typically far from points
# where L has been evaluated), the uncertainty will be large.

#### Pellejero-Ibáñez ALGORITHM ####

# 1. Define a small set of points in parameter space: Hammersley, Latin hypercube, ...
# 2. Define the space and metric to be emulated. Depending on the kernel, in each step the sampling points, s, will be transformed in a way or another.
# 3. Build the emulator for log L. Sugestion: Gpy Python package
# 4. Fit a multivariate Gaussian to the emulated likelihood and estimate the mean and covariance. We use these parameters to estimate the kullnack-Lieber divergence.
# 5. Iterate adding Ni new points to our training set. Free parameters of the iterative process: Ni, alpha, num_restarts
    #if DL<0,1 for 5 consecutive steps: process converged, stop iterations
    #else: we sample Ni new training points, evaluate the full likelihood and start a new iteration


# 1. Define a small set of points in parameter space: Hammersley, Latin hypercube, ...

## CREATE PARAMETER SPACE

n_samples = 10 #number of initial points

space = Space([(0.5, 1.5), (0.5, 3.5)])#[(-10., 10.), (0., 15.)]) # Range the parameter values (min,max) [a_cool, gamma_sn] #! Must be float : prior (rango) planos o no planos? plano: todos con la misma probabilidad uniform prior

hammersly = Hammersly()
ps = hammersly.generate(space.dimensions, n_samples) # Parameter space

def parameter_space(n_samples, limits):
    space = Space(limits)
    hammersly = Hammersly()
    ps = hammersly.generate(space.dimensions, n_samples)
    return ps

#ps = parameter_space(n_samples, [(0.5, 1.5), (0.5, 3.5)] )

#print(space)
#print(np.shape(space))

# 2. Define the space and metric to be emulated. Depending on the kernel, in each step the sampling points, s, will be transformed in a way or another.

## EXAMPLE MODEL
n_data = 50 #lenght of the data that I will have from the model

def model(a,b,nvalues, ndata=50):
    """model that will be the semianalitical model

    Args:
        a (_type_): first parameter
        b (_type_): second parameter
        nvalues (int): number of properties that I want
        ndata (int): number of data for each property, "size of the column"

    Returns:
        _type_: the modelisation
    """
    np.random.seed(42)
    #model = [a**3+b**2+c, a+b**2+c**3]
    """
    random_array1 = np.random.uniform(a, b, size=(len(a),))
    random_array2 = np.random.uniform(b, c, size=(len(b),))
    random_array3 = np.random.uniform(c, a, size=(len(a),))
    random_array4 = np.random.uniform(b, c, size=(len(b),))
    mod = [random_array1, random_array2, random_array3, random_array4]
    """

    # Check if inputs are floats or arrays
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        # If inputs are floats, convert them to arrays
        a = np.full(1, a)
        b = np.full(1, b)
    mod = []

    for ai, bi in zip(a, b):
        inner_arrays = []
        for _ in range(nvalues):
            random_array = np.random.uniform(ai, bi, size=(ndata,))
            inner_arrays.append(random_array)
        mod.append(inner_arrays)

    return mod


## Model in all the parameter space:

#!!!!!!!!!!!!!! This is X !!!!!!!!!!!!!!!!!!!!!!
space_array = np.array(ps)
a_cool = space_array[:,0] # All options for parameter 1
gamma_sn = space_array[:,1] # All options for parameter 2
#print(np.shape(A))
#i = 0
#print(A)
#for parameter in space_array:
    #print(parameter)
    #i+=1
#print(i)
#exit()
#print(np.shape(space_array))
#exit()
mod = model(a_cool,gamma_sn,4) ## mod[i][j][k] where i is the number of points in the space and j in range nvalues, number of properties that we have, such as mass, luminosity, sfr, ... and k the result
#print(mod)
#print(np.shape(mod))
#exit()
## MY "OBSERVED" DATA:

obs_acool = 1.2 #0.8
obs_gammasn = 0.6 #3.2
#param_c = 0.8

obs_model = model(obs_acool,obs_gammasn, 4) ## obs_model[0][i] where i in range nvalues, number of properties that we have, such as mass, luminosity, sfr, ...

# Initial guess params:
#param_init_a = 0.
#param_init_b = 4.

#print(obs_model[0][4])
#print(obs_model)
#print(np.shape(obs_model))

## My study function, it will probably be the luminosity function or the sfmf, lmf...

def mass_function(prop): #, prop2, prop3, prop4, prop5):
    """function where we will calculate the likelihood

    Args:
        prop1 (_type_): first property
        prop2 (_type_): second property
        prop3 (_type_): third property
    """

    function = np.sin(prop) + np.random.normal(0,0.1, size=prop.shape) #**5+prop2**3+prop3+7*prop4+prop5

    return function



## Training data:

#! print(mod[99][4][49]) #[i][j][k] i: nº of points, j: nº of properties, k: nº of measures


### A column with the function evaluated in all my hammersley initial points:
"""
f = []
for i in range(n_samples):
    ff = mass_function(mod[i][0])
    f.append(ff)
f=np.array(f)
f = f.reshape(n_data,n_samples) 
#print(np.shape(f.reshape(50,100))) 
#! What the documentation calls y = f1 and X = prop1 : NO

## Function in the first point:

smass = mod[0][0].reshape(-1,1) # These are the values of the property at the first point of our Hammersley space
f1 = mass_function(smass)  # The function evaluated in the first point of our Hammersley space
#print(np.shape(f))
#print(np.shape(obs_model[0][0]))
#plt.plot(f1)


## Observed function:

obs_smass = obs_model[0][0].reshape(-1,1)
f_obs = mass_function(obs_smass) # The function evaluated in the observed point
#print(np.shape(f_obs))
#print(f_obs.reshape(-1,))
#print(np.shape(f_obs.reshape(-1,)))
#plt.show()
#print(np.shape(f_obs))
"""
#!!!!!! X is the space of parameters X = [[a1, b1],
# !                                       [a2, b2],
# !                                       [a3, b3],
# !                                       ...]

#!!!!! y is the log-likelihood value observed for the corresponding row of X
#!      y = [log_likelihood_1, log_likelihood_2, log_likelihood_3, ...]

#! It is clear that in this case X = space_array

def evaluate_likelihood(observed_function, theor_function, error):
    #TODO: mejorar esta función, es una prueba medio aleatoria, mirar apuntes de machine learning, tengo estas funciones bien
    """Calculate the X2 or the likelihood between the observed and the theoric function

    Args:
        observed_function (_type_): _description_
        theor_function (_type_): _description_
        error(float): error for observations and theoretical

    Returns:
        _type_: likelihood
    """
    likelihoods = np.exp(-(observed_function - theor_function) ** 2)
    likelihood = np.prod(likelihoods)
    chi2 = (observed_function-theor_function)**2/error**2 # *(error*(observed_function-theor_function))
    #chi2 = np.sum(chi2s)
    return (-1/2)*chi2

#likelihood = evaluate_likelihood(f_obs, f1, 0.2)
#print(likelihood)
#exit()
#!!!!! A TEST TO SEE IF IT WORKS FOR THE MOMENT !!!!!!!!!!

best_likelihood = np.inf
best_parameters = None
best_function = None

likelihoods = []
means=[]
obs_properties = model(obs_acool, obs_gammasn, 4)
obs_function = mass_function(obs_properties[:][0][0])

#plt.plot(f_obs, 'b-', label='Observed Function')

for parameters in space_array:
    properties = model(parameters[0], parameters[1], 4)
    #print(np.shape(properties[:][0][0])) #[50][1][4] -> [49][0][3]
    #print(properties[:][0][4])
    #exit()
    function = mass_function(properties[:][0][0]) # Taking only the first one as it is the mass for example
    likelihood = evaluate_likelihood(obs_function, function,0.2)
    mean_likelihood = np.mean(likelihood)
    means.append(mean_likelihood)
    covariance_likelihood = np.cov(likelihood)

    #print(likelihood)
    #print(mean_likelihood)
    #plt.plot(likelihood, 'r--', label='label to count')#, label='Function with Best Likelihood')
    likelihoods.append(np.sum(likelihood))

    #if likelihood<best_likelihood:
    #    best_likelihood = likelihood
    #    best_parameters = parameters
    #    best_function = function

#print("True parameters:", obs_acool, obs_gammasn)
#print("Best parameters without GP:", best_parameters)
#print("Best likelihood without GP:", best_likelihood)

#print("All parameters:", space_array)
#print(likelihoods)
#print(np.shape(likelihoods), np.shape(space_array))

#plt.xlabel('Index')
#plt.ylabel('Function Value')

#plt.plot(f_obs, 'b-', label='Observed Function')
#plt.plot(best_function, 'r--', label='Function with Best Likelihood')
#plt.xlabel('Index')
#plt.ylabel('Function Value')
#plt.title('Observed Function vs. Function with Best Likelihood')
#plt.legend()
#plt.show()
#plt.title('Likelihood of 5 points')
#plt.legend()
#plt.show()

def adquisition_function(likelihood_emulated, standard_deviation_emulated, free_parameter=0):
    """_summary_

    Note that, since the Normal distributed parameter is log L(y),
    σemul corresponds to the lognormal standard deviation
    σemul ≡ [e^σ − 1] e^{2χemul^2+σ].
    New parameter values for the training set are then sampled with a
    probability proportional to A(θ). We then recompute Lemul given the
    new set of L (with the old evaluations plus the new ones) and iterate
    this procedure.

    Args:
        likelihood_emulated (_type_): emulated likelihood
        standard_deviation_emulated (_type_): standard deviation as estimated by the Gaussian process
        free_parameter (_type_): a free parameter that balances sampling regions with high uncertainty. 
                    Defaults to 0 following Pellejero-Ibáñez (maximum exploitation over exploration).

    Returns:
        _type_: _description_
    """
    A = likelihood_emulated + free_parameter*standard_deviation_emulated
    return A

def convergence(nup, nuq, covp, covq, dimension):
    """To define convergence in our iterative method we make use of
    the Kullback-Liebler (KL) divergence.
    We employ the KL divergence between two consecutive steps in our
    iterative scheme to estimate how much information newly-added
    training points provide. If DKL → 0, these newly-added training
    points do not include any extra information and the process can
    be considered as converged. We set a threshold value of DKL ≈ 0.1,
    which roughly correspond to differences in the mean of the multivariate
    Gaussian posteriors of 5% of their standard deviation.
    DKL(P||Q) = 1 2 [ log |Σq | |Σp | − d + tr ( Σ−1 q Σp ) + (μq − μp)TΣ−1 q (μq − μp) ]


    Args:
        nup (array): mean first distribution
        nuq (array): mean second distribution
        covp (array): covariance first distribution
        covq (array): covariance second distribution
        dimension (array): number of dimensions of the parameter space

    Returns:
        _type_: convergence KL divergence
    

    """
    #print(np.shape(covp))
    #print(np.shape(covq))
    #print(np.shape(nup))
    #print(np.shape(nuq))
    D = (1/2)*(np.log(np.linalg.det(covp)/np.linalg.det(covq)) - dimension + np.matrix.trace(np.linalg.inv(covq)*covp) + np.transpose(nuq - nup)*np.linalg.inv(covq)*(nuq-nup)) #! No puedo restar la media training que tiene solo 5 con la media de toda la grid que son 100x100, quizá lo tengo que hacer al menos 2 veces antes de mirar esto
    return D






#exit()


#! I have the start, now the Gaussian Processes start!!!


## Gaussian Processes (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern)

## KERNEL:
# In terms of Gaussian Processes, a kernel is a function that specifies the degree of similarity between variables given their relative positions in parameter space.
# the choice of kernel (a.k.a. covariance function) determines almost all the generalisation properties of a GP model

# https://www.cs.toronto.edu/~duvenaud/cookbook/

#kern1 = Matern(nu=2.5, length_scale=1, length_scale_bounds=(1e-03, 10)) #*Values chosen as said by Rocher+2023

#kernel = GPy.kern.Matern52(input_dim = 2, lengthscale = 1) # input dimension, leghtscale, var = variance, I need to put the lenght_scale_bounds too, from 1e-03 to 10

# Define Matérn kernel with length scale between 10e-3 and 10
kernel = GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)

#kernel.lengthscale.set_prior(GPy.priors.LogGaussian(0, 1))
# When you don't put a prior on a parameter, you're not imposing any constraints on its possible values
# based on prior knowledge or beliefs.
# Instead, the parameter's value is determined solely from the observed data during the model fitting process.
# In this case, the model learns the parameter's value by maximizing the likelihood of the data given the model structure,
# without any additional constraints from a prior distribution.

# Set constraints on the length scale parameter
print('\n')
#! Limites siguiendo Rocher23:
kernel.lengthscale.constrain_bounded(1e-3, 10)

# Print kernel summary
print("\nKERNEL\n\n", kernel)

### Define the Gaussian process model with the desired kernel:
X_train = space_array #! Solo los parametros de 1 shark en la primera corrida
y_train = np.array(likelihoods).reshape(-1,1)
#print(np.shape(X_train), np.shape(y_train))
#print(X_train,y_train)
#exit()

gp_model = GPy.models.GPRegression(X_train,y_train, kernel=kernel)
print("\n\nGP MODEL\n ", gp_model)
#exit()



# Specify the likelihood function (Gaussian likelihood)
GP_likelihood = GPy.likelihoods.Gaussian()

# Set the likelihood function for the GP model
gp_model.likelihood = GP_likelihood

# Optimize the model using self.log_likelihood and self.log_likelihood_gradient,
# as well as self.priors. kwargs are passed to the optimizer.
# Specifying the optimization method (L-BFGS-B) and number of restarts
# L-BFGS-B stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bound constraints.
gp_model.optimize('lbfgsb', messages=False)
gp_model.optimize_restarts(num_restarts=5)# verbose=False, optimizer = 'lbfgsb')
# Print gp_model summary
print("\n\nGP OPTIMISED MODEL\n ",gp_model)
#exit()

# Optimize the model parameters to maximize the likelihood
#gp_model.optimize()

log_likelihood = gp_model.log_likelihood()
print("\nLikelihoods of the initial 10 points: ", likelihoods)
print("\nEMULATED LIKELIHOOD: ", log_likelihood)



# Extract optimized parameters
mean, variance = gp_model.predict(X_train)  # Mean of the Gaussian process
print("Emulated mean of the 10 training points: ", mean)
print("\nThe mean is similar to the likelihood values, so I assume this works.\n\n")
# Generate new input data points


# 4. Fit a multivariate Gaussian to the emulated likelihood and estimate the mean and covariance. We use these parameters to estimate the kullnack-Lieber divergence.
#! desde aquí de tiempo
# Define a grid of points in the two-dimensional input space
#! Volver a hacer un hammersley, con 5000 por ejemplo (si < 1s ok), hacer hammersley cambiando la seed: BUSCAR seed y buscar un número que vaya bien
start_time = time.time()

n_grid=1000000
grid = np.array(hammersly.generate(space.dimensions, n_grid)) # Grid for likelihoods
#print(np.shape(grid), type(grid))
acool_values = grid[:,0] #np.linspace(0.5, 1.5, 100)
gammasn_values = grid[:,1] # np.linspace(0.5, 3.5, 100)
#X_grid, Y_grid= np.meshgrid(acool_values, gammasn_values)
#XY_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
#print(type(XY_grid), np.shape(XY_grid))
#exit()

#coordinates: points where to predict the function.
coordinates = grid #XY_grid

res = gp_model.predict(coordinates)
evalues, cov = (res[0].T)[0], (res[1].T)[0] #! Usar máximo (COMPROBAR) de los likelihoods ind = np.max(): por lo menos 3-> máximo + peor error (?)
#cov = np.abs(cov)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time finding the maximum likelihood: ", elapsed_time)

#!hasta aquí de tiempo: pocos segundos debería ser, si no se rompe con 10000 pues 10000, cuantos más mejor
# Now: find proposed points:

#ind = np.where(np.abs(log_likelihood-evalues)<10e-3)
ind1 = np.argmax(np.abs(evalues))
ind2 = np.argmax(np.abs(cov))

print("\n\nPrediction maximum likelihood:\nlikelihood = ", evalues[ind1]," variance = " ,cov[ind1], "\nNew coordinate:",coordinates[ind1])
print("\n\nPrediction biggest error:\nlikelihood = ", evalues[ind2]," variance = " ,cov[ind2], "\nNew coordinate:",coordinates[ind2])

#print("\nEmulated prediction of the space: ", np.abs(evalues), "emulated variance", cov)

"""
multivariate_normal.fit
Fit a multivariate normal distribution to data.
        Parameters
        ----------
        x : ndarray (m, n)
            Data the distribution is fitted to. Must have two axes.
            The first axis of length `m` represents the number of vectors
            the distribution is fitted to. The second axis of length `n`
            determines the dimensionality of the fitted distribution.
        fix_mean : ndarray(n, )
            Fixed mean vector. Must have length `n`.
        fix_cov: ndarray (n, n)
            Fixed covariance matrix. Must have shape `(n, n)`.

        Returns
        -------
        mean : ndarray (n, )
            Maximum likelihood estimate of the mean vector
        cov : ndarray (n, n)
            Maximum likelihood estimate of the covariance matrix

"""
#print(np.shape(res[1]), res[1])
#print(np.shape(log_likelihood))
#print(log_likelihood)
#exit()
print(res[0])
mv_normal = multivariate_normal.fit(res[0]) #multivariate_normal.fit(log_likelihood)
#print(mv_normal)
print("\nMaximum likelihood estimate of the mean vector:", mv_normal[0], "\nMaximum likelihood estimate of the covariance matrix", mv_normal[1])

# 5. ITERATION
print(np.shape(X_train))
print(coordinates[ind2])
X = np.append(X_train,[coordinates[ind1]])
X = np.append(X, [coordinates[ind2]]).reshape(-1,2)

for n in range (100):
    likelihoods = []
    obs_properties = model(obs_acool, obs_gammasn, 4)
    obs_function = mass_function(obs_properties[:][0][0])

    #plt.plot(f_obs, 'b-', label='Observed Function')

    for parameters in X:
        properties = model(parameters[0], parameters[1], 4)
        function = mass_function(properties[:][0][0]) # Taking only the first one as it is the mass for example
        likelihood = evaluate_likelihood(obs_function, function,0.2)
        likelihoods.append(np.sum(likelihood))

    y = np.array(likelihoods).reshape(-1,1)
    
    gp_model = GPy.models.GPRegression(X,y, kernel=kernel)

    # Specify the likelihood function (Gaussian likelihood)
    GP_likelihood = GPy.likelihoods.Gaussian()

    # Set the likelihood function for the GP model
    gp_model.likelihood = GP_likelihood

    # Optimize the model using self.log_likelihood and self.log_likelihood_gradient,
    # as well as self.priors. kwargs are passed to the optimizer.
    # Specifying the optimization method (L-BFGS-B) and number of restarts
    # L-BFGS-B stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bound constraints.
    gp_model.optimize('lbfgsb', messages=False)
    gp_model.optimize_restarts(num_restarts=5, verbose=False, optimizer = 'lbfgsb')

    # Optimize the model parameters to maximize the likelihood
    #gp_model.optimize()

    log_likelihood = gp_model.log_likelihood()

    # Extract optimized parameters
    mean, variance = gp_model.predict(X)  # Mean of the Gaussian process


    # 4. Fit a multivariate Gaussian to the emulated likelihood and estimate the mean and covariance. We use these parameters to estimate the kullnack-Lieber divergence.
    #! desde aquí de tiempo
    # Define a grid of points in the two-dimensional input space
    #! Volver a hacer un hammersley, con 5000 por ejemplo (si < 1s ok), hacer hammersley cambiando la seed: BUSCAR seed y buscar un número que vaya bien
    start_time = time.time()

    n_grid=1000000
    grid = np.array(hammersly.generate(space.dimensions, n_grid)) # Grid for likelihoods
    acool_values = grid[:,0] #np.linspace(0.5, 1.5, 100)
    gammasn_values = grid[:,1] # np.linspace(0.5, 3.5, 100)

    #coordinates: points where to predict the function.
    coordinates = grid #XY_grid

    res = gp_model.predict(coordinates)
    evalues, cov = (res[0].T)[0], (res[1].T)[0] #! Usar máximo (COMPROBAR) de los likelihoods ind = np.max(): por lo menos 3-> máximo + peor error (?)
    #cov = np.abs(cov)

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time finding the maximum likelihood: ", elapsed_time)

    #!hasta aquí de tiempo: pocos segundos debería ser, si no se rompe con 10000 pues 10000, cuantos más mejor
    # Now: find proposed points:

    #ind = np.where(np.abs(log_likelihood-evalues)<10e-3)
    ind1 = np.argmax(np.abs(evalues))
    ind2 = np.argmax(cov)

    X = np.append(X, [coordinates[ind1]])
    X = np.append(X, [coordinates[ind2]]).reshape(-1,2)
    mv_normal_before = mv_normal
    print(res[0])
    print(evalues.reshape(-1,1))
    mv_normal = multivariate_normal.fit(evalues.reshape(-1,1)) #multivariate_normal.fit(log_likelihood)

    D = convergence(mv_normal_before[0], mv_normal[0], mv_normal_before[1],mv_normal[1], 2)
    if D>0.01:
        print("no converge aún")
    else:
        print("converge")
        print(coordinates[ind1], coordinates[ind2])
        break


#print(np.shape(evalues))
#mv_normal = multivariate_normal.pdf(evalues,mean=log_likelihood, cov=1471.588158209076)
#sample = mv_normal.rvs(size=1)
#print(mv_normal)
exit()
#! From here not too clear:

# Make predictions for the grid points
#! These are the real predictions I think
emulated_mean, emulated_variance = gp_model.predict(XY_grid) #! I think these are the likelihoods emulated
training_covariance = gp_model.posterior.covariance
print("Emulated mean of the space: ", emulated_mean, "emulated variance", emulated_variance)

exit()
print("\nCovariance of the 5 training points: ", training_covariance)


# Reshape mean and variance to match the grid shape
mean_plot = emulated_mean.reshape(X_grid.shape)
variance_plot = emulated_variance.reshape(X_grid.shape)

"""
# Plot the mean predictions
fig=plt.subplot()
plt.contourf(X_grid, Y_grid, mean, cmap='viridis')
plt.colorbar(label='Mean Prediction')

# Plot the confidence intervals
plt.contour(X_grid, Y_grid, variance, levels=[0.68, 0.95], colors='red', alpha=0.5)
"""

# Plot the mean predictions with contour lines
plt.contour(X_grid, Y_grid, mean_plot, levels=5)#, cmap='viridis')
#plt.colorbar(label='Mean Prediction')


# Plot the training data points
plt.scatter(X_train[:, 0], X_train[:, 1], c='k', label='Training Data')

plt.xlim(0.5, 1.5)
plt.ylim(0.5, 3.5)
plt.xlabel('a_cool')
plt.ylabel('gamma_SN')
plt.legend()

#plt.show()

# Get the kernel matrix: it is the covariance matrix without considering the observed data
# covariance matrix for the grid points based on the kernel function used in your Gaussian process model.
K_emulated = kernel.K(X_grid) # In this case it works because len(X_grid)=len(Y_grid)=100, if not: K_emulated = kernel.K(X_grid, Y_grid)

K_train = kernel.K(X_train)

#print(K_emulated)
#exit()

# Calculate the covariance matrix of the training points
#Cn = np.zeros((X_grid, Y_grid))
# Test to calculate the covariance matrix:
training_covariance2 = -np.linalg.inv((K_train+np.eye(len(X_train)))* gp_model.likelihood.variance) + np.eye(len(X_train))
difference = training_covariance-training_covariance2

print(difference)
#exit()
#print(np.shape(K_emulated), np.shape(gp_model.likelihood.variance), np.shape(np.eye(len(X_grid))))
emulated_covariance = -np.linalg.inv((K_emulated+np.eye(len(X_grid)))*gp_model.likelihood.variance)+ np.eye(len(X_grid))
print("\nEmulated covariance matrix of the 5 trainig points",training_covariance2)
print("\nThe training covariances are similar, so I assume this works.\n\n")

print("\nThe covariance matrix of the grid points",emulated_covariance)

#print(np.shape(emulated_covariance), np.shape(emulated_mean))
# Sample from the multivariate Gaussian distribution
#! Me he quedado aquí, esto va bien yo creo
#print(np.shape(emulated_mean), np.shape(emulated_mean.reshape(-1,1)), np.shape(emulated_mean.reshape(-1,1).flatten()), np.shape(emulated_covariance), np.shape(training_covariance2))
emulated_mean=emulated_mean.flatten()
print(len(emulated_mean))
mv_normal = multivariate_normal(mean=emulated_mean, cov=emulated_covariance.flatten())
sample = mv_normal.rvs(size=1)
print(sample, np.shape(sample), len(sample))
#print(X_new)

#conv = convergence(nup=training_mean,nuq=emulated_mean, covp=training_covariance, covq=emulated_covariance,dimension=2)
#print(conv)

#! NO MIRAR A PARTIR DE AQUÍ


def likelihood(ker, prop, function, prop_observed, f_observed):
    # Train GP model with current parameter values
    gp_model = GaussianProcessRegressor(kernel=ker)
    gp_model.fit(prop, function)

    # Predict outputs using the GP model
    y_pred, _ = gp_model.predict(prop_observed, return_std=True)  # Use X_observed here

    # Calculate likelihood (e.g., negative log likelihood)
    negative_log_likelihood = -np.sum((f_observed - y_pred)**2)  # Adjust as needed

    return negative_log_likelihood

params = np.array([param_init_a, param_init_b])

for i in range(n_samples):
    result = minimize(likelihood(kernel, mod[i][0].reshape(-1,1), mass_function(mod[i][0].reshape(-1,1)), obs_smass, f_obs), params, bounds=[(-10., 10.), (0., 15.)])
    params = result.x

optimized_param1, optimized_param2, optimized_param3 = params
print("Optimized parameter 1:", optimized_param1)
print("Optimized parameter 2:", optimized_param2)
print("Optimized parameter 3:", optimized_param3)

exit()


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
#print(np.shape(f), np.shape(f_obs))
gp.fit(smass, f1) #Fit Gaussian process regression model.
#gp.log_marginal_likelihood() #Return log-marginal likelihood of theta for training data.
#gp.score(f, f_obs, sample_weight=None) #Return the coefficient of determination of the prediction.






exit()

y_pred, y_std = gp.predict(obs_smass, return_std=True)
#plt.scatter(obs_prop, y_pred)
#plt.show()
#exit()
# Plot the results
plt.figure(figsize=(10, 6))
#plt.scatter(prop1, f1, color='blue', label='Observations')
plt.plot(obs_smass, y_pred, color='red', label='GP Prediction')
plt.fill_between(obs_smass.flatten(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.2, color='red', label='95% Confidence Interval')
plt.xlabel('data 1 point')
plt.ylabel('function')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

