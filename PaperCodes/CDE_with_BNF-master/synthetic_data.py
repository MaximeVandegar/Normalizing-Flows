import tensorflow as tf
from scipy import stats
import numpy as np

def gen_synthetic_data(dim=2, n_pts=10000, bimodal=False, heteroscedastic=True,
        asymetric=False):
    """gen_synthetic_data generates synthetic data with 1D output which is
    bimodal and heteroscedastic.

    Args:
        dim: dimensionality of the input
        n_pts: number of points to generate
        bimodal: true to generate bimodal data
        heteroscedastic: true to generate heteroscedastic data
        asymetric: set True to have noise have asymetric tails
    """
    ### Generate fake data
    bounds = [-np.pi,np.pi]
    range_size = bounds[1]-bounds[0]
    if heteroscedastic:
        noise_scale = 3. # for noise
    else:
        noise_scale = 0. # for noise

    global_noise = 1.0 # additional homoscedastic gaussian noise
    signal_scale = 5.

    if bimodal:
        n_pts_mode = int(n_pts/2.)
    else:
        n_pts_mode = n_pts

    # Generate X for first half of data
    X = np.random.uniform(bounds[0],bounds[1],size=[n_pts_mode,dim])
    Y_std = noise_scale*abs(np.sin(X).prod(axis=1))+global_noise # Heteroscedastic noise
    if asymetric:
        Y = abs(np.random.normal(0.,abs(Y_std)))+signal_scale*np.sin(X).prod(axis=1)
    else:
        Y = np.random.normal(0.,abs(Y_std))+signal_scale*np.sin(X).prod(axis=1)

    # Generate data from second mode
    if bimodal:
        X_more = np.random.uniform(bounds[0],bounds[1],size=[n_pts_mode,dim])
        Y_std_more = noise_scale*abs(np.sin(X_more)).prod(axis=1)+global_noise
        # The bimodality arises from using 'abs(X_more)' rather than simply
        # X_more within sin
        if asymetric:
            Y_more = abs(np.random.normal(0., abs(Y_std_more)))+signal_scale*np.sin(abs(X_more).prod(axis=1))
        else:
            Y_more = np.random.normal(0., abs(Y_std_more))+signal_scale*np.sin(abs(X_more).prod(axis=1))
        # concatenate two datasets together for bimodal signal
        X, Y = np.array(list(X)+list(X_more)), np.array(list(Y)+list(Y_more))

    Y = Y.reshape([n_pts, 1])
    return X, Y

#### We define the parameters of the mixture globally for now.
n_clust = 4
prior_mu_std = 2. # was 1.
prior_std_scale = 5.5
n_samples = 10000
mix_props = np.random.dirichlet([18.5]*n_clust)# mixing proportions
mus = np.random.normal(loc=[0.,0.],scale=prior_mu_std,size=[n_clust,2])
cov_mats = np.random.gamma(prior_std_scale, scale=0.05,size=[n_clust,2,2]) 
cov_mats[:,0,0] = np.random.uniform(low=0.7,high=1.4,size=[n_clust])*cov_mats[:,1,1]
cov_mats[:,1,0] = np.random.uniform(low=0.2,high=0.4,size=[n_clust])*cov_mats[:,1,1]
def gen_mog_synthetic_data():
    """gen_mog_synthetic_data generates data from a 2D mixture of Gaussians. 
    """
    cov_mats[:,0,1]=cov_mats[:,1,0]
    #print("mus: ",mus)
    #print("cov_mats: ",cov_mats)
    #print("mix_props: ",mix_props)
    Y = []
    for (mix_props_i, mu_i,cov_mat_i) in zip(mix_props,mus,cov_mats):
        Y.extend(np.random.multivariate_normal(mu_i,cov_mat_i,size=[int(n_samples*mix_props_i)]))
    Y = np.array(Y)
    print("Y_shape",Y.shape)
    Y = np.array(Y).reshape([len(Y[:,0]),2])

    ## at this point we don't have any inputs, so X is an empty array.
    X = np.zeros([Y.shape[0],1], dtype=np.float32)
    return X, Y

def p_mixture(y):
    """p_mixture calculates the probility of samples provied undert the
    true, data generating mixture distribution.

    Returns:
        A vector of probabilities of the samples.
    """
    return sum(stats.multivariate_normal.pdf(
        x=y,mean=mu_i, cov= cov_i
    )*mix_props_i for (mix_props_i, mu_i,cov_i) in zip(mix_props,mus,cov_mats))
