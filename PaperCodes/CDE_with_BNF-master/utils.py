from __future__ import division
import io
from pathlib import Path
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import stats

q_var_id = 0
def set_q(name=None, mu_prior=0., sigma_prior=1, mu_init_mu=None,
        mu_init_sigma=None, sigma_init=1e-5, n_samples=1, size=[1],
        save_summary=False, mu_shift=None, mvn=False, mu_init_values=None):
    """set_q creates a variational distribution, defines the KL, and defines
    samples.

    Args:
        name: name base for variables and summaries
        mu_prior: the mean of the prior
        sigma_prior: the std-dev of the prior
        mu_init_mu: the mean of the mu sampling distribution.
        mu_init_sigma: std-dev of mu sampling distribution.
        sigma_init: initial std-dev of q distributions
        n_samples: number of samples to draw, can be 0 (e.g. for weights
            where we want to use local reparameterization trick)
        size: size of the approximate posterior to create.
        mu_shift: we may be parameterizing partly on learned terms and
            partly on the output of a neural network, in which case we shift the
            mean of of the function by mu_shift
        mvn: Set true to use a multivariate normal variational distribution
        mu_values: use to supply specific initial weight mean values

    Returns:
        The parameters of the variational distribution,
        the samples drawn,
        The KL divergence between the approximate posterior and the prior.
    """
    # Set constant initializer for weight means
    if mu_init_values is not None:
        assert mu_init_mu is None and mu_init_sigma is None
        mu_initializer = tf.constant(mu_init_values)
    else:
        if mu_init_mu is None: mu_init_mu = mu_prior
        if mu_init_sigma is None: mu_init_sigma = sigma_prior
        mu_initializer = tf.constant(np.random.normal(
            size=size,loc=mu_init_mu, scale=mu_init_sigma
            ).astype(np.float32))

    if not sigma_init:
        sigma_init = sigma_prior

    if not name:
        name = "q_var_%d"%q_var_id
        q_var_id += 1

    if mvn:
        return set_mvn_q(name, mu_prior, sigma_prior, mu_init_mu,
            mu_init_sigma, sigma_init, n_samples, size, save_summary)

    mu = tf.get_variable(name+"_mu", initializer=mu_initializer)

    log_std = tf.get_variable(name+"_logstd", initializer=tf.constant(np.ones(
        size, dtype=np.float32)*un_softplus(sigma_init)))
    params = [mu, log_std]
    std = tf.nn.softplus(log_std)
    if mu_shift is not None:
        mu = mu+mu_shift
        # if we are shifting mu, we still need to ensure that std is the
        # same dimension in order to take the KL
        std = std[None, :]
        assert len(mu.shape) == len(std.shape)

        # We calculate the KL divergence between two normals in closed form.
        KL = KL_q_to_p(mu, std, mu_prior, sigma_prior)
    else:
        # We calculate the KL divergence between two normals in closed form.
        KL = KL_q_to_p(mu, std, mu_prior, sigma_prior)


    # We draw samples using the reparameterization trick
    if n_samples:
        eps = tf.random_normal(name=name+"_eps", mean=0., stddev=1.,
                shape=[n_samples]+size)
        samples = mu + std*eps
    else:
        samples = None

    # Establish Tensorboard summaries for the variational parameters.
    if save_summary:
        if len(size)==1 and size[0]==1 and mu_shift is None:
            tf.summary.scalar(name+"_mu", mu[0])
            tf.summary.scalar(name+"_sigma", std[0])
        else:
            tf.summary.histogram(name+"_mu", mu)
            tf.summary.histogram(name+"_sigma", std)
    return params, samples, KL

def set_mvn_q(name, mu_prior, sigma_prior, mu_init_mu,
        mu_init_sigma, sigma_init, n_samples, size, save_summary):
    """set_q creates a variational distribution, defines the KL, and defines
    samples.

    Args:
        name: name base for variables and summaries
        mu_prior: the mean of the prior
        sigma_prior: the std-dev of the prior
        mu_init_mu: the mean of the mu sampling distribution.
        mu_init_sigma: std-dev of mu sampling distribution.
        sigma_init: initial std-dev of q distributions
        n_samples: number of samples to draw, can be 0 (e.g. for weights
            where we want to use local reparameterization trick)
        size: size of the approximate posterior to create.

    Returns:
        The parameters of the variational distribution,
        the samples drawn,
        The KL divergence between the approximate posterior and the prior.
    """
    assert mu_init_mu==0
    mu = tf.get_variable(name+"_mu", initializer=tf.constant(np.random.normal(
        size=size,loc=mu_init_mu, scale=mu_init_sigma).astype(np.float32)))
    n_weights = size[0]*size[1]
    mu_flat = tf.reshape(mu, [n_weights])
    prior = tf.contrib.distributions.MultivariateNormalDiag(
            loc=np.zeros([n_weights],dtype=np.float32),
            scale_diag=np.ones(n_weights,
                dtype=np.float32)*sigma_prior
            )

    # this matrix will define the covariance matrix, but precise the
    # covariance matrix.
    W_cov = tf.get_variable(name+"_cov",
            initializer=tf.constant(np.diag([sigma_init]*n_weights).astype(np.float32))) # Lower Triangula
    W_cov_lt = W_cov*lt_mask(n_weights)
    # Cov(W) = W_cov_lt*W_cov_lt.T
    W_cov_true = tf.matmul(W_cov_lt,tf.transpose(W_cov_lt))
    W_dist = tf.contrib.distributions.MultivariateNormalTriL( loc=mu_flat,
            scale_tril=W_cov_lt)
    if n_samples:
        W_samples = W_dist.sample(sample_shape=[n_samples])
        W_samples = tf.reshape(W_samples, [n_samples,
            size[0],size[1]])
    else:
        W_samples = None

    KL = tf.contrib.distributions.kl_divergence(W_dist,prior)
    #KL = tf.contrib.distributions.kl(W_dist,prior)
    if save_summary:
        tf.summary.histogram(name+"_mu", mu)
        tf.summary.histogram(name+"_cov", W_cov)
    return [mu, W_cov], W_samples, KL, W_cov_true

def batch_norm(l, axis=0, decay=0.9, name=""):
    with tf.name_scope("Batch_Norm"):
        moving_mean = tf.get_variable(name+"_moving_mean",
                initializer=tf.constant(np.zeros([l.shape[2]],dtype=np.float32)))
        moving_var = tf.get_variable(name+"_moving_var",
                initializer=tf.constant(np.ones([l.shape[2]],dtype=np.float32)))
        tf.summary.histogram(name+"_moving_mean", moving_mean)
        tf.summary.histogram(name+"_moving_var", moving_var)
        mean, var = tf.nn.moments(l, axes=[axis])
        mean = tf.reduce_mean(mean, axis=0)
        var = tf.reduce_mean(var, axis=0)
        print "mean.shape", mean.shape
        mean_update = moving_mean.assign(decay*moving_mean + (1.-decay)*mean)
        var_update = moving_var.assign(decay*moving_var + (1.-decay)*var)
    return (l-moving_mean)/(tf.sqrt(moving_var+0.01)), [mean_update, var_update]

def un_softplus(x):
    return np.log(np.exp(x) - 1.0)

def plot_posterior_fcn_samples(net, epoch):
    plt.clf()
    n_pts_plot = 100
    x_plot = np.linspace(-3,3,n_pts_plot)
    y_plot = net.flows[0].project(x_plot*0.)
    for _ in range(15):
        y_plot_val = net.sess.run([y_plot], feed_dict={net.x:x_plot[:,None],
            })[0]
        plt.plot(x_plot, y_plot_val[0],c='g', linewidth=.5)
    plt.scatter(net.X_train, net.Y_train, c='k',s=50, marker='*')
    plt.ylim([min(net.Y_train)-15., max(net.Y_train)+15.])
    title="Samples_from_Posterior_Epoch%06d"%epoch
    plt.title(title)
    fn = net.summary_path +"_" + title+".png"
    buff = io.BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    plt.savefig(fn)
    print "finished plot"
    return buff



def plot_pruning2(net, fn_base):
    assert len(net.n_hidden_units) == 1
    assert net.outputs.shape[-1] == 1
    feed_dict = {net.x: net.X_train, net.y_std:net.Y_std}
    H1, W2_mu, W2_std, W1_mu, W1_std = net.sess.run([net.layers[1],
        net.weights['w_%d_mu'%2],
        net.weights['w_%d_std'%2],
        net.weights['w_%d_mu'%1],
        net.weights['w_%d_std'%1]],feed_dict=feed_dict)
    ### find ranges of Q distributions
    dim_x, dim_h = W1_mu.shape
    W1_min_pos, W1_max_pos= W1_mu.argmin(axis=0), W1_mu.argmax(axis=0)
    #print "W1_min_pos.shape", W1_min_pos.shape
    #print "W1_mu.shape", W1_mu.shape
    #print "H1.shape", H1.shape
    W1_min, W1_max = np.min(W1_mu[W1_min_pos,range(dim_h)]), np.max(W1_mu[W1_max_pos,range(dim_h)])
    W1_sd = np.max([W1_std[W1_min_pos,range(dim_h)], W1_std[W1_max_pos,range(dim_h)]])
    #W1_range = (W1_min-W1_sd, W1_max+W1_sd)
    W1_range = (-20, 20)
    print W1_range

    W2_min_pos, W2_max_pos= W2_mu.argmin(), W2_mu.argmax()
    W2_min, W2_max = W2_mu[W2_min_pos], W2_mu[W2_max_pos]
    W2_sd = max([W2_std[W2_min_pos], W2_std[W2_max_pos]])
    #W2_range = (W2_min-W2_sd, W2_max+W2_sd)
    W2_range = (-20, 20)

    act_range = [-20,20]
    n_pruned = 0
    for i in range(H1.shape[-1]):
        print "plotting stats for unit %d"%i
        plt.close('all')
        f, axarr = plt.subplots(3)
        ax1 = axarr[0]
        ax1.set_title(r"$q(V_{%d}|\phi)$"%i)
        mu, sd = W2_mu[i], W2_std[i]
        W2_x = np.linspace(W2_range[0], W2_range[1], 500)
        W2_p = stats.norm(loc=mu,scale=sd).pdf(W2_x)
        ax1.plot(W2_x, W2_p, c='k')

        ax2 = axarr[1]
        ax2.set_title(r"$q(W_{%d,:}|\phi)$"%i)
        W1_x = np.linspace(W1_range[0], W1_range[1], 500)
        for j in range(int(dim_x)):
            mu, sd = W1_mu[j,i], W1_std[j,i]
            if abs(sd - net.w_prior_sigma)/net.w_prior_sigma > 0.2:
                n_pruned += 1
            W1_p = stats.norm(loc=mu, scale=sd).pdf(W1_x)
            ax2.plot(W1_x, W1_p)

        ax3 = axarr[2]
        ax3.set_title(r"$V_{%d}*H_{%d}$"%(i,i))
        mu, sd = W2_mu[i], W2_std[i]
        W2_i_samples = np.random.normal(loc=mu,scale=sd,size=H1.shape[0])
        vals = W2_i_samples[:,None]*H1[:,:, i]
        vals = vals.reshape([np.prod(vals.shape)])
        ax3.hist(vals, bins=np.linspace(act_range[0], act_range[1],100))

        plt.tight_layout()
        plt.savefig(fn_base+"unit%02d.png"%i)
    return n_pruned

def plot_pruning(net, fn_base):
    assert len(net.n_hidden_units) == 1
    assert net.outputs.shape[-1] == 1
    feed_dict = {net.x: net.X_train, net.y_std:net.Y_std}
    H1, W2_mu, W2_std, W1_mu, W1_std = net.sess.run([net.layers[1],
        net.weights['w_%d_mu'%2],
        net.weights['w_%d_std'%2],
        net.weights['w_%d_mu'%1],
        net.weights['w_%d_std'%1]],feed_dict=feed_dict)
    for i in range(H1.shape[-1]):
        plt.close('all')
        f, axarr = plt.subplots(1, 3)
        title = "Hidden Unit %d ---  Weight to output: mu=%0.04f, sigma=%0.04f"%(i, W2_mu[i,0],
                W2_std[i,0])
        ax1 = axarr[0]
        ax1.set_title("activations")
        ax1.hist(H1[:, :, i].reshape([-1]), bins=np.linspace(np.min(H1), np.max(H1),15))

        ax2 = axarr[1]
        ax2.set_title("W1_mus")
        ax2.hist(W1_mu[:, i].reshape([-1]), bins=np.linspace(np.min(W1_mu),
            np.max(W1_mu),15))

        ax3 = axarr[2]
        ax3.set_title("W1_sigmas")
        ax3.hist(W1_std[:, i].reshape([-1]), bins=np.linspace(np.min(W1_std),
            np.max(W1_std),15))

        f.suptitle(title)
        plt.tight_layout()
        plt.savefig(fn_base+"unit%02d.png"%i)


def KL_q_to_p(mu_1, sig_1, mu_2, sig_2):
    """KL_q_to_p_val calculates the KL divergence between normal
    distributions q and p
    """
    # Ensure that the shapes of parameters of the approximate posterior are
    # the same.
    #assert mu_1.shape == sig_1.shape
    assert len(mu_1.shape) == len(sig_1.shape)
    KL_q_to_p_val = 0.0
    KL_q_to_p_val += tf.reduce_mean(tf.log(sig_2/sig_1))*tf.reduce_sum(tf.ones_like(mu_1))
    KL_q_to_p_val += tf.reduce_sum((sig_1**2 +
        (mu_1-mu_2)**2)/(2*sig_2**2))
    KL_q_to_p_val += -0.5*tf.reduce_sum(tf.ones_like(mu_1))
    #KL_q_to_p_val += -0.5*tf.cast(tf.reduce_prod(mu_1.shape),
    #    dtype='float32')
    return KL_q_to_p_val

def lt_mask(n):
    """lt_mask yields mask for lower triangular matrices

    This is taken from Amar's code
    """
    mask = np.zeros((n, n), dtype='float32')
    for i in xrange(n):
        for j in xrange(i+1):
            mask[i, j] = 1.
    return mask

def multilayer_perceptron_divergent(x, weights, biases, n_flows):
    """multilayer_perceptron_divergent creates an MLP

    the network has one shared hidden layer and second hidden
    layers for each of the normalizing flows.
    """
    # Hidden layer with RELU activation
    layer1_ = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer1_)

    out_layers = {}
    for i in range(n_flows):
        # Hidden layer with RELU activation
        layer2_ = tf.add(tf.matmul(layer_1, weights['w2_%02d'%i]), biases['b2_%02d'%i])
        layer_2 = tf.nn.relu(layer2_)
        # Output layer with linear activation
        out_layers["flow_%02d"%i] = tf.matmul(layer_2, weights['out_%02d'%i]) + biases['out_%02d'%i]
    return out_layers

def multilayer_perceptron(x, weights, biases, n_flows, n_layers=1,
        keep_prob=1., n_samples=1):
    """multilayer_perceptron creates an MLP to parameterize normalizing flows.

    Args:
        x: input tensor
        weights: dictionary of weights for each layer
        biases: dictionary of biases for each layer
        n_flows: number of stages in normalizing flow
        n_layers: number of layers
        keep_prob: the drop probability if using dropout.
        n_samples: the number of Monte Carlo dropout samples to use.

    Returns:
        a dictionary of output_layers, one for each flow, if n_samples is 1.
        if n_samples > 1, a list of such dictionaries
    """
    out_layers_samples = []
    for _ in range(n_samples):
        prev_layer = x
        for l in range(1,n_layers+1):
            layer = tf.add(tf.matmul(prev_layer, weights['w%d'%l]), biases['b%d'%l])
            # Hidden layers with RELU activation
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)
            prev_layer = layer

        out_layers = {}
        for i in range(n_flows):
            out_layers["flow_%02d"%i] = biases['out_%02d'%i]

            # if the parameters of the flow are parameterized by the output,
            # include them as such.
            if 'out_%02d'%i in weights:
                out_layers["flow_%02d"%i] += tf.matmul(prev_layer, weights['out_%02d'%i])
        out_layers_samples.append(out_layers)
    return out_layers_samples

def y_at_confidence(x, x_val, y, z_0, z_0_want, y_std, Y_std, max_y=100,min_y=-100):
    """y_at_confidence finds the y value corresponding to desired confidence interval.

    Args:
        x: the conditioning tensor variable
        x_val: the value to use
        y: tensor output.
        z_0: tensor for variable in unwarped space
        z_0_want: distance from median in SDs
        max_y: edge of range
        min_y: edge of range

    Returns:
        the location in input space and the corresponding y
    """
    ## perfrom binary search
    n_steps = 20
    for _ in range(n_steps):
        guess = np.array([[(max_y+min_y)/2.]])
        guess_val = z_0.eval(feed_dict={y:guess, x:x_val, y_std:Y_std})
        guess_val = np.mean(guess_val, axis=0)
        if guess_val > z_0_want:
            max_y = guess[0,0]
        else:
            min_y = guess[0,0]
    return guess, guess_val

# Plot the variance as function on input
def plot_label_predictive_distribution(net, y_len=500, return_buff=False,
        plot_aggregate=False, add_txt=""):
    """plot_predictive_distribution plots the predictive distribution of
    points in the test set.

    The temperature is the log likelihood

    Args:
        net: the model object
        y_len: number of y bins
        return_buff: buffer containing png data to return for Tensorboard
            logging.
        plot_aggregate: set true to aggregate individual predictive
            distributions.
        add_txt: additional text to tack onto plot title.

    """
    n_pts_plot = min([100, net.X_test.shape[0]])
    y_bounds = [np.min(net.Y_train), np.max(net.Y_train)]
    x_dim = net.X_test.shape[1]
    x_probe = np.tile( net.X_test[:n_pts_plot], [y_len,1])

    y_probe = np.arange(y_bounds[0],y_bounds[1],(y_bounds[1]- y_bounds[0])/y_len)
    y_probe = y_probe[:y_len]
    y_heatmap = y_probe.reshape([y_len,1])
    y_heatmap = np.tile(y_heatmap,[1,n_pts_plot])

    # evaluate predictive distribution
    nlog_ls = net.nlog_ls_eval if hasattr(net, 'nlog_ls_eval') else net.nlog_ls
    p_points = -nlog_ls.eval(feed_dict={
        net.x:x_probe.reshape([n_pts_plot*y_len,x_dim]),
        net.y:y_heatmap.reshape([n_pts_plot*y_len,1]),
        net.y_std:net.Y_std,
    })
    p_points = np.exp(p_points)
    p_labels = p_points.reshape([y_len,n_pts_plot])

    ### Check integral over y.  Should integrate to 1
    if False: # we skip this check"w
        dy = y_heatmap[1,1]-y_heatmap[0,0]
        py_int = np.sum(p_labels, axis=0)*dy*np.prod(net.Y_std)
        print "mean p %f, min %f, max %f"%(np.mean(py_int), np.min(py_int),
                np.max(py_int))

    ### Aggregate over all test points
    if plot_aggregate:
        p_labels = np.sum(p_labels, axis=1)

    # plot predictive distribution(s)
    plt.close('all')
    plt.figure(figsize=(10,5))
    y_probe_scaled = y_probe*net.Y_std +net.Y_mean
    if not plot_aggregate:
        for i in range(p_labels.shape[1]):
            plt.plot(y_probe_scaled, p_labels[:, i])
    else:
        plt.plot(y_probe_scaled, p_labels)
    plt.xlabel("y")
    plt.ylabel("p(y)")
    if plot_aggregate:
        plt.title("Predictive Distribution Averaged Over Test Set"+add_txt)
    else:
        plt.title("Predictive Distributions for Test Points"+add_txt)

    if return_buff:
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return buff
    else:
        plt.show()

# Plot the variance as function on input
def plot_toy_input_noise(net, title, y_len=50, epoch=0):
    """plot_predictive_distribution plots the predictive distribution of
    points in the test set.

    The temperature is the log likelihood

    Args:
        net: the model object
        y_len: number of y bins
        return_buff: buffer containing png data to return for Tensorboard
            logging.
        plot_aggregate: set true to aggregate individual predictive
            distributions.
        add_txt: additional text to tack onto plot title.

    """
    y_bounds = [np.min(net.Y_train)-0.7, np.max(net.Y_train)+0.7]
    x_len, x_dim = net.X_test.shape
    x_probe = np.tile( net.X_test[:x_len], [y_len,1])*0.

    y_probe = np.arange(y_bounds[0],y_bounds[1],(y_bounds[1]- y_bounds[0])/y_len)
    y_probe = y_probe[:y_len]
    y_heatmap = y_probe.reshape([y_len,1])
    y_heatmap = np.tile(y_heatmap,[1,x_len])

    # evaluate predictive distribution
    p_points = -net.nlog_ls_eval.eval(feed_dict={
        net.x:x_probe.reshape([x_len*y_len,x_dim]),
        net.y:y_heatmap.reshape([x_len*y_len,1]),
        net.y_std:net.Y_std,
    })
    p_points = np.exp(p_points)
    p_labels = p_points.reshape([y_len,x_len])
    p_labels = np.mean(p_labels, axis=1)


    # plot predictive distribution(s)
    plt.close('all')
    plt.figure(figsize=(10,5))
    y_probe_scaled = y_probe*net.Y_std +net.Y_mean
    n_bins = 100
    plt.hist(net.Y[:,0], bins=n_bins,
            weights=[max(p_labels)/float(len(net.Y[:,0]))]*net.Y.shape[0])
    plt.plot(y_probe_scaled, p_labels,c='k')
    plt.xlabel("y")
    plt.ylabel("p(y)")
    plt.title(title)

    fn = net.summary_path +"_" + title+"_epoch%04d"%epoch+ ".png"
    buff = io.BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    plt.savefig(fn)
    print "finished plot"
    return buff

# Plot the variance as function on input
def plot_predictive_distribution(net,
        x_len=90, y_len=120, x_label = "X", y_label="Y", x_pts_tr=None,
        y_pts_tr=None, x_pts_val=None, y_pts_val=None,
        return_buff=False, conf_interval=2., plot_pts=True,
        plot_title=None, ax=None, plot_post_pred_dist=True,
        plot_interval=False, plot_x_axis=False, plot_y_axis=False):
    """plot_predictive_distribution plots the predictive distribution as a heatmap.

    The temperature is the log likelihood

    Args:
        nlog_p: tensor for evaluation log likelihood
        x: x input tensor for caluclating nlog_p
        y: y input tensor for caluclating nlog_p
        y_scale: size of the y_range to evaluate
        x_len: number of x bins
        y_len: number of y bins
        plot_pts: if we are to plot datapoints and confidence intervals
    """
    x_bounds = [np.min(net.X_train)-0., np.max(net.X_train)+0.]
    y_bounds = [np.min(net.Y_train), np.max(net.Y_train)]
    x_probe = np.arange(x_bounds[0],x_bounds[1],(x_bounds[1]-x_bounds[0])/x_len)
    x_probe = x_probe[:x_len]
    x_heatmap = np.tile( x_probe, [y_len,1])

    y_heatmap = np.arange(y_bounds[0],y_bounds[1],(y_bounds[1]- y_bounds[0])/y_len)
    y_heatmap = y_heatmap[:y_len]
    y_heatmap = y_heatmap.reshape([y_len,1])
    y_heatmap = np.tile(y_heatmap,[1,x_len])

    # create heatmap
    nlog_ls = net.nlog_ls_eval if hasattr(net, 'nlog_ls_eval') and plot_post_pred_dist else net.nlog_ls
    p_points = -nlog_ls.eval(feed_dict={
        net.x:x_heatmap.reshape([x_len*y_len,1]),
        net.y:y_heatmap.reshape([x_len*y_len,1]),
        net.y_std:net.Y_std
    })

    p_points = np.exp(p_points)
    p_points = p_points.reshape([y_len,x_len])

    ## check integral over y.
    dy = y_heatmap[1,1]-y_heatmap[0,0]
    py_int = np.sum(p_points, axis=0)

    # plot heatmap
    if ax is None:
        plt.close('all')
        f, axes = plt.subplots(figsize=(10,5))
    else:
        axes=ax

    if True:
        heatmap = axes.pcolor(x_heatmap, y_heatmap, p_points,cmap='RdBu_r')
    else:
        heatmap = axes.pcolor(x_heatmap, y_heatmap, p_points,cmap='RdBu_r',
                vmin=0., vmax=0.4)

    font_size = 7
    if plot_x_axis:
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)
        axes.set_xlabel("x", fontdict={'fontsize':font_size})
    else:
        axes.set_xticks([])
    if plot_y_axis:
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)
        axes.set_ylabel("y", fontdict={'fontsize':font_size})
        #axes.set_ylabel("p(x)", fontdict={'fontsize':font_size})
    else:
        axes.set_yticks([])
    if ax is None:
        f.colorbar(heatmap)

    if plot_title is None:
        title = "p(Y|X) Learned with Input Noise"
    else:
        title = plot_title
    if ax is None:
        axes.set_title(title)

    # Plot training and Test points
    axes.axis([x_heatmap.min(), x_heatmap.max(), y_heatmap.min(), y_heatmap.max()])
    if plot_pts:
        n_pts_to_plot = 1500
        axes.scatter(net.X_train[:n_pts_to_plot],net.Y_train[:n_pts_to_plot],c='k',s=1.5, alpha=1.0)
        axes.scatter(net.X_test[:n_pts_to_plot],net.Y_test[:n_pts_to_plot],c='g',s=1.5, alpha=1.0)

    # Plot median and confidence intervals for one sample (there could be more)
    if plot_interval and hasattr(net, 'z_0'):
        z_0 = net.z_0
        if len(z_0.shape) != 3:
            y_proj_sd_u  = [y_at_confidence(net.x, np.array([[x_i]]), net.y, z_0,
                np.array([[conf_interval]]),net.y_std,
                net.Y_std)[0][0,0] for x_i in x_probe ]
            y_proj_med = [y_at_confidence(net.x, np.array([[x_i]]), net.y, z_0,
                np.array([[0.]]), net.y_std, net.Y_std)[0][0,0] for x_i in x_probe ]
            y_proj_sd_d = [y_at_confidence(net.x, np.array([[x_i]]), net.y, z_0,
                np.array([[-conf_interval]]), net.y_std,
                net.Y_std)[0][0,0] for x_i in x_probe]
            y_proj_med, y_proj_sd_u, y_proj_sd_d = np.array(y_proj_med), np.array(y_proj_sd_u), np.array(y_proj_sd_d)
            axes.plot(x_probe, y_proj_sd_d,c='k',linewidth=2.)
            axes.plot(x_probe, y_proj_med,c='k',linewidth=4.)
            axes.plot(x_probe, y_proj_sd_u,c='k',linewidth=2.)

    if return_buff:
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return buff
    else:
        return heatmap

# Plot the variance as function on input
def plot_density(flow, x_len=100, ax=None,series_label=None,
        plot_x_axis=False, plot_y_axis=False):
    """plot_density a probability density sampled from the prior over
    normalizing flows.

    Args:
        x_len: number of x bins
        ax: the axis to plot on.
    """
    axes=ax
    x_limit = 3.
    px_limit = .6
    x_bounds = [-x_limit, x_limit]
    x_probe = np.arange(x_bounds[0],x_bounds[1],(x_bounds[1]-x_bounds[0])/x_len)
    x_probe = x_probe[:x_len].reshape([x_len, 1])

    # evaluate p(x)
    nlog_ls = flow.nlog_ls
    p_x = -nlog_ls.eval(feed_dict={
        flow.x:x_probe,
    })

    p_x = np.exp(p_x)

    heatmap = axes.plot(x_probe[:,0], p_x[0], label=series_label)
    if plot_x_axis:
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        axes.set_xlabel("x", fontdict={'fontsize':18})
    else:
        axes.set_xticks([])
    if plot_y_axis:
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        axes.set_ylabel("p(x)", fontdict={'fontsize':18})
    else:
        axes.set_yticks([])

    # Plot training and Test points
    axes.axis([x_probe.min(), x_probe.max(), 0, px_limit])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plot_fig(extent=[-74.0702316,-73.7375154,40.5933855,40.8929847],
        alpha=1.0, chicago=False, axis=None):
    if axis is None: axis=plt
    if chicago:
        fn='../../chicago_data/osm_chicago.png'
    else:
        fn='../nyc/map40.8859847,-74.0682316_40.5933855,-73.7375154.png'
    fn_path = Path(fn)
    if not fn_path.is_file():
        fn=fn[3:]
    img_arr =np.flipud(plt.imread(fn))
    img_arr = rgb2gray(img_arr)
    axis.imshow(img_arr, origin='lower',
            extent=extent,
            alpha=1.0,cmap = plt.get_cmap('gray')
            )

def plot_heatmap(Y1, Y2, p_y, max_color=None,
        extent=[-74.0702316,-73.7375154,40.5933855,40.8929847],
        include_map=True, chicago=False, axis=None, fig=None, colorbar=True,
        tanh_b=0, tanh_m=1, tanh_max=1):
    if axis is None: axis=plt
    # Define opaque colormap
    cmap = plt.cm.YlOrRd
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)**(0.5)
    #my_cmap[:,-1] = np.sqrt(np.linspace(0, 1, cmap.N))
    #my_cmap[:,-1] = np.tanh(tanh_m*(np.linspace(0, 1, cmap.N)-0.5+tanh_b))
    #my_cmap[:,-1] = (my_cmap[:,-1]+1)/2.*tanh_max
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    if include_map: plot_fig(extent, chicago=chicago, axis=axis)
    heatmap = axis.pcolor(Y1,Y2,p_y,cmap=my_cmap, vmin=0.,
            vmax=max_color,edgecolors='none')
    if colorbar:
        fig.colorbar(heatmap)

def sample_heatmap_chicago(net, x, n_grid_pts, n_samples, y_probe, nlog_ls):
    p_y = np.zeros([n_grid_pts**2])
    for _ in range(n_samples):
        x_cpy = x.copy()
        # pick random point to fill unset dimensions in from
        # This ensures that these dimensions are sampled from the data
        # distribution.
        x_random = net.X[np.random.randint(net.X.shape[0])]
        if x_cpy.has_key('hour'):
            x_cpy['hour_vec'] = chicago.hour_to_value(x_cpy['hour'])
        else:
            idx = net.feature_idxs['hour']
            x_cpy['hour_vec'] = x_random[idx:idx+2]

        if x_cpy.has_key('month'):
            x_cpy['month_vec'] = chicago.month_to_value(x_cpy['month'])
        else:
            idx = net.feature_idxs['month']
            x_cpy['month_vec'] = x_random[idx:idx+2]


        if not x_cpy.has_key('crime'):
            idx = net.feature_idxs['crime']
            x_cpy['crime'] = x_random[idx:idx+4]

        # we check if year is included by checking the length of X
        if len(x_random) == 9: # year included
            x_vec = np.array([x_cpy['year']] + list(x_cpy['month_vec'])+
                list(x_cpy['hour_vec']) + list(x_cpy['crime']))
            if not x_cpy.has_key('year'):
                idx = net.feature_idxs['year']
                x_cpy['year'] = x_random[idx]
        else:
            x_vec = np.array(list(x_cpy['month_vec'])+
                list(x_cpy['hour_vec']) + list(x_cpy['crime']))
        x_scaled = x_to_scaled_value(net, x_vec)
        x_probe = np.ones([y_probe.shape[0],len(x_vec)],
            dtype=np.float32)*x_scaled[None]
        # add to p_y in increments of the batch_size so as to not cause
        # a memory error
        batch_size = net.batch_size
        for b in range(int(np.ceil(float(len(p_y))/net.batch_size))):
            x_batch = x_probe[b*batch_size:(b+1)*batch_size]
            y_batch = y_probe[b*batch_size:(b+1)*batch_size]
            p_y[b*batch_size:(b+1)*batch_size] += np.exp(-nlog_ls.eval(feed_dict={net.x:x_batch, net.y:y_batch,
                net.y_std:net.Y_std}))
    p_y /= n_samples
    p_y = p_y.reshape([n_grid_pts,n_grid_pts])
    return p_y

def plot_predictive_distribution_chicago(net, n_grid_pts,
        x_labels=("hour","month"), tanh_b=0, tanh_m=1, tanh_max=1,
        title_base="chicago crime distribution", x_for_eval=[(0., 0.)],
        max_color=None, n_samples=10, panel=False):
    """plot_predictive_distribution plots the predictive distribution as a heatmap.

    The temperature is the log likelihood

    Args:
        net: the model object
        n_grid_pts: number of bins on both y1 and y2 dimensions.
        return_buff: Set true to return the buffer rather that showing the
            plot.
        x_for_eval: the unscaled values of features which we are to plot
            x_0 is age, x_1 is time of day in the interval (0, 24)
    """
    # set bounds in location for plotting
    y1_bounds_real = [41.64,42.05]
    y2_bounds_real = [-87.9,-87.5]
    plot_extent = y2_bounds_real + y1_bounds_real
    y1 = np.linspace(y2_bounds_real[0],y2_bounds_real[1],n_grid_pts)
    y2 = np.linspace(y1_bounds_real[0],y1_bounds_real[1],n_grid_pts)
    Y1, Y2 = np.meshgrid(y1,y2)

    y_bounds = y_to_scaled_value(net, zip(y2_bounds_real, y1_bounds_real))
    y1_bounds, y2_bounds = y_bounds[:,1], y_bounds[:,0]

    # Set probe labels for calculating likelihoods.
    y1_probe = np.linspace(y1_bounds[0],y1_bounds[1],
            n_grid_pts).reshape([n_grid_pts,1])
    y1_probe = np.tile(y1_probe,reps=n_grid_pts)
    y2_probe = np.linspace(y2_bounds[0],y2_bounds[1],
            n_grid_pts).reshape([n_grid_pts,1])
    y2_probe = np.tile(y2_probe,reps=n_grid_pts).T

    # Reshape and stack into shape for querying computational graph
    y1_probe_list = Y1.reshape([n_grid_pts**2])
    y2_probe_list = Y2.reshape([n_grid_pts**2])
    y_probe = np.stack([y1_probe_list, y2_probe_list],axis=1)
    y_probe = y_to_scaled_value(net, y_probe)

    # Evaluate probabilities of each point.  If evaluating a Bayesian model
    # (where nlog_ls_eval is defined) evaluate under the posterior
    # predictive distribution.
    max_pts_eval = net.batch_size
    nlog_ls = net.nlog_ls_eval if hasattr(net, 'nlog_ls_eval') else net.nlog_ls
    p_y_data = np.exp(-nlog_ls.eval(feed_dict={net.x:net.X_train[:max_pts_eval],
        net.y:net.Y_train[:max_pts_eval], net.y_std:net.Y_std}))
    print "Minimum p_y_data: ",np.min(p_y_data), "Minimum p_y_data(no nans): ",np.nanmin(p_y_data)
    print "Max p_y_data: ",np.max(p_y_data), "Max p_y_data(no nans): ",np.nanmax(p_y_data)
    plt.close('all'); plt.figure(figsize=(20,15))
    if panel:
        print "creating panel"
        assert len(x_for_eval) != 1
        f, axarr = plt.subplots(1, 1+len(x_for_eval),
                figsize=(10*len(x_for_eval),10))
    else:
        f, axarr = plt.subplots(1, figsize=(20,15))
    for i, x in enumerate(x_for_eval):
        axis = axarr[i] if panel else axarr
        p_y = sample_heatmap_chicago(net, x, n_grid_pts, n_samples, y_probe,
                nlog_ls)

        # Estimate integral.
        dy = (Y1[1,1]-Y1[0,0])*(Y2[1,1]-Y2[0,0])
        integral = dy*np.sum(p_y)
        print "integral: ",integral


        ### Log the maximimum and minimum likelihood points.  This serves as a
        # good sanity check.
        print "Minimum p_y: ",np.min(p_y), "Minimum p_y(no nans): ",np.nanmin(p_y)
        print "Max p_y: ",np.max(p_y), "Max p_y(no nans): ",np.nanmax(p_y)

        # Plot heatmap and format figure
        plot_heatmap(Y1, Y2, p_y, max_color=max_color, chicago=True,
                extent=plot_extent, axis=axis, fig=f, colorbar=(i==0),
                tanh_b=tanh_b, tanh_m=tanh_m, tanh_max=tanh_max)
        axis.set_xlabel("Longitude")
        if i==0: axis.set_ylabel("Latitude")
        if i!=0: axis.get_yaxis().set_visible(False)
        title = title_base
        fn = net.log_base_dir + "/" + title_base
        for x_label in x_labels:
            title += "_%s_%s"%(x_label, str(x[x_label]))
            if x_label == 'crime':
                fn += "_%s_%s"%(x_label, str(x[x_label]))
            else:
                fn += "_%s_%05.02f"%(x_label, x[x_label])
        axis.set_title(title)
        #axis.set_xlim([-87.9,-87.5]); axis.set_ylim([41.64,42.05])
        y_lim = [41.70,41.98]
        x_lim = [-87.78,-87.55]
        axis.set_xlim(x_lim); axis.set_ylim(y_lim)

        fn += ".png"
        if not panel:
            plt.savefig(fn)
    if panel:
        plt.tight_layout()
        plt.savefig(fn)

    # Return the buffer of bytes encoding the png so that it can be written
    # for viewing within tensorboard.
    buff = io.BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    return buff

def plot_predictive_distribution_nyc(net, n_grid_pts, x_label = "Longitude",
        y_label="Lattitude", return_buff=False, n_pts_to_plot=10000,
        x_labels=("Age","Time_of_Day"), taxi=False,
        title_base="NYC Stop and Frisk Density", x_for_eval=[(0., 0.)],
        max_color=None, n_samples=1,
        contour=False,levels=[0,0.625,1.25,2.5,5,10,25,50,100,200,400],
        return_results=True):
    """plot_predictive_distribution plots the predictive distribution as a heatmap.

    The temperature is the log likelihood

    Args:
        net: the model object
        n_grid_pts: number of bins on both y1 and y2 dimensions.
        return_buff: Set true to return the buffer rather that showing the
            plot.
        x_for_eval: the unscaled values of features which we are to plot
            x_0 is age, x_1 is time of day in the interval (0, 24)
    """
    # set bounds in location for plotting
    y1_bounds_real = [40.5935855, 40.886808]
    y2_bounds_real = [-74.068968,-73.7371332]
    y1 = np.linspace(y2_bounds_real[0],y2_bounds_real[1],n_grid_pts)
    y2 = np.linspace(y1_bounds_real[0],y1_bounds_real[1],n_grid_pts)
    Y1, Y2 = np.meshgrid(y1,y2)

    y_bounds = y_to_scaled_value(net, zip(y2_bounds_real, y1_bounds_real))
    y1_bounds, y2_bounds = y_bounds[:,1], y_bounds[:,0]
    print "y1_bounds, y2_bounds: ", y1_bounds, y2_bounds

    # Set probe labels for calculating likelihoods.
    y1_probe = np.linspace(y1_bounds[0],y1_bounds[1],
            n_grid_pts).reshape([n_grid_pts,1])
    y1_probe = np.tile(y1_probe,reps=n_grid_pts)
    y2_probe = np.linspace(y2_bounds[0],y2_bounds[1],
            n_grid_pts).reshape([n_grid_pts,1])
    y2_probe = np.tile(y2_probe,reps=n_grid_pts).T

    # Reshape and stack into shape for querying computational graph
    y1_probe_list = Y1.reshape([n_grid_pts**2])
    y2_probe_list = Y2.reshape([n_grid_pts**2])
    y_probe = np.stack([y1_probe_list, y2_probe_list],axis=1)
    Y_train_plot = y_to_unscaled_value(net, net.Y_train[:n_pts_to_plot])
    Y_test_plot = y_to_unscaled_value(net, net.Y_test[:n_pts_to_plot])
    y_probe = y_to_scaled_value(net, y_probe)
    print "y_probe[0]: ", y_probe[0]
    print "y_probe[-1]: ", y_probe[-1]

    # Evaluate probabilities of each point.  If evaluating a Bayesian model
    # (where nlog_ls_eval is defined) evaluate under the posterior
    # predictive distribution.
    max_pts_eval = 5000
    nlog_ls = net.nlog_ls_eval if hasattr(net, 'nlog_ls_eval') else net.nlog_ls
    p_y_data = np.exp(-nlog_ls.eval(feed_dict={net.x:net.X_train[:max_pts_eval],
        net.y:net.Y_train[:max_pts_eval], net.y_std:net.Y_std}))
    print "x_train[0]:", net.X_train[0]
    print "y_train[0]: ", net.Y_train[0]
    print "Minimum p_y_data: ",np.min(p_y_data), "Minimum p_y_data(no nans): ",np.nanmin(p_y_data)
    print "Max p_y_data: ",np.max(p_y_data), "Max p_y_data(no nans): ",np.nanmax(p_y_data)
    results= {}

    for x in x_for_eval:
        p_y = np.zeros([n_grid_pts**2])
        for _ in range(n_samples):
            if x.has_key('pickup_time'):
                x['pickup_time_vec'] = nyc.hour_to_value(x['pickup_time'])

            if x['tip_percent']==None:
                tip = 0.
            else:
                tip = x['tip_percent']

            if x['fare']==None:
                fare = 0.
            else:
                fare = x['fare']
            x_vec = np.array([fare,tip, x['pickup_time_vec'][0],
                    x['pickup_time_vec'][1], x['passenger_count']])
            x_scaled = x_to_scaled_value(net, x_vec)
            if x['fare']==None: x_scaled[0]=np.float32(np.random.normal(0,1.))
            if x['tip_percent']==None: x_scaled[1]=np.float32(np.random.normal(0,1.))
            x_probe = np.ones([y_probe.shape[0],len(x_vec)],
                dtype=np.float32)*x_scaled[None]
            # add to p_y in increments of the batch_size so as to not cause
            # a memory error
            batch_size = net.batch_size
            for b in range(int(np.ceil(float(len(p_y))/batch_size))):
                x_batch = x_probe[b*batch_size:(b+1)*batch_size]
                y_batch = y_probe[b*batch_size:(b+1)*batch_size]
                batch_ps = np.exp(-nlog_ls.eval(feed_dict={net.x:x_batch, net.y:y_batch,
                    net.y_std:net.Y_std}))
                p_y[b*batch_size:(b+1)*batch_size] += batch_ps
        p_y /= n_samples
        p_y = p_y.reshape([n_grid_pts,n_grid_pts])

        # Estimate integral.
        dy = (Y1[1,1]-Y1[0,0])*(Y2[1,1]-Y2[0,0])
        integral = dy*np.sum(p_y)
        print "integral: ",integral


        ### Log the maximimum and minimum likelihood points.  This serves as a
        # good sanity check.
        print "Minimum p_y: ",np.min(p_y), "Minimum p_y(no nans): ",np.nanmin(p_y)
        print "Max p_y: ",np.max(p_y), "Max p_y(no nans): ",np.nanmax(p_y)
        results[str(x)]=p_y

        # Plot heatmap and format figure
        plt.close('all')
        f, axarr = plt.subplots(1, figsize=(20,15))
        if contour:
            CS=plt.contour(Y1,Y2,p_y, levels=levels, colors='k')
            plt.clabel(CS, fontsize=9, inline=1)
        else:
            plot_heatmap(Y1, Y2, p_y, max_color=max_color, fig=f)
        plt.xlabel(x_label); plt.ylabel(y_label)
        plt.title(title_base + " -- %s %05.01f -- %s %05.01f"%(x_labels[0],
            x[x_labels[0]], x_labels[1],  x[x_labels[1]]))
        plt.axis(y2_bounds_real + y1_bounds_real)
        if True:
            plt.axis([-74.05,-73.9,40.65,40.85])

        # Plot some of the training and Test points onto heatmap
        # taxis was 0.05
        alpha = 0.01 if taxi else 0.2
        # Don't plot actual data
        #plt.scatter(Y_train_plot[:10000,0],Y_train_plot[:10000,1],c='k',s=1.,
        #        alpha=alpha)
        #plt.scatter(Y_test_plot[:10000,0],Y_test_plot[:10000,1],c='g',s=1.,
        #        alpha=alpha)

        fn = net.log_base_dir + "/" + title_base + \
                "__%s_%05.01f__%s_%05.01f.png"%(x_labels[0],
            x[x_labels[0]], x_labels[1],  x[x_labels[1]])
        plt.savefig(fn)

    # Return the buffer of bytes encoding the png so that it can be written
    # for viewing within tensorboard.
    if return_buff:
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return buff
    elif return_results:
        return Y1, Y2, results

def x_to_scaled_value(net, x):
    return (np.array(x, dtype=np.float32)-net.X_mean)/net.X_std

def y_to_unscaled_value(net, y):
    """this does the opposite of x_to_scaled_value, moving an scaled y to
    its original domain"""
    return np.array(y, dtype=np.float32)*net.Y_std+net.Y_mean

def y_to_scaled_value(net, y):
    from_mean = np.array(y, dtype=np.float32)-net.Y_mean
    return (from_mean)/net.Y_std


def plot_predictive_distribution_mog(net, n_grid_pts, x_label = "Y1",
        y_label="Y2", return_buff=False, n_pts_to_plot=10000):
    """plot_predictive_distribution plots the predictive distribution as a heatmap.

    The temperature is the log likelihood

    Args:
        net: the model object
        n_grid_pts: number of bins on both y1 and y2 dimensions.
        return_buff: Set true to return the buffer rather that showing the
            plot.
    """
    # identify bounds in Training Data
    y1_bounds = [np.min(net.Y_train[:,0]), np.max(net.Y_train[:,0])]
    y2_bounds = [np.min(net.Y_train[:,1]), np.max(net.Y_train[:,1])]

    # Set probe labels for calculating likelihoods.
    y1_probe = np.linspace(y1_bounds[0],y1_bounds[1],
            n_grid_pts).reshape([n_grid_pts,1])
    y1_probe = np.tile(y1_probe,reps=n_grid_pts)
    y2_probe = np.linspace(y2_bounds[0],y2_bounds[1],
            n_grid_pts).reshape([n_grid_pts,1])
    y2_probe = np.tile(y2_probe,reps=n_grid_pts).T

    # Reshape and stack into shape for querying computational graph
    y1_probe_list = y1_probe.reshape([n_grid_pts**2])
    y2_probe_list = y2_probe.reshape([n_grid_pts**2])
    y_probe = np.stack([y1_probe_list, y2_probe_list],axis=1)
    x_probe = np.zeros([y_probe.shape[0],1], dtype=np.float32)

    # Evaluate probabilities of each point.  If evaluating a Bayesian model
    # (where nlog_ls_eval is defined) evaluate under the posterior
    # predictive distribution.
    nlog_ls = net.nlog_ls_eval if hasattr(net, 'nlog_ls_eval') else net.nlog_ls
    #nlog_ls = net.nlog_ls[0]
    if hasattr(net, 'x'):
        p_y = np.exp(-nlog_ls.eval(feed_dict={net.x:x_probe, net.y:y_probe,
            net.y_std:net.Y_std}))
        p_y_data = np.exp(-nlog_ls.eval(feed_dict={net.x:net.X_train,
            net.y:net.Y_train, net.y_std:net.Y_std}))
    else:
        p_y = np.exp(-nlog_ls.eval(feed_dict={net.y:y_probe,
            net.y_std:net.Y_std}))
        p_y_data = np.exp(-nlog_ls.eval(feed_dict={net.y:net.Y_train,
            net.y_std:net.Y_std}))
    p_y = p_y.reshape([n_grid_pts,n_grid_pts])

    # Estimate integral.
    dy = (y1_probe[1,1]-y1_probe[0,0])*(y2_probe[1,1]-y2_probe[0,0])
    integral = dy*np.sum(p_y)
    print "integral: ",integral

    ### Log the maximimum and minimum likelihood points.  This serves as a
    # good sanity check.
    print "Minimum p_y: ",np.min(p_y), "Minimum p_y(no nans): ",np.nanmin(p_y)
    print "Max p_y: ",np.max(p_y), "Max p_y(no nans): ",np.nanmax(p_y)
    print "Minimum p_y_data: ",np.min(p_y_data), "Minimum p_y_data(no nans): ",np.nanmin(p_y_data)
    print "Max p_y_data: ",np.max(p_y_data), "Max p_y_data(no nans): ",np.nanmax(p_y_data)

    if False:# hasattr(net, 'weights'):
        for k, weights in net.weights.items():
            weights_vals = weights.eval()
            print k
            print "\tMinimum weight: ",np.min(weights_vals), "Minimum weight(no nans):",np.nanmin(weights_vals)
            print "\tmax weight: ",np.max(weights_vals), "max weight(no nans):",np.nanmax(weights_vals)
        for k, biases in net.biases.items():
            bias_vals = biases.eval()
            print k
            print "\tMinimum bias: ",np.min(bias_vals), "Minimum bias(no nans):", np.nanmin(bias_vals)
            print "\tmax bias: ",np.max(bias_vals), "max bias(no nans):", np.nanmax(bias_vals)

    # Plot heatmap and format figure
    plt.close('all'); plt.figure(figsize=(10,5))
    heatmap = plt.pcolor(y1_probe, y2_probe, p_y, cmap='RdBu_r')
    plt.colorbar(heatmap)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.title("p(Y_1, Y_2) Learned with Normalizing Flows")
    plt.axis(y1_bounds + y2_bounds)

    # Plot some of the training and Test points onto heatmap
    plt.scatter(net.Y_train[:n_pts_to_plot,0],net.Y_train[:n_pts_to_plot,1],c='k',s=1., alpha=0.5)
    plt.scatter(net.Y_test[:n_pts_to_plot,0],net.Y_test[:n_pts_to_plot,1],c='g',s=1., alpha=0.5)

    # plot a star at the center point of the radial flows, scaling
    # it's size based on the magnitude of the distortion.
    if hasattr(net, 'flows'):
        for flow in net.flows:
            if not hasattr(flow,'z_0'):
                continue
            # if the values are input dependent or sampled, we need to
            # handle the larger dimensionality of the center points
            if hasattr(net, 'input_dependent') and net.input_dependent:
                z_0 = flow.z_0.eval(feed_dict={net.x:net.X,
                    })[0]
                beta = np.sum(abs(flow.beta.eval(feed_dict={net.x:net.X,
                    })[0][0]))
            else:
                z_0 = flow.z_0.eval()[0]
                beta = flow.beta.eval()[0]
            plt.scatter([z_0[0]],[z_0[1]], c='r',marker='*',s=(abs(beta)*55.+50.))

    # Return the buffer of bytes encoding the png so that it can be written
    # for viewing within tensorboard.
    if return_buff:
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return buff
    else:
        plt.show()

def add_image_summary():
    with tf.name_scope('pred_distribution_img'):
        pred_prob_buff = tf.placeholder(tf.string,
                name='pred_prob_buff')
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(pred_prob_buff, channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        tf.summary.image("predictive_distribution", image, max_outputs=50)
    return pred_prob_buff
