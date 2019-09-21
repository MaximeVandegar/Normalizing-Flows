from __future__ import division
import sys
import network
import my_utils as utils
import tensorflow as tf
import numpy as np

class noise_network(network.model_base):
    """noise_network is a subclass of model_base for Bayesian networks with input
    noise.

    """
    def __init__(self, init_sigma_params=1e-4, w_prior_sigma=1.0,
            n_samples_noise=10, **kwargs):
        """initialize a network

        We must make several chooses in our initialization.  Namely:
            the initial values for the posterior means
            the initial values for the posterior variances
            the prior variances, which may be different by layer and may be
                learned later in optimization.

        Args:
            init_sigma_params: the initial posterior variances for the
                weights.
            w_prior_sigma: prior std-dev on weights
            n_samples_noise: number of samples over the latent variable to
                use.
        """
        self.w_prior_sigma = w_prior_sigma
        self.init_sigma_params = init_sigma_params
        self.n_samples_noise = n_samples_noise

        n_outputs = 1 # we just produce 1D samples
        network.model_base.__init__(self, n_outputs=n_outputs, **kwargs)

        self.KL = self.KL_BNN
        self.nlog_ls, self.nlog_l, self.nlog_l_eval = self.likelihood(self.y)

        # scale expected log likelihood by number of datapoints
        self.cost = self.nlog_l*self.Y.shape[0] + self.KL
        self.set_summaries()

        ### set 2 optimizer stages
        self.construct_optimizer()

    def set_summaries(self):
        tf.summary.histogram("nlog_ls",self.nlog_ls)
        tf.summary.scalar("nlog_l",self.nlog_l)
        tf.summary.scalar("nlog_l_eval",self.nlog_l_eval)
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("sigma_obs", self.sigma_obs)
        tf.summary.scalar("KL", self.KL)

    def construct_network(self, n_units, n_samples=1, noise_dim=0,
            keep_p=1., nonlinearity=True, name=""):
        """construct_network establishes all weight matrices and biases and
        connects them.

        Args:
            n_units: the sizes of all layers including input and output
                layer
            input_layer: the input tensor, if none is provided, we simply
            use self.x
        """
        print "constructing network, n_units: ",n_units
        # TODO use kwargs for more elagant solutions to being called by this
        # base class
        assert keep_p ==1. and nonlinearity

        ### Define parameters of the network
        self.weights, self.biases, KL = {}, {}, 0.
        self.layers = []
        # Establish paramters of appromiate posterior over weights and
        # biases.
        sigma_init = self.init_sigma_params
        n_samples_noise = self.n_samples_noise
        for l in range(1, len(n_units)):
            with tf.variable_scope(name+'Layer_%d'%l):
                n_in, n_out = n_units[l-1], n_units[l]
                if l==1: n_in += self.noise_dim

                # use non neglidgible uncertainty if we are doing VI
                w_prior_sigma, b_prior_sigma = self.w_prior_sigma, self.w_prior_sigma
                mu_init_sigma_w, mu_init_sigma_b =  np.sqrt(1./(n_in)), 1.

                (w_mu, w_logstd), _, w_KL = utils.set_q(name+"w_%d"%l,
                        sigma_prior=w_prior_sigma, mu_init_sigma=mu_init_sigma_w,
                        sigma_init=sigma_init, n_samples=0,
                        size=[n_in, n_out], save_summary=True)

                # We use same init_sigma for weights and biases.
                (b_mu, b_logstd), _, b_KL = utils.set_q(name+"b_%d"%l,
                        sigma_prior=b_prior_sigma, mu_init_sigma=mu_init_sigma_b,
                        sigma_init=sigma_init, n_samples=0,
                        size=[n_out], save_summary=True)
                self.weights['w_%d_mu'%l], self.weights['w_%d_std'%l] = w_mu, tf.nn.softplus(w_logstd)
                self.biases['b_%d_mu'%l], self.biases['b_%d_std'%l] = b_mu, tf.nn.softplus(b_logstd)

                self.params += [w_mu, b_mu, w_logstd, b_logstd]
                KL += w_KL + b_KL

        # Separate out weights for input noise
        if noise_dim !=0 :
            noise_weights_mu = self.weights['w_1_mu'][n_units[0]:]
            noise_weights_std = self.weights['w_1_std'][n_units[0]:]
            self.weights['w_1_mu'] = self.weights['w_1_mu'][:n_units[0]]
            self.weights['w_1_std'] = self.weights['w_1_std'][:n_units[0]]

        # Add an extra dimension to correspond to samples.
        prev_layer = tf.stack([self.x]*n_samples_noise)
        prev_layer = tf.stack([prev_layer]*n_samples)
        self.layers.append(prev_layer)
        # shape is [n_samples, n_samples_noise, ?, dim(x)]

        ### Define activations in each layer
        for l in range(1,len(n_units)):
            print "defining activations in layer %d"%l
            print "prev_layer.shape", prev_layer.shape
            # Multiply with weight matrix and add bias
            layer_pre_bias = tf.tensordot(prev_layer, self.weights['w_%d_mu'%l],
                    axes=[[3],[0]])
            # Shape of layer_pre_bias is [n_samples, n_samples_noise, ?, n_units[l]]

            # add mean bias term
            layer = layer_pre_bias + self.biases['b_%d_mu'%l]

            # Calculate the noise in each hidden unit.
            # must use absolute value of activation because final layer may
            # have negative values.
            layer_var = tf.tensordot(prev_layer**2, self.weights['w_%d_std'%l]**2,
                    axes=[[3], [0]])
            layer_var += self.biases['b_%d_std'%l]**2

            # Add noise at the first layer
            if l == 1 and noise_dim !=0 :
                # we don't use different noise for each sample of weights.
                input_noise = tf.random_normal([self.n_samples_noise, noise_dim], mean=0.,stddev=1.)
                # To add noise, we must expand in variable batch size
                # dimension.
                layer_noise = tf.matmul(input_noise, noise_weights_mu)
                layer_var_noise = tf.tensordot(input_noise[:, :, None]**2,
                        noise_weights_std**2,axes=[[1],[0]])[:, None, :]
                # layer_var_noise should  be shape [n_samples_noise, n_out]
                print "layer_var_noise.shape", layer_var_noise.shape
                layer_var_noise = layer_var_noise[:, 0, :, :]
                print "layer_var_noise.shape (post collapse)", layer_var_noise.shape
                layer_var += layer_var_noise
                layer += layer_noise[:, None, :]

            # Now sample noise and add scaled noise.
            # This constitutes the local reparameterization trick.
            print "adding noise to graph"
            eps = tf.random_normal(name='eps_%d'%l, mean=0.,
                        stddev=1.0, shape=[n_samples, 1, 1, n_units[l]])
            layer_sigma = tf.sqrt(layer_var)
            layer += layer_sigma*eps
            with tf.name_scope(name+"Neural_Network_Activations_%d"%l):
                tf.summary.histogram(name+"Layer_%d_sigmas"%l, layer_sigma)
                tf.summary.histogram(name+"Layer_%d_activations_pre_tanh"%l, layer)

            # Add tanh nonlinearity
            if l != (len(n_units) - 1): layer = tf.nn.tanh(layer)

            with tf.name_scope(name+"Neural_Network_Activations_%d"%l):
                tf.summary.histogram(name+"Layer_%d_activations_post_tanh"%l,layer)

            prev_layer = layer
            self.layers.append(prev_layer)
        self.KL_BNN = KL
        return prev_layer

    def likelihood(self, y):
        """define the input noise network likelihood.

        Returns:
            Expected log likelihood (per example and summed) and log expected
            likelihood (i.e. MC approximation of the likelihood under the
            approximated posterior predictive)
        """

        # in this case, this is the sum of log jacobian determinents.
        y = tf.stack([y]*self.n_samples_noise)
        y = tf.stack([y]*self.n_samples)

        # Define the base distribution that will be warped as unit gaussian
        log_sigma_obs = tf.get_variable("log_sigma_obs",
                initializer=tf.constant(np.float32(utils.un_softplus(1.0))))
        self.params.append(log_sigma_obs)
        self.all_params.append(log_sigma_obs)
        self.sigma_obs = tf.nn.softplus(log_sigma_obs)

        # Calculate the negative log likelihood
        dist = tf.contrib.distributions.Normal(loc=self.outputs, scale=self.sigma_obs)
        # calculate likelihood for each sample of weights and noise
        log_ls = dist.log_prob(y) - tf.reduce_sum(tf.log(self.y_std))
        log_ls = tf.reduce_logsumexp(log_ls, axis=1) - tf.log(float(self.n_samples_noise))
        nlog_ls = -log_ls
        print "log_ls.shape", log_ls.shape
        # this should now be shape [n_samples, ?]
        nlog_l = tf.reduce_mean(nlog_ls)

        ### Do log sum exp
        self.nlog_ls_eval = -tf.reduce_logsumexp(-nlog_ls, axis=0)
        self.nlog_ls_eval += tf.log(float(self.n_samples))
        nlog_l_eval = tf.reduce_mean(self.nlog_ls_eval)
        return nlog_ls, nlog_l, nlog_l_eval
