from __future__ import division
import network
import tensorflow as tf
import numpy as np
import my_utils as utils

class mixture_density_network(network.model_base):
    """mixture_density_network is a subclass of model_base for networks with a mixture
    of Gaussians likelihood model.

    The mixing proportions are defined by a softmax transformed output.

    The means and variances may also be outputs
    """
    def __init__(self, n_components=1, init_sigma_params=1e-4,
            w_prior_sigma= 1., **kwargs):
        """initialize a network with normalizing flows.

        Args:
            n_components: number of mixture components.
            w_prior_sigma: we set this to 100., following Iain Murray's
                paper (which set the precision to 0.01)
        """
        self.n_components = n_components

        # 1 output for predicting offset.
        # if input dependent, then predict mixing proportion, mean and variance
        # for each mixing component.
        self.w_prior_sigma = w_prior_sigma
        self.init_sigma_params = init_sigma_params
        n_outputs = 1 + self.n_components*3

        network.model_base.__init__(self, n_outputs=n_outputs, **kwargs)
        print "nework outputs.shape", self.outputs.shape

        ### Construct likelihood using normalizing flows
        # In this case, the likelihood is defined by our normalizing flow.
        self.construct_mog(self.outputs)
        self.nlog_ls, self.nlog_l, self.nlog_l_eval = self.likelihood(self.y)
        self.KL = self.KL_BNN

        ### Construct Cost (likelihood and regularizers)
        self.cost = self.nlog_l*self.Y.shape[0] + self.KL

        ### set optimizer stages
        self.construct_optimizer()
        self.set_summaries()

    def set_summaries(self):
        tf.summary.scalar("cost", self.cost)
        tf.summary.histogram("nlog_ls", self.nlog_ls)
        tf.summary.scalar("nlog_l", self.nlog_l)
        tf.summary.scalar("nlog_l_eval", self.nlog_l_eval)

    def construct_network(self, n_units, n_samples=1, noise_dim=0,
            keep_p=1., nonlinearity=True, init_params=None, name=""):
        """construct_network establishes all weight matrices and biases and
        connects them.

        The outputs may include parameters of the flow

        Args:
            n_units: the sizes of all layers including input and output
                layer
        """
        print "constructing network, n_units: ",n_units
        # TODO use kwargs for more elagant solutions to being called by this 
        # base class
        assert keep_p ==1. and nonlinearity and noise_dim == 0

        assert init_params is None # this is implemented only in the Bayesian flow version of this function

        ### Define parameters of the network
        self.weights, self.biases, KL = {}, {}, 0.
        self.layers = []
        # Establish paramters of appromiate posterior over weights and
        # biases.
        for l in range(1, len(n_units)):
            with tf.variable_scope(name+'Layer_%d'%l):
                n_in, n_out = n_units[l-1], n_units[l]

                # use non neglidgible uncertainty if we are doing VI
                sigma_init = self.init_sigma_params

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

        # Add an extra dimension to correspond to samples.
        prev_layer = tf.stack([self.x]*n_samples)
        self.layers.append(prev_layer)
        # shape is [n_samples, ?, dim(x)]

        ### Define activations in each layer
        for l in range(1,len(n_units)):
            print "defining activations in layer %d"%l
            # Multiply with weight matrix and add bias
            prev_layer = tf.reshape(prev_layer, [-1, n_units[l-1]])
            layer_pre_bias = tf.matmul(prev_layer, self.weights['w_%d_mu'%l])
            layer_pre_bias = tf.reshape(layer_pre_bias, [n_samples, -1, n_units[l]])
            # Shape of layer_pre_bias is [n_samples, ?, n_units[l]]

            # add mean bias term
            layer = tf.add(layer_pre_bias, self.biases['b_%d_mu'%l][None, None, :])

            # Calculate the noise in each hidden unit.
            # must use absolute value of activation because final layer may
            # have negative values.
            layer_var = tf.matmul(tf.reshape(prev_layer**2,[-1,
                n_units[l-1]]), self.weights['w_%d_std'%l]**2)
            layer_var = tf.reshape(layer_var, [n_samples, -1, n_units[l]])
            layer_var += self.biases['b_%d_std'%l]**2

            # Now sample noise and add scaled noise.
            # This constitutes the local reparameterization trick.
            eps = tf.random_normal(name='eps_%d'%l, mean=0.,
                        stddev=1.0, shape=[n_samples, 1, n_units[l]])
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

    def construct_mog(self, outputs):
        """construct_flow builds and links together the normalizing flow and
        establishes the log likelihood of samples.

        args:
            outputs: the outputs of the neural network which we will use to
                parameterize the flows.

        Returns:
            new parameters of mog (i.e. those not defined as outputs of
                the network), and the negative log likelihoods
        """
        # check for correct number of input dimensions.
        assert outputs.shape[-1] == (self.n_components)*3 + 1
        out_idx = 0 # keep track of which output we are working with.
        self.shift = outputs[0, :, out_idx:out_idx+1]; out_idx += 1
        with tf.name_scope("Mixture_of_Gaussians"):
            with tf.variable_scope('network'):
                # get mixing proportions
                theta_raw = outputs[:, :,out_idx:out_idx+self.n_components];
                out_idx += self.n_components
                self.theta = tf.nn.softmax(theta_raw)

                log_sigmas = outputs[:, :, out_idx:out_idx+self.n_components];
                out_idx += self.n_components
                self.sigmas = tf.exp(log_sigmas,name="sigmas")

                self.mus = outputs[:, :, out_idx:out_idx+self.n_components];
                out_idx += self.n_components
                for k in range(self.n_components):
                    tf.summary.histogram("Gaussian_%d_sigma"%k,self.sigmas[:,k])
                    tf.summary.histogram("Gaussian_%d_proportion"%k,self.theta[:,k])
                    tf.summary.histogram("Gaussian_%d_mus"%k,self.mus[:,k])

        ## Check that every output has been used
        assert out_idx == outputs.shape[-1]

    def likelihood(self, y):
        ### Construct Likelihood for MOG
        dist = tf.contrib.distributions.Normal(loc=self.mus,scale=self.sigmas)
        print "self.shift.shape", self.shift.shape
        assert len(y.shape) == 2 and y.shape[1] == 1
        # shift y based on the shift prediction.
        y = tf.transpose([y[:,0]-self.shift[:, 0]]*self.n_components)
        obs = tf.stack([y]*self.n_samples)

        log_ls = dist.log_prob(obs)
        log_ls += tf.log(self.theta)
        print "log_ls shape(pre reduce): ",log_ls.shape
        log_ls = tf.reduce_logsumexp(log_ls, axis=-1)
        print "log_ls shape(post reduce): ",log_ls.shape

        # Calculate the negative log likelihood
        nlog_ls = -(log_ls - tf.log(self.y_std))
        nlog_l = tf.reduce_mean(nlog_ls)

        self.nlog_ls_eval = -tf.reduce_logsumexp(-nlog_ls, axis=0)
        print "nlog_ls.shape", nlog_ls.shape
        print "nlog_ls_eval.shape", self.nlog_ls_eval.shape
        self.nlog_ls_eval += tf.log(float(self.n_samples))
        nlog_l_eval = tf.reduce_mean(self.nlog_ls_eval)
        return nlog_ls, nlog_l, nlog_l_eval

