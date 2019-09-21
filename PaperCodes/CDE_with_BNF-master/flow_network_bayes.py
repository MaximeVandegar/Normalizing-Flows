from __future__ import division
import sys
import network
import utils
import flows
import tensorflow as tf
import numpy as np

class flow_network_bayes(network.model_base):
    """flow_network_bayes is a subclass of model_base for networks with normalizing
    flows on top.

    The flow networks consists of an MLP which is used to parameterize the
    target distribution, whose form is defined by a normalizing flow.
    """
    def __init__(self, n_flows=1, alpha_std=1., beta_std=1.0, z_std=1.0,
            init_sigma_params=1e-5, bayes_layers=None, noise_mag=1.0, init_sigma_obs=1.0, lmbda=.5,
            length_scale=1., learn_ls=False,  learn_lmbda=False, inference='VI',
            w_prior_sigma=None, anneal_in_KL=False, nonlinearity='tanh',
            learn_sigma_weights=True, learn_beta_std=False,
            learn_sigma_obs=True,  **kwargs):
        """initialize a network with normalizing flows.

        We must make several chooses in our initialization.  Namely:
            the initial values for the posterior means
            the initial values for the posterior variances
            the prior variances, which may be different by layer and may be
                learned later in optimization.

        Args:
            n_flows: number of stages in the normalizing flows.
            init_sigma_params: the initial posterior variances for the
                weights.
            bayes_layers: list of layers in which to track posterior
                variances.
            lmbda: interpolation between homoscedastic and heteroscedastic,
                as lmbda goes to 0, we have a homoscedastic system Bounded (0,1)
            length_scale: defines a pre-activation scaling of hidden units.
                this can be thought of as defining the steepness of the
                nonlinearity, which in part controls the fucntion's length scale.
            learn_ls: set true to learn the length scale of the network as
                part of inference.
            learn_lmbda: set true to learn the degree of heteroscedasticity
                as part of inference.
            learn_beta_std: set true to learn beta_std
            learn_sigma_weights: set to False to use weight noise.
            inference: inference method to use, must be VI, MAP, or MLE
        """
        self.noise_mag = noise_mag
        self.learn_sigma_obs = learn_sigma_obs
        self.w_prior_sigma = w_prior_sigma
        self.init_sigma_obs = init_sigma_obs
        self.alpha_std = tf.get_variable('alpha_std',
                initializer=tf.constant(np.float32(alpha_std)))
        self.beta_std = tf.get_variable('beta_std',
                initializer=tf.constant(np.float32(beta_std)))
        self.z_std = tf.get_variable('z_std',
            initializer=tf.constant(np.float32(z_std)))
        self.learn_sigma_weights = learn_sigma_weights
        assert lmbda <= 1. and lmbda >= 0.
        assert inference == 'MAP' or inference == 'VI' or inference == 'MLE'
        self.inference = inference
        self.lmbda_val = lmbda
        if learn_lmbda:
            self.log_lmbda = tf.get_variable('log_lmbda', initializer=tf.constant(
                        np.float32(np.log(lmbda/(1.-lmbda)))))
            self.lmbda = tf.sigmoid(self.log_lmbda)
        else:
            self.lmbda = lmbda

        log_length_scale = utils.un_softplus(length_scale)
        self.log_length_scale =  tf.get_variable('log_length_scale',
                initializer=tf.constant(np.float32(log_length_scale)))
        self.length_scale = tf.nn.softplus(self.log_length_scale)
        self.nonlinearity = nonlinearity

        ## Set initial posterior variances
        if inference != 'VI':
            init_sigma_params = 1e-7
        self.init_sigma_params = init_sigma_params

        print "initializing , bayes layers: ", bayes_layers
        if bayes_layers == None:
            self.bayes_layers = list(range(1,len(kwargs['n_hidden_units'])+2))
        else:
            self.bayes_layers = bayes_layers

        n_outputs = 1 + 3*n_flows
        network.model_base.__init__(self, n_outputs=n_outputs, **kwargs)
        # This calls construct network and defines: self.nn_mus,
        # self.nn_sigmas, self.KL_BNN and self.nn_prior_sigma

        ### Construct likelihood using normalizing flows
        # In this case, the likelihood is defined by our normalizing flow.
        self.flows, self.KL_flows = self.construct_flow(
                self.outputs, self.y, n_flows+1, n_samples=self.n_samples)

        # Get the log losses for individual samples and the mean log loss
        # both for individual samples and averaged across the posterior
        # Specifically, self.nlog_l represents the expected log likelihood
        # in the ELBO
        self.KL = self.KL_flows + self.KL_BNN
        self.nlog_ls, self.nlog_l, self.nlog_l_eval = self.likelihood(self.y, self.flows)

        # If we are doing a 2-stage training, we will only optimize wrt this
        # full set of parameters in the 2nd stage.
        self.all_params = list(self.params)

        self.learn_ls, self.learn_lmbda = learn_ls, learn_lmbda
        if learn_ls: self.all_params.append(self.log_length_scale)
        if learn_lmbda: self.all_params.append(self.log_lmbda)
        if learn_beta_std: self.all_params.append(self.beta_std)

        iteration_after_switch = tf.cast(tf.abs(self.epoch-self.epoch_switch_opt)+
            (self.epoch-self.epoch_switch_opt), tf.float32)
        if anneal_in_KL:
            assert inference != 'MLE'
            KL_weight =  1. - tf.exp(-0.03*(tf.cast(tf.abs(self.epoch-self.epoch_switch_opt)+
                (self.epoch-self.epoch_switch_opt), tf.float32))/2.)
        elif inference == 'MLE':
            KL_weight = 0.
        else:
            KL_weight = 1.

        ### Construct Cost (likelihood and KL)
        self.KL_weighted =  self.KL*KL_weight # weight if reverse anealing KL in.
        # scale expected log likelihood by number of datapoints
        if self.noise_dim != 0:
            assert inference != "VI"
            self.cost = self.nlog_l_eval*self.Y.shape[0] + self.KL_weighted
        else:
            self.cost = self.nlog_l*self.Y.shape[0] + self.KL_weighted
        self.set_summaries()

        ### set 2 optimizer stages
        self.construct_optimizer()

    def set_summaries(self):
        tf.summary.histogram("nlog_ls",self.nlog_ls)
        tf.summary.scalar("nlog_l",self.nlog_l)
        tf.summary.scalar("nlog_l_eval",self.nlog_l_eval)
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("KL", self.KL)
        tf.summary.scalar("KL_BNN", self.KL_BNN)
        tf.summary.scalar("KL_flows", self.KL_flows)
        tf.summary.scalar("beta_std", self.beta_std)
        if self.learn_ls:
            tf.summary.scalar("length_scale", self.length_scale)
        if self.learn_lmbda and self.lmbda.shape != []:
            tf.summary.scalar("lmbda_y1", self.lmbda[0])
            tf.summary.scalar("lmbda_y2", self.lmbda[1])
        elif self.learn_lmbda:
            tf.summary.scalar("lmbda", self.lmbda)

    def construct_network(self, n_units, n_samples=1, noise_dim=0,
            keep_p=1., nonlinearity=True, input_layer=None, init_params=None, name=""):
        """construct_network establishes all weight matrices and biases and
        connects them.

        The outputs may include parameters of the flow

        Args:
            n_units: the sizes of all layers including input and output
                layer
            n_samples: number of MC samples of noise (not variational samples)
            noise_dim: dimension of random noise
            input_layer: the input tensor, if none is provided, we simply
                use self.x
            init_params: values for the weights and biases to set as the initial
                means, this is a dictionary indexed as in self.weights and
                self.biases
        """
        print "constructing network, n_units: ",n_units
        # TODO use kwargs for more elagant solutions to being called by this
        # base class
        assert keep_p ==1. and nonlinearity

        # we cannot handle input noise and be Bayesian at this time.
        if noise_dim != 0: assert self.bayes_layers == []

        ### Define parameters of the network
        self.weights, self.biases, KL = {}, {}, 0.
        self.layers = []
        # Establish paramters of appromiate posterior over weights and
        # biases.
        print "constructin net, n_units:", n_units
        for l in range(1, len(n_units)):
            with tf.variable_scope(name+'Layer_%d'%l):
                n_in, n_out = n_units[l-1], n_units[l]
                if l==1: n_in += self.noise_dim

                # use non negligible uncertainty if we are doing VI
                sigma_init = self.init_sigma_params if l in self.bayes_layers else 1e-7

                if self.w_prior_sigma is not None:
                    w_prior_sigma, b_prior_sigma = self.w_prior_sigma, self.w_prior_sigma*np.sqrt(n_in)
                else:
                    w_prior_sigma, b_prior_sigma = np.sqrt(1./(n_in)), 1.

                # We use same init_sigma for weights and biases.
                # If initial parameters have been provided, we use those.
                if init_params is not None:
                    (w_mu, w_logstd), _, w_KL = utils.set_q(name+"w_%d"%l,
                            sigma_prior=w_prior_sigma, sigma_init=sigma_init,
                            n_samples=0, size=[n_in, n_out], save_summary=True,
                            mu_init_values=init_params['w_%d'%l])
                    (b_mu, b_logstd), _, b_KL = utils.set_q(name+"b_%d"%l,
                            sigma_prior=b_prior_sigma, sigma_init=sigma_init,
                            n_samples=0, size=[n_out], save_summary=True,
                            mu_init_values=init_params['b_%d'])
                else:
                    if l == len(n_units)-1:
                        prior_scaling = np.array([1]+[self.z_std, self.alpha_std,
                            self.beta_std]*int((n_out-1)/3))
                        w_prior_sigma *= prior_scaling
                        b_prior_sigma *= prior_scaling
                    (w_mu, w_logstd), _, w_KL = utils.set_q(name+"w_%d"%l,
                            sigma_prior=w_prior_sigma, mu_init_mu=0.,
                            mu_init_sigma=w_prior_sigma, sigma_init=sigma_init,
                            n_samples=0, size=[n_in, n_out], save_summary=True)
                    (b_mu, b_logstd), _, b_KL = utils.set_q(name+"b_%d"%l,
                            sigma_prior=b_prior_sigma, mu_init_mu=0.,
                            mu_init_sigma=b_prior_sigma, sigma_init=sigma_init,
                            n_samples=0, size=[n_out], save_summary=True)
                self.weights['w_%d_mu'%l], self.weights['w_%d_std'%l] = w_mu, tf.nn.softplus(w_logstd)
                self.biases['b_%d_mu'%l], self.biases['b_%d_std'%l] = b_mu, tf.nn.softplus(b_logstd)

                # For the final layer, we must scale the weights defining
                # the parameters of the normalizing flows.
                # We scale the input dependent portion (coming
                # from the weights and previous layer activations) by lmbda, we
                # will finally add additional offsets and scalings when we build
                # the normalizing flow.
                if l == (len(n_units) - 1) and n_out != 1:
                    # construct mask
                    mask = np.ones([n_out],dtype=np.float32)
                    # we don't want to adjust the first output.
                    mask[1:] *= self.lmbda_val
                    self.weights['w_%d_mu'%l] *= mask
                    self.weights['w_%d_std'%l] *= mask

                self.params += [w_mu, b_mu]
                KL += w_KL + b_KL
                if l in self.bayes_layers and self.learn_sigma_weights and self.inference == 'VI':
                    print "adding uncertainties for layer %d"%l
                    # if we are not being bayesian in this layer,
                    # we don't learn the variances of these parameters.
                    self.params += [w_logstd, b_logstd]

        # Separate out weights for input noise
        if noise_dim !=0 :
            noise_weights_mu = self.weights['w_1_mu'][n_units[0]:]
            noise_weights_std = self.weights['w_1_std'][n_units[0]:]
            self.weights['w_1_mu'] = self.weights['w_1_mu'][:n_units[0]]
            self.weights['w_1_std'] = self.weights['w_1_std'][:n_units[0]]

        # Add an extra dimension to correspond to samples.
        prev_layer = tf.stack([self.x]*n_samples) if input_layer is None else input_layer
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

            # Add noise at the first layer
            if l == 1 and noise_dim !=0 :
                # we don't use different noise for each sample of weights.
                input_noise = tf.random_uniform([n_samples, noise_dim], minval=-self.noise_mag,
                    maxval=self.noise_mag)
                # To add noise, we must expand in variable batch size
                # dimension.
                layer_var += tf.matmul(input_noise**2,
                        noise_weights_std**2)[:, None, :]
                layer_noise = tf.matmul(input_noise, noise_weights_mu)
                layer += layer_noise[:, None, :]

            # Now sample noise and add scaled noise.
            if l in self.bayes_layers and self.inference == 'VI':
                # This constitutes the local reparameterization trick.
                print "adding noise to graph"
                eps = tf.random_normal(name='eps_%d'%l, mean=0.,
                            stddev=1.0, shape=[n_samples, 1, n_units[l]])
                layer_sigma = tf.sqrt(layer_var)
                layer += layer_sigma*eps
                with tf.name_scope(name+"Neural_Network_Activations_%d"%l):
                    tf.summary.histogram(name+"Layer_%d_sigmas"%l, layer_sigma)

            with tf.name_scope(name+"Neural_Network_Activations_%d"%l):
                tf.summary.histogram(name+"Layer_%d_activations_pre_tanh"%l, layer)

            # Add nonlinearity
            if l != (len(n_units) - 1):
                if self.nonlinearity == 'ReLU':
                    layer = tf.nn.relu(layer/self.length_scale)
                else:
                    assert self.nonlinearity == 'tanh'
                    layer = tf.nn.tanh(layer/self.length_scale)

            with tf.name_scope(name+"Neural_Network_Activations_%d"%l):
                tf.summary.histogram(name+"Layer_%d_activations_post_tanh"%l,layer)

            prev_layer = layer
            self.layers.append(prev_layer)
        self.KL_BNN = KL
        return prev_layer

    def set_linear_flow(self, b, name=""):
        """set_linear_flow constructs the first stage of normalizing flow,
        which is a linear flow.

        This function adds the necessary parameters to self.params.

        The offset term is entirely a funtion of network output

        Args:
            b: the output of the network, to be used for defining the
                offset, b

        Returns: the flow object and the KL divergence for the approximate posterior
        """
        # the slope is learned by VI, establish approximate
        # posterior.  We use the initial level of observation
        # noise as the prior mean.
        log_m_prior_sigma = 2.0
        (log_m_mu, log_m_logstd), log_m, log_m_KL = utils.set_q(
                name=name+"log_m", mu_prior=utils.un_softplus(1.0), sigma_prior=log_m_prior_sigma,
                mu_init_mu=utils.un_softplus(self.init_sigma_obs), mu_init_sigma=0.0,
                sigma_init=self.init_sigma_params, n_samples=self.n_samples, save_summary=True)
        m = tf.nn.softplus(log_m, name=name+"linear_flow_slope")

        # actually create the flow
        flow = flows.LinearFlow(b=b, m=m)

        # add variational parameters of linear flow to list.
        KL = log_m_KL
        if self.learn_sigma_obs: self.params += [log_m_mu]
        if self.inference == 'VI': self.params += [log_m_logstd]

        # Create Tensorboard logs
        tf.summary.histogram(name+"LinearFlow_b",b)
        tf.summary.histogram(name+"LinearFlow_m",m)
        return flow, KL

    def set_flow(self, flow_id, output_z, output_a, output_b, lmbda=None,
            name="", prev_flow=None):
        """

        Args:
            flow_id: id for specifying name of flow
            output_z: output for parameter z
            output_a: output for parameter a
            output_b: output for parameter b
            prev_flow: provide a previous flow if chaining outputs of flows
                together.

        Returns:
            returns flow object and KL

        """
        # actually create flow
        flow = flows.RadialFlow(output_z, output_a, output_b, prev=prev_flow)

        tf.summary.histogram(name+"flow_%d_alpha_hat"%flow_id,output_a)
        tf.summary.histogram(name+"flow_%d_beta_hat"%flow_id,output_b)
        tf.summary.histogram(name+"flow_%d_zi"%flow_id,flow.z_0)
        return flow

    def construct_flow(self, outputs, y, n_flows, n_samples, lmbda=None,
            name="", chain_flows=False):
        """construct_flow builds and links together the normalizing flow and
        establishes the log likelihood of samples.

        We construct the flow parameterized by the batch-normalized outputs
        of a neural network.  This means that each output has zero mean and
        unit variance.

        However, we not not want to enforce this trait onto the parameters
        of the normalizing flows.  Instead we want to learn the expectation
        (like the bias term when we do not use batch_norm)

        The variance for each of these parameters is a hyper-parameter,
        which determines the extent to which the system is heteroscedastic.
        In the limit of this being very small, the outputs of the network
        have no bearing on the values of the paramters.  For large values,
        these parameters will change dramatically.

        The expected values have distinct interpretations for each of the
        parameters.

        For z_0, this is the std-dev of location of the inflection point of the
        radial flows.
        For beta, this is the std-dev of the log-derivative of the flows.
        For alpha, this is the std-dev of the pre-softplus rate of linear decay.

        Additionally, output_scaling controls the variance of the predicted
        offset from the normalizing flows.

        args:
            y: the placeholder tensor for the outputs to be passed through the
                flow.
            n_flows: number of stages in the flow.
            n_samples: number of MC samples
            chain_flows: set true to link flows together, which keeps
                the z_0s grounded in the output space.

        Returns:
            new parameters of flows (i.e. those not defined as outputs of
                the network) and the negative log likelihoods
        """
        all_flows = []
        print "outputs.shape", outputs.shape[-1]
        print "n_flows", n_flows
        assert outputs.shape[-1] == (n_flows-1)*3 + 1
        with tf.name_scope(name+"Normalizing_Flows"):
            with tf.variable_scope(name+'network'):
                ## Construct the Radial Flows
                out_idx = 0 # keep track of which output we are working with.

                ## Construct the first, linear stage of the flow.
                flow, KL_flow = self.set_linear_flow(outputs[:, :,
                    out_idx], name=name); out_idx += 1
                all_flows.append(flow)
                for f_i in range(1, n_flows):
                    flow = self.set_flow(f_i, outputs[:, :, out_idx],
                            outputs[:, :, out_idx+1], outputs[:, :,
                                out_idx+2], name=name,
                            prev_flow=flow if chain_flows else None)
                    all_flows.append(flow)
                    out_idx += 3

                ## Check that every output has been used
                assert out_idx == outputs.shape[-1]
        return all_flows, KL_flow

    def likelihood(self, y, all_flows):
        ### Link the stages of the flow together.  The zs are ordered from the
        # base distribution to the observation distribution.

        # in this case, this is the sum of log jacobian determinents.
        y = tf.stack([y]*self.n_samples)
        zs, log_dz0_dy = flows.link(all_flows, y[:, :, 0])

        # Consider the observed values mapped through flows and make histogram.
        self.z_0 = zs[-1]
        print "self.z_0.shape", self.z_0.shape
        print "log_dz0_dzy.shape", log_dz0_dy.shape
        tf.summary.histogram("z_0", self.z_0)

        # Define the base distribution that will be warped as unit gaussian
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        # Calculate the negative log likelihood
        log_p_base = dist.log_prob(self.z_0)
        nlog_ls = -(log_dz0_dy+log_p_base - tf.reduce_sum(tf.log(self.y_std)))
        nlog_l= tf.reduce_mean(nlog_ls)

        ### Do log sum exp
        self.nlog_ls_eval = -tf.reduce_logsumexp(-nlog_ls, axis=0)
        self.nlog_ls_eval += tf.log(float(self.n_samples))
        nlog_l_eval = tf.reduce_mean(self.nlog_ls_eval)
        return nlog_ls, nlog_l, nlog_l_eval
