from __future__ import division
import network
import flow_network_bayes
import utils
import flows
import tensorflow as tf
import numpy as np

class nade_bayes(flow_network_bayes.flow_network_bayes):
    """nade_bayes is a subclass of flow_network_bayes for networks with normalizing
    flows on top using nade for multi output.

    The flow networks consists of an MLP which is used to parameterize the
    target distribution, whose form is defined by a normalizing flow.
    """
    def __init__(self, n_flows=[1,1], alpha_std=.5, beta_std=1.0, z_std=1.0,
            init_sigma_params=1e-5, bayes_layers=None,
            noise_mag=1.0, init_sigma_obs=1.0, output_scaling=1.0, lmbda=0.5,
            length_scale=1., learn_ls=False, learn_lmbda=False,  inference='VI',
            w_prior_sigma=None, anneal_in_KL=False, nonlinearity='tanh',
            learn_sigma_weights=True, learn_beta_std=False,
            learn_sigma_obs=True, **kwargs):
        """initialize a network with normalizing flows.

        We use the same network structure for predicting p(y1|x) and
        p(y2|y1,x), though of course with different parameters.

        Args:
            n_flows: number of stages in the normalizing flows for each
                flow.
            init_sigma_params: the initial posterior variances for the weights.
            bayes_layers: list of layers in which to track posterior
                variances.
            output_scaling: multiplicative scaling on batch_normalized
                network outputs for offset term (b in linear flow)
            lmbda: interpolation between homoscedastic and heteroscedastic,
                as lmbda goes to 0, we have a homoscedastic system, as it goes
                to 1., the length scale is determined entirely based on the
                variation of the network output. Bounded (0,1)
            length_scale: defines a pre-activation scaling of hidden units.
                this can be thought of as defining the steepness of the
                nonlinearity, which in part controls the fucntion's length scale.
            learn_ls: set true to learn the lenght scale of the network as
                part of inference.
            inference: inference method to use, must be VI, MAP, or MLE
            learn_sigma_weights: set False to use 'weight noise/guassian
                dropout' variational approximation rather than mean field
                with learned variances.
            anneal_in_KL: set to True to start with a small KL penalty and
                anneal into full weight.  This can help reduce overpruning
                of hidden units.

        """
        self.noise_mag = noise_mag
        self.learn_sigma_obs = learn_sigma_obs
        self.w_prior_sigma = w_prior_sigma
        self.init_sigma_obs = init_sigma_obs
        self.alpha_std, self.beta_std, self.z_std = alpha_std, beta_std, z_std

        self.learn_sigma_weights = learn_sigma_weights
        assert lmbda <= 1. and lmbda >= 0.
        assert inference == 'MAP' or inference == 'VI' or inference == 'MLE'
        self.inference = inference
        self.lmbda_val = lmbda
        if learn_lmbda:
            self.log_lmbda = tf.get_variable('log_lmbda',
                    initializer=tf.constant(
                        [np.float32(np.log(lmbda/(1.-lmbda)))]*2))
            self.lmbda = tf.sigmoid(self.log_lmbda)
        else:
            self.lmbda = [lmbda]*2

        log_length_scale = utils.un_softplus(length_scale)
        self.log_length_scale =  tf.get_variable('log_length_scale',
                initializer=tf.constant(np.float32(log_length_scale)))
        self.output_scaling = tf.get_variable('output_scaling',
                initializer=tf.constant(output_scaling))
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

        n_outputs = 1
        n_outputs += 3*(n_flows[0]-1)
        network.model_base.__init__(self, n_outputs=n_outputs,
                construct_nn=False, **kwargs)

        # Now construct Neural networks.
        n_hidden_units1, n_hidden_units2 = kwargs['n_hidden_units']

        # Construct the First network
        n_units = [self.X.shape[-1]]+n_hidden_units1+[n_outputs]
        outputs_1 = self.construct_network(n_units, self.n_samples,
                noise_dim=self.noise_dim, name="y1")
        KL_BNN1 = self.KL_BNN

        # Construct the second network
        input_layer = tf.stack([tf.concat([self.x, self.y[:,:1]],axis=1)]*self.n_samples)
        n_in = self.X.shape[-1]+1
        n_outputs = 1
        n_outputs += 3*(n_flows[1]-1)
        n_units = [n_in]+n_hidden_units2+[n_outputs]
        outputs_2 = self.construct_network(n_units, self.n_samples,
                noise_dim=self.noise_dim, input_layer=input_layer, name="y2")
        KL_BNN2 = self.KL_BNN
        self.KL_BNN = KL_BNN2 + KL_BNN1

        ### Construct likelihood using normalizing flows
        # In this case, the likelihood is defined by our normalizing flow.
        self.flows1, self.KL_flows1 = self.construct_flow(
                outputs_1, self.y[:, :1], n_flows[0],
                n_samples=self.n_samples,lmbda=self.lmbda[0], name="y1_")
        self.flows2, self.KL_flows2 = self.construct_flow(
                outputs_2, self.y[:, 1:2], n_flows[1],
                n_samples=self.n_samples,lmbda=self.lmbda[1], name="y2_")
        self.params.append(self.output_scaling)
        self.KL_flows = self.KL_flows1 + self.KL_flows2

        # Get the log losses for individual samples and the mean log loss
        # both for individual samples and averaged across the posterior
        # Specifically, self.nlog_l represents the expected log likelihood
        # in the ELBO
        tf.summary.scalar("KL_BNN1", KL_BNN1)
        tf.summary.scalar("KL_BNN2", KL_BNN2)
        tf.summary.scalar("KL_flows1", self.KL_flows1)
        tf.summary.scalar("KL_flows2", self.KL_flows2)
        self.KL = self.KL_flows1 + KL_BNN1 + self.KL_flows2 + KL_BNN2
        self.nlog_ls, self.nlog_l, self.nlog_l_eval = self.likelihood(self.y)

        # If we are doing a 2-stage training, we will only optimize wrt this
        # full set of parameters in the 2nd stage.
        self.all_params = list(self.params)

        self.learn_ls, self.learn_lmbda = learn_ls, learn_lmbda
        if learn_ls: self.all_params.append(self.log_length_scale)
        if learn_lmbda: self.all_params.append(self.log_lmbda)

        iteration_after_switch = tf.cast(tf.abs(self.epoch-self.epoch_switch_opt)+
            (self.epoch-self.epoch_switch_opt), tf.float32)
        if anneal_in_KL:
            assert inference != 'MLE'
            KL_weight =  1. - tf.exp(-0.03*(tf.cast(tf.abs(self.epoch-self.epoch_switch_opt)+
                (self.epoch-self.epoch_switch_opt), tf.float32))/2.)
        elif inference == 'MLE':
            KL_weight =  0.
        else:
            KL_weight = 1.

        ### Construct Cost (likelihood and KL)
        self.KL_weighted =  self.KL*KL_weight # weight if reverse anealing KL in.
        # scale expected log likelihood by number of datapoints
        self.cost = self.nlog_l*self.Y.shape[0] + self.KL_weighted
        self.set_summaries()

        ### set 2 optimizer stages
        self.construct_optimizer()

    def likelihood(self, y):
        ### Link the stages of the flow together.  The zs are ordered from the
        # base distribution to the observation distribution.

        # in this case, this is the sum of log jacobian determinents.
        y = tf.stack([y]*self.n_samples)
        zs1, log_dz0_dy1 = flows.link(self.flows1, y[:, :, 0])
        zs2, log_dz0_dy2 = flows.link(self.flows2, y[:, :, 1])

        # Consider the observed values mapped through flows and make histogram.
        self.z_01, self.z_02 = zs1[-1], zs2[-1]
        tf.summary.histogram("z_0_d1", self.z_01)
        tf.summary.histogram("z_0_d2", self.z_02)

        # Define the base distribution that will be warped as unit gaussian
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        # Calculate the negative log likelihood
        log_p_base = dist.log_prob(self.z_01)
        log_p_base += dist.log_prob(self.z_02)
        nlog_ls = -(log_dz0_dy1+log_dz0_dy2+log_p_base - tf.reduce_sum(tf.log(self.y_std)))
        nlog_l = tf.reduce_mean(nlog_ls)

        ### Do log sum exp
        self.nlog_ls_eval = -tf.reduce_logsumexp(-nlog_ls, axis=0)
        self.nlog_ls_eval += tf.log(float(self.n_samples))
        nlog_l_eval = tf.reduce_mean(self.nlog_ls_eval)
        return nlog_ls, nlog_l, nlog_l_eval
