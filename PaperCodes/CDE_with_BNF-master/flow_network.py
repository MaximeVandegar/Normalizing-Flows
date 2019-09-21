from __future__ import division
import network
import utils
import flows
import tensorflow as tf
import numpy as np

# TODO write and use inverse softplus function to reduce redundancy of
# rewriting.

class flow_network(network.model_base):
    """flow_network is a subclass of model_base for networks with normalizing
    flows on top.

    The flow networks consists of an MLP which is used to parameterize the
    target distribution, whose form is defined by a normalizing flow.
    """
    def __init__(self, n_flows=1, predict_var=False, input_dependent=True,
            r_mag_W=0., r_mag_alpha=0., r_mag_beta=0., r_mag_z_0=0., **kwargs):
        """initialize a network with normalizing flows.

        Args:
            n_flows: number of stages in the normalizing flows.
            input_dependent: if the parameter of the flow should be a
                function of the input.
        """

        ### Determine number of outputs based on number of flows and types
        self.input_dependent = input_dependent
        if kwargs.has_key('noise_dim') and  kwargs['noise_dim'] != 0:
            assert input_dependent

        # 1 output for predicting mean, 2 if also variance.  Then add an
        # additional 3 paramters for each of input-dependent the radial flows
        n_outputs = 2 if predict_var else 1
        if input_dependent: n_outputs += 3*n_flows

        network.model_base.__init__(self, n_outputs=n_outputs, **kwargs)
        print "network outputs.shape", self.outputs.shape

        ### Construct likelihood using normalizing flows
        # In this case, the likelihood is defined by our normalizing flow.
        additional_params = self.construct_flow(
                self.outputs, self.y, n_flows+1, predict_var, input_dependent
                )

        self.nlog_ls, self.nlog_l = self.likelihood(self.y)
        tf.summary.histogram("nlog_ls",self.nlog_ls)
        tf.summary.scalar("nlog_l",self.nlog_l)

        # If we are doing a 2-stage training, we will only optimize wrt this
        # full set of parameters in the 2nd stage.
        self.all_params.extend(additional_params)

        ### Construct Cost (likelihood and regularizers)
        self.cost = self.nlog_l
        if r_mag_W != 0.:
            self.add_weight_decay(r_mag_W)
        if r_mag_alpha != 0 or r_mag_beta != 0 or r_mag_z_0 !=0:
            self.add_flow_regularization(r_mag_alpha, r_mag_beta, r_mag_z_0)
        tf.summary.scalar("cost", self.cost)

        ### Create 2 optimizer stages
        self.construct_optimizer()

    def construct_flow(self, outputs, y, n_flows, predict_var,
            input_dependent=False):
        """construct_flow builds and links together the normalizing flow and
        establishes the log likelihood of samples.

        args:
            outputs: the outputs of the neural network which we will use to
                parameterize the flows.
            y: the placeholder tensor for the outputs to be passed through the
                flow.
            n_flows: number of stages in the flow.
            predict_var: true is predicting slope of first flow
            input_dependent: True if predicting the variance of flows
                after the first one.

        Returns:
            new parameters of flows (i.e. those not defined as outputs of
                the network) and the negative log likelihoods
        """
        self.flows, flow_params = [], []
        # check for correct number of input dimensions.
        if input_dependent or (self.noise_dim != 0):
            assert outputs.shape[-1] == (n_flows-1)*3 + (2 if predict_var else 1)
        else:
            assert outputs.shape[-1] == 2 if predict_var else 1

        out_idx = 0 # keep track of which output we are working with.
        with tf.name_scope("Normalizing_Flows"):
            with tf.variable_scope('network'):

                ## Construct first flow, a Linear Flow
                b = outputs[:, :,out_idx]; out_idx += 1
                tf.summary.histogram("LinearFlow_b",b)
                if predict_var:
                    log_m = outputs[:, :, out_idx]; out_idx += 1
                else:
                    log_m = tf.get_variable('log_m', shape=[],
                            initializer=tf.constant_initializer(0.0))
                    ### Add this to the core set of parameters
                    self.params.append(log_m)
                    flow_params.append(log_m)

                m = tf.exp(log_m,name="linear_flow_slope")
                if predict_var:
                    tf.summary.histogram("LinearFlow_m",m)
                else:
                    tf.summary.scalar("LinearFlow_m",m)
                self.flows.append(flows.LinearFlow(b=b, m=m))
                print "b.shape", self.flows[-1].b.shape
                print "m.shape", self.flows[-1].m.shape

                ## Construct the Subsequent Flows
                for f_i in range(1, n_flows):
                    if input_dependent or (self.noise_dim != 0):
                        z_i = outputs[:, :,out_idx]; out_idx += 1
                        beta_raw = outputs[:, :,out_idx]; out_idx += 1
                        alpha_raw = outputs[:, :,out_idx]; out_idx += 1
                    else:
                        z_i = tf.get_variable('z_%d'%f_i,
                                initializer=tf.constant(np.random.normal(size=[1],scale=0.25).astype(np.float32)))
                        beta_raw = tf.get_variable('beta_%d'%f_i, shape=[1],
                                initializer=tf.constant_initializer(0.0))
                        alpha_raw = tf.get_variable('alpha_%d'%f_i, shape=[1],
                                initializer=tf.constant_initializer(0.0))
                        flow_params.extend([z_i, beta_raw, alpha_raw])

                    # In the 1D regression case, it does not make sense ot
                    # thread the z_0's through previous flows, so prev=None
                    flow = flows.RadialFlow(z_i, alpha_raw, beta_raw,
                            prev=None)
                    self.flows.append(flow)

                    if input_dependent or (self.noise_dim != 0):
                        tf.summary.histogram("flow_%d_zi"%f_i,flow.z_0)
                        tf.summary.histogram("flow_%d_beta"%f_i,flow.beta)
                        tf.summary.histogram("flow_%d_alpha"%f_i,flow.alpha)
                    else:
                        tf.summary.scalar("flow_%d_zi"%f_i,flow.z_0[0])
                        tf.summary.scalar("flow_%d_beta"%f_i,flow.beta[0])
                        tf.summary.scalar("flow_%d_alpha"%f_i,flow.alpha[0])

        ## Check that every output has been used
        assert out_idx == outputs.shape[-1]

        return flow_params

    def likelihood(self, y):
        ### Link the stages of the flow together.  The zs are ordered from the
        # base distribution to the observation distribution.
        print "y.shape", y.shape
        y = tf.stack([y]*self.n_samples)
        zs, log_dz0_dy = flows.link(self.flows, y[:, :,0])

        # Consider the observed values mapped through flows and make histogram.
        self.z_0 = zs[-1]
        print "z_0.shape", self.z_0.shape
        print "log_dz0_dy.shape", log_dz0_dy.shape
        print "y_std.shape", self.y_std.shape
        tf.summary.histogram("z_0", self.z_0)

        # Define the base distribution that will be warped as unit gaussian
        dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        print "log_prob.shape",dist.log_prob(self.z_0)
        # Calculate the negative log likelihood
        self.log_dz0_dy = log_dz0_dy
        self.base_log_prob = dist.log_prob(self.z_0)
        nlog_ls = -(log_dz0_dy+dist.log_prob(self.z_0) - tf.log(self.y_std[0]))

        ### Do log sum exp, if only 1 sample this merely cuts the first dimension.
        nlog_ls = -tf.reduce_logsumexp(-nlog_ls, axis=0)
        nlog_ls += tf.log(float(self.n_samples))

        nlog_l = tf.reduce_mean(nlog_ls)
        return nlog_ls, nlog_l

    def add_flow_regularization(self, r_mag_alpha, r_mag_beta, r_mag_z_0):
        """add_flow_regularization adds regularization terms on the flows to
        the objective.

        Args:
            r_mag_alpha: penalty on 1/alpha
            r_mag_beta: L2 norm penalty on beta
            r_mag_z_0: L2 norm penaty on z_0
        """
        with tf.name_scope("regularization"):
            ## Add regularization to Normalizing Flow
            self.z_0_cost, self.beta_cost, self.alpha_cost = 0., 0., 0.
            for flow in self.flows[1:]:
                self.alpha_cost += tf.reduce_mean(1./flow.alpha)*r_mag_alpha
                self.beta_cost += tf.reduce_mean(flow.beta**2)*r_mag_beta
                self.z_0_cost += tf.reduce_mean(flow.z_0**2)*r_mag_z_0
            tf.summary.scalar('alpha_cost', self.alpha_cost)
            tf.summary.scalar('beta_cost', self.beta_cost)
            tf.summary.scalar('z_0_cost', self.z_0_cost)
        self.cost += self.beta_cost + self.z_0_cost + self.alpha_cost
        if len(self.flows) > 1:
            self.regularizers.update({
                    "Radius":self.z_0_cost, "Magnitude":self.beta_cost,
                    "Alpha":self.alpha_cost
                    })
