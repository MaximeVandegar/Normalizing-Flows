from __future__ import division
import network
import tensorflow as tf
import numpy as np

class mixture_density_network(network.model_base):
    """mixture_density is a subclass of model_base for networks with a mixture
    of Gaussians likelihood model.

    The mixing proportions are defined by a softmax transformed output.

    The means and variances may also be outputs
    """
    def __init__(self, n_components=1, input_dependent=False,
            r_mag_W= 0., **kwargs):
        """initialize a network with normalizing flows.

        Args:
            n_components: number of mixture components.
        """

        ### Determine number of outputs based on number of flows and types
        self.input_dependent = input_dependent
        self.n_components = n_components

        # 1 output for predicting offset.
        # if input dependent, then predict mixing proportion, mean and variance
        # for each mixing component.
        n_outputs = 1
        if self.input_dependent: n_outputs += self.n_components*3

        network.model_base.__init__(self, n_outputs=n_outputs, **kwargs)
        print "nework outputs.shape", self.outputs.shape

        ### Construct likelihood using normalizing flows
        # In this case, the likelihood is defined by our normalizing flow.
        additional_params = self.construct_mog(
                self.outputs, input_dependent
                )
        self.nlog_ls, self.nlog_l = self.likelihood(self.y)
        tf.summary.histogram("nlog_ls", self.nlog_ls)
        tf.summary.scalar("nlog_l", self.nlog_l)

        # If we are doing a 2-stage training, we will only optimize wrt this
        # full set of parameters in the 2nd stage.
        self.all_params.extend(additional_params)

        ### Construct Cost (likelihood and regularizers)
        self.cost = self.nlog_l
        self.add_weight_decay(r_mag_W)

        tf.summary.scalar("cost", self.cost)

        ### set 2 optimizer stages
        self.construct_optimizer()

    def construct_mog(self, outputs, input_dependent=False):
        """construct_flow builds and links together the normalizing flow and
        establishes the log likelihood of samples.

        args:
            outputs: the outputs of the neural network which we will use to
                parameterize the flows.
            input_dependent: True if predicting the components and proportions
            should be a function of the input.

        Returns:
            new parameters of mog (i.e. those not defined as outputs of
                the network), and the negative log likelihoods
        """
        mog_params = []
        # check for correct number of input dimensions.
        if input_dependent:
            assert outputs.shape[-1] == (self.n_components)*3 + 1
        else:
            assert outputs.shape[-1] == 1

        out_idx = 0 # keep track of which output we are working with.
        self.shift = outputs[0, :, out_idx:out_idx+1]; out_idx += 1
        with tf.name_scope("Mixture_of_Gaussians"):
            with tf.variable_scope('network'):
                # get mixing proportions
                if input_dependent:
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
                else:
                    theta_raw = tf.get_variable('theta', shape=[self.n_components],
                            initializer=tf.constant_initializer(0.0))
                    self.theta = tf.nn.softmax(theta_raw)

                    log_sigmas = tf.get_variable('log_sigmas', shape=[self.n_components],
                            initializer=tf.constant_initializer(0.0))
                    self.sigmas = tf.exp(log_sigmas,name="sigmas")

                    self.mus =  tf.get_variable('mus', initializer=tf.constant(
                        np.random.normal(size=[self.n_components],scale=0.75).astype(np.float32))
                        )

                    ### Add these to the core set of parameters
                    self.params.extend([theta_raw, log_sigmas, self.mus])
                    mog_params.extend([theta_raw, log_sigmas, self.mus])

                    for k in range(self.n_components):
                        tf.summary.scalar("Gaussian_%d_proportion"%k,self.theta[k])
                        tf.summary.scalar("Gaussian_%d_sigma"%k,self.sigmas[k])
                        tf.summary.scalar("Gaussian_%d_mu"%k,self.mus[k])

        ## Check that every output has been used
        assert out_idx == outputs.shape[-1]
        return mog_params

    def likelihood(self, y):
        ### Construct Likelihood for MOG
        dist = tf.contrib.distributions.Normal(loc=self.mus,scale=self.sigmas)
        print "y.shape", y.shape
        if len(y.shape) == 2:
            obs = tf.transpose([y[:,0]-self.shift[:, 0]]*self.n_components)
        else:
            obs = tf.transpose([y[:,0]-self.shift[:,0]]*self.n_components,[1,2,0])

        likelihoods = dist.prob(obs)
        likelihoods = likelihoods*self.theta
        print "likelihoods shape(pre reduce): ",likelihoods.shape
        likelihoods = tf.reduce_sum(likelihoods, axis=-1,keep_dims=True)
        print "likelihoods shape(post reduce): ",likelihoods.shape

        # Calculate the negative log likelihood
        nlog_ls = -(tf.log(likelihoods) - tf.log(self.y_std))
        nlog_l = tf.reduce_mean(nlog_ls)
        return nlog_ls, nlog_l
