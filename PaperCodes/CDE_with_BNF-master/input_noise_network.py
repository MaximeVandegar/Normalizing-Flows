from __future__ import division
import network
import tensorflow as tf
import numpy as np

class input_noise_network(network.model_base):
    """input_noise_network is a subclass of model_base for networks with
    input noise

    """
    def __init__(self, r_mag_W=0.,  **kwargs):
        """initialize a network with input noise

        """
        ## for 1d outputs
        n_outputs =  1

        network.model_base.__init__(self, n_outputs=n_outputs, **kwargs)

        ### Construct likelihood using normalizing flows
        # In this case, the likelihood is defined by our normalizing flow.
        self.nlog_ls, self.nlog_l = self.likelihood(self.y)
        tf.summary.histogram("nlog_ls",self.nlog_ls)
        tf.summary.scalar("nlog_l",self.nlog_l)

        # If we are doing a 2-stage training, we will only optimize wrt this
        # full set of parameters in the 2nd stage.

        ### Construct Cost (likelihood and regularizers)
        self.cost = self.nlog_l
        self.add_weight_decay(r_mag_W)
        tf.summary.scalar("cost", self.cost)

        ### set 2 optimizer stages
        self.construct_optimizer()

    def likelihood(self, y):
        ### Link the stages of the flow together.  The zs are ordered from the
        # base distribution to the observation distribution.

        # in this case, this is the sum of log jacobian determinents.
        errors = self.outputs-self.y

        sigma = tf.Variable( 1.0, name="sigma")
        self.params.append(sigma)
        self.all_params.append(sigma)
        dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=np.zeros(1, dtype=np.float32),
                scale_diag=sigma*np.ones(1, dtype=np.float32)
                )
        # Calculate the negative log likelihood
        nlog_ls = -(dist.log_prob(errors) - tf.reduce_sum(tf.log(self.y_std)))

        ### Do log sum exp
        if self.n_samples != 1:
            nlog_ls = -tf.reduce_logsumexp(-nlog_ls, axis=0)
            nlog_ls += tf.log(float(self.n_samples))
        nlog_l = tf.reduce_mean(nlog_ls)
        return nlog_ls, nlog_l
