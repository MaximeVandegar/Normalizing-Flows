""" This Module contains normaling flow mappings and inverses"""
import tensorflow as tf
import utils
import numpy as np

class LinearFlow:
    """LinearFlow is a normalizing flow to capture variance and mean
    """
    def __init__(self, b, m):
        """__init__ initializes the flow.

        out = m*(x-b)
        Args:
            b: offset of linear function
            m: slope of linear function
        """
        self.m, self.b = m, b

    def project(self, z):
        """project is the forward mapping
        """
        return z*self.m + self.b

    def invert(self, out):
        """invert returns the inverse of the flow called on out, and its derivative

        """
        z_in, dz_dy  = (out - self.b) / self.m, 1. / self.m
        return z_in, dz_dy

    def project_old(self, z):
        """project is the forward mapping
        """
        return (z-self.b)*self.m

    def invert_old(self, out):
        """invert returns the inverse of the flow called on out, and its derivative

        """
        z_in, dz_dy  = out / self.m + self.b, 1. / self.m
        return z_in, dz_dy

class LeakyReLUFlow:
    """LeakyReLUFlow is a normalizing flow that is nonlinear
    """
    def __init__(self, b, m):
        """__init__ initializes the flow.

        out = m1*x+ I[x>b](x-b)(m2-m1)+I[b<0](m2-m1)b
        where m1*m2 = 1
        Args:
            b: offset of linear function
            m: slope of linear function
        """
        self.m, self.m2, self.b = m, 1./m, b

        self.m_inv, self.m2_inv = 1./self.m, 1./self.m2

        # if b is greater than zero (i.e. the kink is after the origin), the bias of the inverse is simlpe b*m
        # if the bias is less than zero, the kink is at m2*b, where b is negative, so = -|b*m2|

        # b_inv    =  b*m1    -     (       I[b < 0]      *     |b| * (  m2  + m1 ) )
        self.b_inv = b*self.m + tf.to_float(tf.less(self.b, 0.))*b*(self.m2+self.m)

    def mapping(f_in, m, m2, b):
        """mapping returns mapping of a piecewise linear function

        Args:
            f_in: iinput tensor
            m: slope of the left linear segment
            m2: slope of the right linear segment
            b: the location (on x-axis) of the kink between the two slopes
        """
        f_out = m*f_in

        # add in slow from m2
        f_out += tf.to_float(tf.less(b, f_in))* (m2-m)*(f_in-b) # I[b < f_in]* (m2-m1)*(f_in-b)

        # adjust for the bias term, if the kink is before the origin.
        f_out += tf.to_float(tf.less(b,0))*(m2-m)*b # -(I[b<0]*|b|*(m2-m1))
        return f_out

    def project(self, z):
        """project is the forward mapping
        """
        return LeakyReLUFlow.mapping(z, self.m, self.m2, self.b)

    def invert(self, out):
        """invert returns the inverse of the flow called on out, and its derivative

        """
        z_in =  LeakyReLUFlow.mapping(out, self.m_inv, self.m2_inv, self.b_inv)
        dz_dy = tf.gradients(z_in, [out])[0]
        #dz_dy = self.m_inv + tf.to_float(tf.less(self.b_inv, out))*(self.m2_inv-self.m_inv)
        return z_in, dz_dy

class RadialFlow:
    """RadialFlow provides an implementation of a normalizing flow presented in Rezende et al. 2015

    However, in contrast to this method, we define it the reverse direction
    (since we are concerned only with the inverse function and its derivative.)


    f(z) = z_0 + r*\hat
    """
    def __init__(self, z_0, alpha_raw, beta_raw, prev=None):
        """__init__ initializes the flow object, if a previous flow is
        provided, the radius, z_0, is set to be the provided z_0 passed through
        the previous inverse flow, such that it is stationed in the correct
        location.

        Args:
            z_0: the location of the center point in the output space.
            alpha_raw: controls how dispersed the warping is.
            beta_raw: controls the magnitude of warping
        """
        alpha = tf.nn.softplus(alpha_raw)
        beta = tf.exp(beta_raw) -1. # this is bounded to be above -1
        self.alpha, self.beta = alpha, beta
        if prev is None:
            self.z_0 = z_0
        else:
            self.z_0, _ = prev.invert(z_0)

    def invert(self, out):
        r = tf.abs(out - self.z_0)
        h = (self.beta*self.alpha)/(self.alpha + r)

        z_in = out + h*(out-self.z_0)
        dz_dy = tf.gradients(z_in, [out])[0]
        return z_in, dz_dy

class RadialFlow2DNew:
    """RadialFlow2DNew provides an implementation of an adaptation of the
    normalizing flow presented in Rezende et al. 2015

    However, in contrast to this method, we define it the reverse direction
    (since we are concerned only with the inverse function and its derivative.)

    Additionally, we make nonlinearity in the flow asymetric.  The orientation
    of the asymetry is given by theta, the extent of how squashed it is, is
    given by kappa.

    if kappa is 0, the nonlinearity does not occur along the orthogonal
    dimension...

    if theta = 0, the nonlinearity is most quickly significant along the x1 direction,
    if theta = pi/2 or -pi/2, the nonlinearity is most quickly significant along the x2 direction,


    """
    def __init__(self, z_0, alpha_raw, beta_raw, theta_raw, prev=None):
        """__init__ initializes the flow object, if a previous flow is
        provided, the radius, z_0, is set to be the provided z_0 passed through
        the previous inverse flow, such that it is stationed in the correct
        location.

        Args:
            z_0: the 2D location of the center point in the output space.
            alpha_raw: controls how dispersed the warping is.
            beta_raw: controls the slope of warping
            theta_raw: rotation
            prev: previous flow, if provided z_0 is passed through that to
                anchor it's location in the output.
        """
        alpha = tf.nn.softplus(alpha_raw)
        beta = tf.exp(beta_raw) -1. # this is bounded to be above -1
        theta = tf.tanh(theta_raw)*np.pi/2
        #print "init flow, z_0.shape",z_0.shape
        #print "z_0 input shape", z_0.shape
        #print "theta shape", theta.shape
        gridlocked = False
        if gridlocked:
            self.A = tf.stack([
                tf.stack([1.,0.]),
                tf.stack([0.,1.])
                ])
        else:
            self.A = tf.stack([
                tf.stack([tf.cos(theta), -tf.sin(theta)]),
                tf.stack([tf.sin(theta), tf.cos(theta)])
                ])

        if len(self.A.shape) != 2:
            self.A = tf.transpose(self.A,[2,3,0,1])

        self.alpha, self.beta, self.theta = alpha, beta, theta
        if prev is None:
            self.z_0 = z_0
        else:
            self.z_0, _ = prev.invert(z_0)

    def invert(self, out):
        out_sub_z0 = out-self.z_0
        if len(out_sub_z0.shape) != 2:
            out_sub_z0_proj = tf.multiply(self.A, out_sub_z0[:,:,None,:])
            out_sub_z0_proj = tf.reduce_sum(out_sub_z0_proj,axis=-1)
            out_proj = tf.multiply(self.A, out[:, :, None, :])
            out_proj = tf.reduce_sum(out_proj, axis=-1)
            # now shape is [n_samples, ?, 2]
        else:
            out_sub_z0 = tf.transpose(out_sub_z0)
            out_sub_z0_proj = tf.matmul(self.A, out_sub_z0)
            out_sub_z0_proj = tf.transpose(out_sub_z0_proj)

        # We want to do this per dimension...
        r = tf.abs(out_sub_z0_proj)
        h = 1./(self.alpha + r)

        # in a departure from the original asymetric implementation, we use
        # out_proj, rather than out.  This ensures the function is
        # monotonically increasing.
        z_in = out_proj + self.beta*self.alpha*h*out_sub_z0_proj
        if len(z_in.shape) != 2:
            jacobian_matrices = tf.stack([tf.gradients(
                z_in[:,:,i], out, name="jacobian_matrix"
                )[0] for i in range(2)], axis=2)
            jacobian_determinant = tf.stack(tf.matrix_determinant(
                jacobian_matrices, name="jacobian_determinant"))
        else:
            jacobian_matrix = tf.stack([tf.gradients(
                z_in[:,i], out, name="jacobian_matrix"
                )[0] for i in range(2)],axis=1)
            jacobian_determinant = tf.matrix_determinant(jacobian_matrix,
                    name="jacobian_determinant")
        return z_in, jacobian_determinant

class RadialFlow2D:
    """RadialFlow2D provides an implementation of an adaptation of the
    normalizing flow presented in Rezende et al. 2015

    However, in contrast to this method, we define it the reverse direction
    (since we are concerned only with the inverse function and its derivative.)

    Additionally, we make nonlinearity in the flow asymetric.  The orientation
    of the asymetry is given by theta, the extent of how squashed it is, is
    given by kappa.

    if kappa is 0, the nonlinearity does not occur along the orthogonal
    dimension...

    if theta = 0, the nonlinearity is most quickly significant along the x1 direction,
    if theta = pi/2 or -pi/2, the nonlinearity is most quickly significant along the x2 direction,


    """
    def __init__(self, z_0, alpha_raw, beta, theta_raw, prev=None,
            pos_beta=False):
        """__init__ initializes the flow object, if a previous flow is
        provided, the radius, z_0, is set to be the provided z_0 passed through
        the previous inverse flow, such that it is stationed in the correct
        location.

        Args:
            z_0: the 2D location of the center point in the output space.
            alpha: controls how dispersed the warping is.
            beta: controls the magnitude of warping
            theta: rotation
            prev: previous flow, if provided z_0 is passed through that to
                anchor it's location in the output.
        """
        theta = tf.tanh(theta_raw)*np.pi/2
        gridlocked = False
        if gridlocked:
            self.A = tf.stack([
                tf.stack([1.,0.]),
                tf.stack([0.,1.])
                ])
        else:
            self.A = tf.stack([
                tf.stack([tf.cos(theta), -tf.sin(theta)]),
                tf.stack([tf.sin(theta), tf.cos(theta)])
                ])

        # Sometimes, better performance is observed when betas for both
        # dimensions have the same sign.  We enforce this here.
        if pos_beta:
            beta = -tf.nn.softplus(beta)
        else:
            beta = tf.nn.softplus(beta)

        if len(self.A.shape) != 2:
            self.A = tf.transpose(self.A,[2,3,0,1])

        # Beta must be greater than -alpha, we enforce this through the
        # opposite condition, alpha > -Beta.
        alpha_min = (tf.abs(beta)-beta)/2. # this is 0 if beta > 0
        alpha = tf.log(tf.exp(alpha_raw)+1.) + alpha_min
        self.alpha, self.beta, self.theta = alpha, beta, theta

        # We thead the cetner points throught the previous flows if given.
        if prev is None:
            self.z_0 = z_0
        else:
            self.z_0, _ = prev.invert(z_0)

    def invert(self, out):
        out_sub_z0 = out-self.z_0
        if len(out_sub_z0.shape) != 2:
            out_sub_z0_proj = tf.multiply(self.A, out_sub_z0[:,:,None,:])
            out_sub_z0_proj = tf.reduce_sum(out_sub_z0_proj,axis=-1)
            # now shape is [n_samples, ?, 2]
        else:
            out_sub_z0 = tf.transpose(out_sub_z0)
            out_sub_z0_proj = tf.matmul(self.A, out_sub_z0)
            out_sub_z0_proj = tf.transpose(out_sub_z0_proj)

        # We want to do this per dimension...
        r = tf.abs(out_sub_z0_proj)
        h = 1./(self.alpha + r)
        z_in = out + self.beta*h*out_sub_z0_proj

        if len(z_in.shape) != 2: # i.e. if sampling or input dependent.
            jacobian_matrices = tf.stack([tf.gradients(
                z_in[:,:,i], out, name="jacobian_matrix")[0] for i in
                range(2)], axis=2)#[j] for j in range(z_in.get_shape().as_list()[0])]
            jacobian_determinants = tf.matrix_determinant(jacobian_matrices,
                name="jacobian_determinant")
            jacobian_determinant = tf.stack(jacobian_determinants)
        else:
            jacobian_matrix = tf.stack([tf.gradients(
                z_in[:,i], out, name="jacobian_matrix"
                )[0] for i in range(2)],axis=1)
            jacobian_determinant = tf.matrix_determinant(jacobian_matrix,
                    name="jacobian_determinant")
        return z_in, jacobian_determinant

# Store layers weight & bias
def construct_radial_network(x, n_flows, n_input, n_hidden_units,
        reuse=False, linear_first=True, predict_var=False, n_samples=1,
        keep_prob=1., fixed_flows=False):
    """construct_radial_network creates a radial flow network whose parmeters
    are defined as the output of an MLP.

    Args:
        x: input tensor
        n_flows: the number of stages in the normalizing flow
        n_input: number of input dimensions
        n_hidden_units: list containing the number of hidden units in each layer
        reuse: theano flag for if allocating additional memory.
        linear_first: if first stage of the flow should be a linear flow
        predict_var: set to have the slope of the linear flow predicted
        n_samples: if using dropout, the number of samples to use.
        keep_prob: the probability of dropping hidden units.
        fixed_flows: set True to not parameterized flows as a function of the
            input.

    Returns:
        the weights, biases and list of output samples.  If n_samples is 1, we
        return just the output (not as a list)

    """
    ### Check For Valid input
    if n_samples != 1: assert keep_prob != 1. # only sample if using dropout
    if predict_var: assert linear_first # only a valid option if using linear stage

    # Define weights and Biases
    n_units = [n_input] + n_hidden_units
    n_layers = len(n_hidden_units)
    with tf.variable_scope('radial_network', reuse=reuse):
        # if we run this more than once, we reuse the scope to avoid
        # redefining the variables.
        weights = {
            'w%d'%i: tf.get_variable(
                'weights%d'%i,shape=[n_units[i-1], n_units[i]],
                initializer=tf.contrib.layers.xavier_initializer(),)
            for i in range(1,n_layers+1)
        }
        biases = {
            'b%d'%i: tf.get_variable("biases%d"%i,
                initializer = tf.constant(
                    np.random.normal(size=[n_units[i]]))
                ) for i in range(1,n_layers+1)
        }
        for i in range(n_flows):
            n_outputs = 2 if linear_first and i==0 else 3
            biases['out_%02d'%i]=tf.get_variable("bias_output_%02d"%i,
                initializer= tf.constant(np.random.normal(size=[n_outputs]))
            )
            # To learn a fixed flow (other than the first flow) don't add
            # output weights
            if fixed_flows and i > 0:
                continue
            weights['out_%02d'%i]=tf.get_variable(
                'output_weights_%02d'%i,shape=[n_units[-1], n_outputs],
                initializer=tf.contrib.layers.xavier_initializer()
            )
    out_layers_samples = utils.multilayer_perceptron(
            x, weights, biases, n_flows, n_layers, keep_prob=keep_prob,
            n_samples=n_samples
            )
    if linear_first and not predict_var:
        print("Fixing Variance to be shared across all points")
        for out_layers in out_layers_samples:
            out_layers["flow_00"] *= np.array([0.,1.])
            out_layers["flow_00"] += np.array([1.,0.])*biases["out_00"]

    # Define Normalizing Flow
    flows_samples = []
    for out_layers in out_layers_samples:
        flows = []
        for f_l in range(n_flows):
            if f_l ==0 and linear_first:
                # set bias and slope
                b, m =  out_layers["flow_%02d"%f_l][:,1], tf.exp(out_layers["flow_%02d"%f_l][:,0],name="slope_%02d"%f_l)
                flow = LinearFlow(b=b, m=m)
            else:
                #TODO: Why is this exp???
                z_0 = tf.exp(out_layers["flow_%02d"%f_l][:,0],name="z0_%02d"%f_l)
                ### Rezende parameterization
                #alpha = tf.nn.softplus(out_layers["flow_%02d"%f_l][:,1],name="alpha_%02d"%f_l)
                #beta = -alpha + tf.nn.softplus(out_layers["flow_%02d"%f_l][:,2],name="beta_%02d"%f_l)

                # New parameterization
                beta = out_layers["flow_%02d"%f_l][:,2]
                alpha_min = (tf.abs(beta)-beta)/2.
                alpha = tf.log(tf.exp(out_layers["flow_%02d"%f_l][:,1])+1.)+alpha_min
                flow = RadialFlow(z_0, alpha, beta)
            flows.append(flow)
        if n_samples == 1:
            return weights, biases, flows
        flows_samples.append(flows)
    return weights, biases, flows_samples

def link(flows, y):
    zs_bkd = []
    log_dz_dy_bkd = []
    for i, flow in enumerate(flows):
        z_bkd, dz_dy = flow.invert(y if i == 0 else zs_bkd[-1])
        tf.summary.histogram('dz_dy_%d'%i, dz_dy)
        zs_bkd.append(z_bkd)
        log_dz_dy_bkd.append(tf.log(dz_dy))
    zs_bkd, log_dz_dy_bkd = list(zs_bkd), list(log_dz_dy_bkd)
    log_dz_dy_total = 0.
    for log_dz_dy in log_dz_dy_bkd:
        log_dz_dy_total += log_dz_dy
    return zs_bkd, log_dz_dy_total
