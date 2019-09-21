from __future__ import division
import os
import pickle
from sklearn.model_selection import train_test_split
import synthetic_data
import utils
import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class model_base:
    """model_base provides a base class for building neural network models with
    arbitrary likelihoods for one dimensional regression tasks.

    """

    def __init__(self, data_dir_base=".", dataset="", n_outputs=1, n_hidden_units=[50],
            lr=0.001, n_epochs=1000, summary_fn="", keep_p=1.0,
            log_base_dir='logs', display_freq=100, batch_size=10**10,
            log_to_stdout=True, n_samples=1, noise_dim=0,
            nonlinearity="ReLU", standardize_data=True,
            epoch_switch_opt=10**10, n_pts=None, log_image_summary=True,
            plot_pts=True, plot_title=None, construct_nn=True,
            restore_epoch=None, plot_interval=True, save_model=False, init_params=None):
        """initialize a network

        Args:
            data_dir_base: directory with all UCI datsets and splits.
            dataset: toy, mog or the name of a UCI dataset
            n_outputs: number of outputs of the neural network
            n_hidden_units: a list of the number of units in each hidden
                layer.
            lr: learning rate
            log_base_dir: the directory in which to store logs.
            batch_size: set very high by default as to have only 1 batch
            epoch_switch_opt: the epoch at which to transition to training
                all parameters.
            n_pts: numbere of points generate if synthetic data
            plot_pts: if datapoint are to be plotted on predictive
                distribution plot
            plot_title: Title for saved plots
            restore_epoch: epoch at which to restore model to
        """
        ### Parameters of the Optimization Procedure
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.epoch = tf.placeholder('int32', [],  name='epoch')
        self.display_freq = display_freq
        self.restore_epoch = restore_epoch
        self.save_model = save_model
        self.standardize_data = standardize_data
        self.lr = lr
        self.plot_pts = plot_pts
        self.plot_title = plot_title
        self.plot_interval = plot_interval
        self.regularizers = {}
        self.bn_updates = []
        # Switch over to late training optimizer at this epoch.
        self.epoch_switch_opt = epoch_switch_opt
        self.log_to_stdout = log_to_stdout
        self.n_samples = n_samples
        self.noise_dim = noise_dim

        ### Set base directory for data
        self.summary_fn = summary_fn
        self.save_summaries = summary_fn is not ""
        self.data_dir_base = data_dir_base
        self.log_base_dir = log_base_dir
        self.summary_path = self.log_base_dir+summary_fn

        ### Load data
        # only allow no dataset and dir if using synthetic data
        assert dataset is not ""
        self.dataset = dataset

        # this also establishes placeholders
        # we only ever specify the number of points for synthetic datasets
        if n_pts is not None: assert dataset == 'toy' or dataset == 'mog' or dataset == "toy_small"
        print "loading dataset"
        self.load_data(n_pts=n_pts)

        # Save images of the predictive distributions for 1D toy data
        self.log_image_summary = log_image_summary
        if self.log_image_summary:
            self.pred_prob_buff = utils.add_image_summary()

        ### Construct network
        self.params = [] # to keep parameters of network and noise distribution
        self.n_hidden_units = n_hidden_units
        self.weights = {}
        self.biases = {}

        if n_hidden_units != [] and construct_nn:
            print "constructing network"
            self.outputs = self.construct_network(
                    [self.n_inputs] + self.n_hidden_units +  [n_outputs],
                    n_samples=n_samples, noise_dim=noise_dim, keep_p=keep_p,
                    nonlinearity=nonlinearity, init_params=init_params
                    )

        # Make a copy of self.params, the parameters to train in the late stage
        # optimization.
        self.all_params = list(self.params)

    def construct_optimizer(self, beta1=0.9, beta2=0.99,
            scale_cost_by_N=True, sgd=False):
        ### Construct optimizers for early and late stage training
        if scale_cost_by_N:
            obj = self.cost/self.X.shape[0]
        else:
            obj = self.cost
        with tf.variable_scope('optimizer'):
            if sgd:
                self.optimizer_early = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(obj,
                        var_list=self.params)
                self.optimizer_late = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(obj,
                        var_list=self.all_params)
            else:
                self.optimizer_early = tf.train.AdamOptimizer(learning_rate=self.lr,
                        beta2=beta2,beta1=beta1).minimize(obj, var_list=self.params)
                self.optimizer_late = tf.train.AdamOptimizer(learning_rate=self.lr,
                        beta2=beta2,beta1=beta1).minimize(obj, var_list=self.all_params)

    def split(self, split_id, X=None, Y=None):
        """split uses the preset train/test splits from Miguel's Probabilistic
        Backpropagation paper.

        The indices defining these splits are in the data directory.

        If the X and Y are provided, we use these both as the training and
            the test sets.

        Args:
            the index of the split, between 0 and 19

        """
        ### Load indices of training and test sets
        if X is not None and Y is not None:
            self.X_train, self.X_test, self.Y_train, self.Y_test = X, X, Y, Y
            #self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            #        X, Y, test_size=0.10, random_state=42)
        elif self.dataset == 'toy' or self.dataset == 'mog' or \
            self.dataset == 'nyc' or self.dataset=='nyc_taxi' or \
            self.dataset == 'toy_small' or self.dataset=='chicago':
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                    self.X, self.Y, test_size=0.10, random_state=42)
        else:
            data_dir = self.data_dir
            test_idxs = np.loadtxt(data_dir+"index_test_{}.txt".format(split_id), dtype=int)
            train_idxs = np.loadtxt(data_dir+"index_train_{}.txt".format(split_id), dtype=int)
            self.X_train, self.Y_train = self.X[train_idxs,:], self.Y[train_idxs,:]
            self.X_test, self.Y_test = self.X[test_idxs,:], self.Y[test_idxs,:]

        # set number of batches
        self.n_batches = int(np.ceil(self.X_train.shape[0]/self.batch_size))

        ### Normalize based on Training data
        if self.standardize_data:
            self.Y_mean =  np.mean(self.Y_train, axis=0)
            self.Y_std = np.std(self.Y_train, axis=0)
            self.X_mean =  np.mean(self.X_train, axis=0)
            self.X_std = np.std(self.X_train, axis=0)
        else:
            self.Y_std = np.ones(self.Y_train[0].shape)
            self.Y_mean = np.zeros(self.Y_train[0].shape)
            self.X_std = np.ones(self.X_train[0].shape)
            self.X_mean =  np.zeros(self.X_train[0].shape)

        self.X_train = (self.X_train-self.X_mean)/self.X_std
        self.X_test = (self.X_test-self.X_mean)/self.X_std
        self.Y_train = (self.Y_train-self.Y_mean)/self.Y_std
        self.Y_test = (self.Y_test-self.Y_mean)/self.Y_std

    def load_data(self, n_pts=None, bimodal=True):
        """load_data loads the data and test set of the given split and scales
        the mean and variance accordingly.

        Data is stored as self.X and self.Y
        The number of inputs and targets are also stored.

        Args:
            n_pts: the number of pts to generate if synthetic data.
            bimodal: if synthetic data should be bimodal.

        """
        if self.dataset == 'toy':
            if n_pts == None:
                n_pts = 5000
            self.X, self.Y = synthetic_data.gen_synthetic_data(
                    n_pts=n_pts,bimodal=bimodal, dim=1,
                    heteroscedastic=True, asymetric=True)
        self.n_inputs, self.n_targets = self.X.shape[1], self.Y.shape[1]

        ### Establish placeholders for data and y_std
        with tf.name_scope("Data"):
            self.x = tf.placeholder("float", [None, self.n_inputs], name='X')
            self.y = tf.placeholder("float", [None, self.n_targets], name='Y')

            # We keep the std-dev of the labels a placeholder because we need
            # to use it for calculating likelihoods.
            self.y_std = tf.placeholder("float", [self.n_targets], name='y_std')

    def construct_network(self, n_units, n_samples=1, noise_dim=0,
            keep_p=1., nonlinearity="ReLU", input_layer=None,
            init_params=None,  name=""):
        """construct_network establishes all weight matrices and biases and
        connects them.

        The outputs may include parameters of the flow

        Args:
            n_units: the sizes of all layers including input and output
                layer
        """
        assert init_params is None # this is implemented only in the Bayesian flow version of this function


        ### Define parameters of the network
        with tf.variable_scope(name+'network'):
            self.weights.update({
                    name+'w_%d'%i: tf.get_variable(
                        name+'weights_%d'%i, shape=[n_units[i-1] + (noise_dim if i==1
                            else 0), n_units[i]],
                        initializer=tf.contrib.layers.xavier_initializer(),)
                    for i in range(1,len(n_units))
                    })
            self.biases.update({
                    name+'b_%d'%i: tf.get_variable(name+"biases_%d"%i,
                        initializer = tf.constant(
                            np.random.normal(size=[n_units[i]]).astype(np.float32)))
                        for i in range(1,len(n_units))
                        })
            self.params.extend(self.weights.values())
            self.params.extend(self.biases.values())
            if self.noise_dim != 0:
                self.noise_weights = self.weights[name+'w_1'][n_units[0]:]
                self.weights[name+'w_1'] = self.weights[name+'w_1'][:n_units[0]]

        ### If using multiple samples, duplicate the input to accomodate
        # multiple samples
        input_layer = self.x if input_layer is None else input_layer
        prev_layer = tf.stack([input_layer]*n_samples)

        ### Define activations in each layer
        for l in range(1,len(n_units)):
            # If sampling, we must flatten before matrix multiplication
            print "prev_layer initial shape",prev_layer.shape
            prev_shape = prev_layer.get_shape().as_list()
            assert len(prev_shape)==3  # shape should be [n_samples, ?, dim]
            prev_layer = tf.reshape(prev_layer, [-1, prev_shape[2]])
            print "prev_layer reshaped",prev_layer.shape
            layer = tf.add(tf.matmul(prev_layer,
                self.weights[name+'w_%d'%l]), self.biases[name+'b_%d'%l])
            # Then reshape the new layer to the orignal shape
            layer = tf.reshape(layer, [prev_shape[0], -1, layer.get_shape().as_list()[-1]])
            print "next layer shaped back",layer.shape, "\n"

            # Add input noise at the first layer.
            if l == 1 and self.noise_dim != 0:
                layer_noise = tf.matmul(tf.random_uniform([n_samples,
                    noise_dim]), self.noise_weights)
                # To add noise, we must expand in variable batch size
                # dimension.
                layer += layer_noise[:,None,:]

            # Hidden layers with RELU activation, but no nonlinearity at output
            if l != len(n_units) - 1:
                layer = tf.nn.dropout(layer, keep_prob=keep_p)
                if nonlinearity == 'ReLU':
                    layer = tf.nn.relu(layer)
                elif nonlinearity == 'softplus':
                    print "using softplus"
                    layer = tf.nn.softplus(layer)
                else:
                    assert nonlinearity == None
            prev_layer = layer
            with tf.name_scope(name+"Neural_Network_Activations_%d"%l):
                tf.summary.histogram(name+"Layer_%d_activations"%l,layer)
            with tf.name_scope(name+"Neural_Network_Params_%d"%l):
                tf.summary.histogram(name+"Layer_%d_weights"%l,self.weights[name+"w_%d"%l])
                tf.summary.histogram(name+"Layer_%d_biases"%l,self.biases[name+"b_%d"%l])
        return prev_layer

    def add_weight_decay(self, r_mag_W):
        """add_weight_decay adds weight decay regularization to the objective.

        Args:
            the magnitude of the L2 norm penalty
        """
        with tf.name_scope("regularization"):
            l2_cost = sum(tf.reduce_sum(W**2) for W in
                    self.weights.values())*r_mag_W
            tf.summary.scalar('l2_cost', l2_cost)
            self.cost += l2_cost
            self.regularizers.update({"L2":l2_cost})

    def train(self, sess=None):
        """train reinitializes all parameters and trains the model

        Returns:
           the mean negative log likelihoods on the test and training sets.
           These are averaged over the last epochs.

        """
        if sess is None:
            sess = tf.Session(config=config)
            sess_given = False
        else:
            sess_given = True
        self.sess = sess
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.summary_path+'_train', sess.graph)
        test_writer = tf.summary.FileWriter(self.summary_path + '_test')
        with sess.as_default():
            if not sess_given: tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            if self.restore_epoch is not None:
                ckpt_name = self.summary_path + "_epoch%05d_model.ckpt"%self.restore_epoch
                saver.restore(sess, ckpt_name)
                begin_epoch = self.restore_epoch
            else:
                begin_epoch = 0

            x, y = self.x, self.y
            X_train, Y_train = self.X_train, self.Y_train
            X_test, Y_test = self.X_test, self.Y_test
            mean_nlogls_train, mean_nlogls_test = [], []
            for epoch in range(begin_epoch, self.n_epochs):
                if epoch < self.epoch_switch_opt:
                    optimizer = self.optimizer_early
                else:
                    optimizer = self.optimizer_late
                avg_cost = 0.
                avg_nlog_l = 0.
                # Loop over all batches
                if self.regularizers:
                    reg_names, reg_tensors = zip(*self.regularizers.items())
                    reg_vals = {reg_name:0. for reg_name in reg_names}
                else:
                    reg_names, reg_tensors = [], []
                for i in range(self.n_batches):
                    batch_x = X_train[i*self.batch_size: (i+1)*self.batch_size,:]
                    batch_y = Y_train[i*self.batch_size: (i+1)*self.batch_size, :]
                    returns = sess.run([optimizer, self.cost, self.nlog_l]+
                            list(reg_tensors)+self.bn_updates, feed_dict={x: batch_x, y:
                                batch_y, self.y_std: self.Y_std,
                                self.epoch:epoch}
                            )
                    # Compute average loss
                    avg_cost += returns[1] / self.n_batches
                    avg_nlog_l += returns[2] / self.n_batches
                    for i, reg_name in enumerate(reg_names):
                        reg_vals[reg_name] += returns[3+i]

                # Check performance on validation set
                if (epoch % self.display_freq) == 0 or epoch == self.n_epochs-1:
                    # Log the state of the optimization
                    print "\nEpoch:", '%04d' % (epoch + 1), "cost=", \
                            "{:.9f}".format(avg_cost), "nlog_l=",\
                            "{:.9f}".format(avg_nlog_l)
                    print "avg cost:  %f"%avg_cost, "avg nlog_l: %f"%avg_nlog_l

                    # Log State for Training Set
                    max_eval_size=5000
                    feed_dict={x: X_train[:max_eval_size], y:
                            Y_train[:max_eval_size], self.y_std: self.Y_std,
                            self.epoch:np.int32(epoch),
                            }
                    # Get the value of the nlog_l in the training
                    # objective and evaluation.
                    if hasattr(self, 'nlog_l_eval'):
                        train_nlogl_obj, train_nlogl = sess.run(
                                [self.nlog_l, self.nlog_l_eval], feed_dict=feed_dict)
                        feed_dict[y] = Y_test[:max_eval_size]; feed_dict[x] = X_test[:max_eval_size]
                        test_nlogl_obj, test_nlogl= sess.run(
                                [self.nlog_l, self.nlog_l_eval],feed_dict=feed_dict)
                        print "Scaled n_log_l_obj --- Train: %f\tTest: %f"%(
                                train_nlogl_obj, test_nlogl_obj)
                        print "Scaled n_log_l_eval --- Train: %f\tTest: %f"%(
                                train_nlogl, test_nlogl)
                    else:
                        train_nlogl = sess.run(
                                [self.nlog_l], feed_dict=feed_dict)[0]
                        feed_dict[y] = Y_test[:max_eval_size]; feed_dict[x] = X_test[:max_eval_size]
                        test_nlogl = sess.run(
                                [self.nlog_l], feed_dict=feed_dict)[0]
                        print "Scaled n_log_l --- Train: %f\tTest: %f"%(
                                train_nlogl, test_nlogl)

                    # If performing VI, also log the weighted KL term, where
                    # the weight corresponds to the reverse annealing of
                    # this term.
                    if hasattr(self, "KL_weighted") and type(self.KL_weighted) != float:
                        KL_weighted_val = sess.run( [self.KL_weighted],
                                feed_dict=feed_dict)[0]
                        print "\tKL_weighted=", "{:.9f}".format(KL_weighted_val)

                    # Log Regularizers if any exists
                    if self.regularizers:
                        print "\tRegularization --- ", "\t".join("%s: %f"%(reg_name,
                            reg_val) for reg_name, reg_val in reg_vals.items())

                    # Find RMSE on testing and training sets if simply using
                    # a single stange linear flow.
                    if hasattr(self, 'flows') and len(self.flows)==1 and self.n_samples == 1:
                        rmse_train, rmse_test = self.RMSE(X_train, Y_train), self.RMSE(X_test, Y_test)
                        print "\tRMSE --- Train: %f\tTest: %f"%(rmse_train,
                                rmse_test)

                    ### Save Tensorboard logs if set to do so.
                    if not self.save_summaries:
                        continue

                    # First load an image of predictive distribution
                    if self.dataset == 'nyc' and self.log_image_summary:
                        # make a 2D heatmap if using a 2D dataset
                        title_base=self.summary_fn+"_Epoch_%04d"%epoch
                        ages = [15., 20., 40., 60.]
                        times = [1., 6., 15., 21.]
                        x_for_eval = []
                        for time in times:
                            x_for_eval += [(age, time) for age in ages]
                        pred_prob_buff_val = utils.plot_predictive_distribution_nyc(
                                self, n_grid_pts=100, return_buff=True,
                                title_base=title_base, x_for_eval=x_for_eval).getvalue()
                        feed_dict[self.pred_prob_buff] = pred_prob_buff_val
                    elif self.dataset == 'chicago' and self.log_image_summary:
                        title_base=self.summary_fn+"_Epoch_%04d"%epoch
                        x_for_eval = [{'hour':23}]
                        pred_prob_buff_val = utils.plot_predictive_distribution_chicago(
                                self, n_grid_pts=200,
                                title_base=title_base, max_color=100,
                                x_for_eval=x_for_eval, x_labels=['hour']).getvalue()
                        feed_dict[self.pred_prob_buff] = pred_prob_buff_val
                    elif self.dataset == 'nyc_taxi' and self.log_image_summary:
                        # make a 2D heatmap if using a 2D dataset
                        title_base=self.summary_fn+"_Epoch_%04d"%epoch
                        #fares = [3., 10., 52, 70.]
                        #ips = [0., 10., 23.]
                        fares = [10.]
                        tips = [10.]
                        x_for_eval = []
                        for fare in fares:
                            x_for_eval += [{"fare":fare, "tip_percent":tip,
                                'pickup_time':10.0, 'passenger_count':1.} for tip in tips]
                        pred_prob_buff_val = utils.plot_predictive_distribution_nyc(
                                self, n_grid_pts=200, return_buff=True,
                                title_base=title_base, taxi=True,
                                x_for_eval=x_for_eval, x_labels=['fare',
                                    'tip_percent']).getvalue()
                        feed_dict[self.pred_prob_buff] = pred_prob_buff_val
                    elif self.dataset == 'toy_small' and self.log_image_summary:
                        post_fcn_sample_buff = utils.plot_posterior_fcn_samples(self, epoch=epoch)
                        feed_dict[self.pred_prob_buff] = post_fcn_sample_buff.getvalue()
                        print "added buffer"
                    elif self.dataset == 'mog' and self.log_image_summary:
                        # make a 2D heatmap if using a 2D dataset
                        pred_prob_buff_val = utils.plot_predictive_distribution_mog(
                                self, n_grid_pts=100, return_buff=True).getvalue()
                        feed_dict[self.pred_prob_buff] = pred_prob_buff_val
                    elif self.dataset =='toy' and self.log_image_summary:
                        pred_prob_buff_val = utils.plot_predictive_distribution(
                                self, return_buff=True, plot_pts=self.plot_pts,
                                plot_title=self.plot_title,
                                plot_interval=self.plot_interval).getvalue()
                        feed_dict[self.pred_prob_buff] = pred_prob_buff_val
                    elif self.log_image_summary:
                        # otherwise, simply look at the label distribution
                        # in 1D

                        # First establish caption with nlogl, predictive std
                        # and rmse
                        if hasattr(self, 'flows') and len(self.flows)==1 and self.n_samples == 1:
                            additional_text = (" --- RMSE=%0.02f nlogl=%0.02f "
                                    "sigma_obs=%0.02f"%(
                                rmse_test, test_nlogl,
                                self.flows[0].m.eval()*self.Y_std))
                        else:
                            additional_text = ""
                        pred_prob_buff_val = utils.plot_label_predictive_distribution(
                                self, return_buff=True, add_txt=additional_text).getvalue()
                        feed_dict[self.pred_prob_buff] = pred_prob_buff_val

                    # Summaries for Train Set
                    feed_dict[y] = Y_train[:max_eval_size]; feed_dict[x] = X_train[:max_eval_size]
                    summary = sess.run([merged], feed_dict=feed_dict)[0]
                    train_writer.add_summary(summary, epoch)

                    # Summaries for Test Set
                    feed_dict[y] = Y_test[:max_eval_size]; feed_dict[x] = X_test[:max_eval_size]
                    summary = sess.run([merged],feed_dict=feed_dict)[0]
                    test_writer.add_summary(summary, epoch)
                    if self.save_model and (epoch % (self.display_freq*5)) == 0:
                        print "saving model checkpoint"
                        saver.save(sess, self.summary_path + "_epoch%05d_model.ckpt"%epoch)
                        print "finished saving"

            if self.save_model:
                saver.save(sess, self.summary_path + "_finalmodel.ckpt")
            # Return nlogl on train and test set, as well as RMSE if defined.
        try:
            return train_nlogl, test_nlogl, rmse_train, rmse_test
        except:
            return train_nlogl, test_nlogl

    def RMSE(self, X_eval, Y_eval):
        """RMSE calculates the root-mean-squared-eror on X_eval and Y_eval.
        If the model consists of only a single linear flow, this is done
        using the analytic closed form, otherwise, it is done by a binary
        search, implemented in utils.

        Args:
            X_eval: the features upon which to regress
            Y_eval: the corresponding target values

        Returns:
            The average RMSE

        """
        assert hasattr(self, 'flows') and len(self.flows)==1
        flow = self.flows[0]
        y_pred = flow.project(0.)
        # assert only one dimensional target and only one sample
        assert Y_eval.shape[-1] == 1
        assert y_pred.shape[0] == 1
        y_pred = y_pred[0, :, None]
        assert len(y_pred.shape) == len(Y_eval.shape)
        ses = (y_pred-Y_eval)**2
        rmse = tf.reduce_mean(ses)*(self.Y_std**2)

        return rmse.eval({self.x: X_eval, self.y:Y_eval})

    def test(self):
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.final_model_fn)
            X_test, Y_test = self.X_test, self.Y_test
            return self.nlog_l_of_mean_eval.eval({self.x: self.X_test, self.y:self.Y_test, self.y_std:
                self.Y_std})

