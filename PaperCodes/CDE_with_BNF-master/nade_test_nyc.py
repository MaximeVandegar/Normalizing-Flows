import sys
import pickle
import tensorflow as tf
import network
import nade_bayes
import numpy as np
from scipy import stats
import argparse

def main():
    ### Establish command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-log_fn", dest="log_fn", action="store",type=str,
            help="root filename for tb logs", required=True)
    parser.add_argument("-inference", dest="inference", action="store",type=str,
            help="must be MLE MAP or VI", required=True)
    parser.add_argument("-display_freq", dest="display_freq", action="store",type=int,
            default=1, help="display step frequency")
    parser.add_argument("-batch_size", dest="batch_size", action="store",type=int,
            default=10**10, help="batch size")
    parser.add_argument("-restore_epoch", dest="restore_epoch", action="store",type=int,
            default=None, help="restore epoch")
    parser.add_argument("-epochs", dest="epochs", action="store",type=int,
            default=7000, help="number of epochs")
    parser.add_argument("-lr", dest="lr", action="store", type=float,
            default=.01, help="learning rate")
    parser.add_argument("-w_prior_sigma", dest="w_prior_sigma", action="store", type=float,
            default=None, help="prior on weights")
    parser.add_argument("-lmbda_init", dest="lmbda_init", action="store", type=float,
            default=1.0, help=("initial level of interpolation between",
            "homoscedastic and heteroscedastic."))
    parser.add_argument("-n_samples", dest="n_samples", action="store",type=int,
            help="number of samples to take", default=1)
    parser.add_argument("-n_flows", dest="n_flows", action="store",type=str,
            help="number of stages in flow, comma separated", required=True)
    parser.add_argument("-log_base_dir", dest="log_base_dir", action="store",type=str,
            default='logs/', help="directory for tensorboard logs")
    parser.add_argument("-bayes_layers", dest="bayes_layers", action="store",type=str,
            default=None, help=("layers to learn approximate posterior in, comma"
            "separated"))
    parser.add_argument("-n_hidden", dest="n_hidden", action="store",type=str,
            default='50-50', help=("number of units in each hidden layer, comma"
            "separated within network, two networks separated by a hyphen"))
    parser.add_argument("-learn_sigma_weights",
            dest="learn_sigma_weights", action="store_true",
            default=False, help="to learn variances in VI")
    parser.add_argument("-init_sigma_params",
            dest="init_sigma_params", action="store", type=float,
            default=1e-5, help="initial variance of approximate posteriors")
    parser.add_argument("-init_sigma_obs", dest="init_sigma_obs",
            action="store", type=float, default=1.0, help="init_sigma_obs")


    ### scalings for flow variables
    parser.add_argument("-output_scaling", dest="output_scaling",
            action="store", type=float, default=1., help="output_scaling")
    parser.add_argument("-alpha_mu", dest="alpha_mu",
            action="store", type=float, default=1., help="alpha_mu")
    parser.add_argument("-alpha_std", dest="alpha_std",
            action="store", type=float, default=.5, help="alpha_std")
    parser.add_argument("-z_std", dest="z_std",
            action="store", type=float, default=1., help="z_std")
    parser.add_argument("-beta_std", dest="beta_std",
            action="store", type=float, default=1., help="beta_std")

    ### Parse args
    try:
        args = parser.parse_args()
        print(args)
        with open(args.log_base_dir +"/"+args.log_fn+"_args.pkl",'w') as f:
            pickle.dump(args, f)
        print "saved args"
    except IOError as e:
        parser.error(e)
    units1, units2 = args.n_hidden.split("-")
    n_hidden_units1 = [int(n) for n in units1.split(",")]
    n_hidden_units2 = [int(n) for n in units2.split(",")]
    n_hidden_units = [n_hidden_units1, n_hidden_units2]

    n_flows1, n_flows2 = args.n_flows.split("-")
    n_flows = [int(n_flows1), int(n_flows2)]


    if args.bayes_layers is None:
        # This is learning approximate posterior for every layer.
        bayes_layers = None
    elif len(args.bayes_layers) == 0:
            bayes_layers = []
    else:
        bayes_layers = [int(l) for l in args.bayes_layers.split(",")]

    net = nade_bayes.nade_bayes(
            summary_fn=args.log_fn,
            init_sigma_params=args.init_sigma_params,
            n_hidden_units=n_hidden_units,
            n_epochs=args.epochs,
            n_flows=n_flows,
            lr=args.lr,
            dataset='nyc_taxi',
            inference=args.inference,
            log_base_dir=args.log_base_dir,
            display_freq=args.display_freq,
            n_samples=args.n_samples,
            output_scaling=args.output_scaling,
            init_sigma_obs=args.init_sigma_obs,
            bayes_layers=bayes_layers,
            batch_size=args.batch_size,
            lmbda=args.lmbda_init,
            save_model=True,
            w_prior_sigma=args.w_prior_sigma,
            learn_sigma_weights=args.learn_sigma_weights,
            restore_epoch=args.restore_epoch,

            alpha_mu=args.alpha_mu,
            alpha_std=args.alpha_std,
            beta_std=args.beta_std,
            z_std=args.z_std,
            )

    print "Beginning Training"
    net.split(0)
    net.summary_path = net.log_base_dir+args.log_fn
    print net.train()

if __name__ == "__main__":
    main()
