import network, flow_network
import argparse

def main():
    ### Establish command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", dest="lr", action="store",
            help="The learning rate", type=float, default=0.05)
    parser.add_argument("-epochs", dest="epochs", action="store",
            help="Number of epochs to run", type=int, default=1000)
    parser.add_argument("-batch_size", dest="batch_size", action="store",
            help="Number of epochs to run", type=int, default=10**10)
    parser.add_argument("-log_base_dir", dest="log_base_dir", action="store",type=str,
            default='logs/', help="directory for tensorboard logs")
    parser.add_argument("-log_fn", dest="log_fn", action="store",type=str,
            help="root filename for tb logs", required=True)
    parser.add_argument("-n_flows", dest="n_flows", action="store",type=int,
            help="number of stages in flow", required=True)
    parser.add_argument("-display_freq", dest="display_freq", action="store",type=int,
            help="log state every this many epochs",default=100)
    parser.add_argument("-het_sced", dest="het_sced", action="store_true",
            default=False, help="if variance is to be predicted")
    parser.add_argument("-input_independent", dest="input_independent", action="store_false",
            default=False, help="if parameters of the flow are to be input independent")
    parser.add_argument("-r_mag_alpha", dest="r_mag_alpha", action="store",
            type=float,default=0., help="magnitude of regularizer on alphas")
    parser.add_argument("-r_mag_beta", dest="r_mag_beta", action="store",
            type=float,default=0., help="magnitude of regularizer on betas")
    parser.add_argument("-r_mag_z_0", dest="r_mag_z_0", action="store",
            type=float,default=0., help="magnitude of regularizer on z")
    parser.add_argument("-r_mag_W", dest="r_mag_W", action="store",
            type=float,default=0., help="magnitude of L2 regularizer on W")
    ### Parse args
    try:
        args = parser.parse_args()
        print(args)
    except IOError as e:
        parser.error(e)

    net = flow_network.flow_network(
            summary_fn=args.log_fn,
            n_hidden_units=[50],
            n_epochs=args.epochs,
            predict_var=args.het_sced,
            n_flows=args.n_flows,
            input_dependent=not args.input_independent,
            lr=args.lr,
            dataset="toy",
            n_pts=5000,
            log_base_dir=args.log_base_dir,
            r_mag_W=args.r_mag_W,
            r_mag_alpha=args.r_mag_alpha,
            r_mag_beta=args.r_mag_beta,
            r_mag_z_0=args.r_mag_z_0,
            display_freq=args.display_freq,
            batch_size=args.batch_size,
            )
    net.split(0)
    net.train()

if __name__ == "__main__":
    main()
