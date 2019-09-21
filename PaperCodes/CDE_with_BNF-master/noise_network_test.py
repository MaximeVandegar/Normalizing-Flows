import network, input_noise_network
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
    parser.add_argument("-n_samples", dest="n_samples", action="store",type=int,
            help="number of noise samples", required=True)
    parser.add_argument("-noise_dim", dest="noise_dim", action="store",type=int,
            help="dimensionality on input noise",default=1)
    parser.add_argument("-display_freq", dest="display_freq", action="store",type=int,
            help="log state every this many epochs",default=100)
    parser.add_argument("-r_mag_W", dest="r_mag_W", action="store",
            type=float,default=0., help="magnitude of L2 regularizer on W")
    ### Parse args
    try:
        args = parser.parse_args()
        print(args)
    except IOError as e:
        parser.error(e)

    net = input_noise_network.input_noise_network(
            summary_fn=args.log_fn,
            n_hidden_units=[50],
            n_epochs=args.epochs,
            n_samples=args.n_samples,
            noise_dim=args.noise_dim,
            lr=args.lr,
            dataset="toy",
            n_pts=5000,
            log_base_dir=args.log_base_dir,
            r_mag_W=args.r_mag_W,
            display_freq=args.display_freq,
            batch_size=args.batch_size,
            )
    net.split(0)
    net.train()

if __name__ == "__main__":
    main()
