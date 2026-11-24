import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--DS', type=str, default="NCI1", help='Name of dataset for trainig.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--feat', type=str, default='label', choices=['deg', 'default', 'label'],
                        help='Node features: "deg" (degree), "default" (node attribute file; for BZR/COX2/DHFR these are 3D coordinates), '
                             '"label" (one-hot node labels, e.g. atom types).')
    
    parser.add_argument('--wgan_epochs', type=int, default=10000, help='Number of epochs for WGAN training.')
    parser.add_argument('--epochs_decay', type=int, default=5000, help='Number of epochs over which to decay the learning rate.')
    parser.add_argument('--lr_update_step', type=int, default=1000, help='Step size for updating learning rates.')
    parser.add_argument('--patience', type=int, default=200, help='Number of epochs to wait for improvement before stopping.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training from a checkpoint.')
    
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate for the generator.')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Learning rate for the discriminator.')
    
    parser.add_argument('--e_lr', type=float, default=2e-4, help='Learning rate for the encoder.')
    parser.add_argument('--encoder_epochs', type=int, default=1000, help='Number of epochs for the encoder.')

    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--gumbell_type', type=str, default='sigmoid',
                        choices=['hard', 'soft', 'sigmoid', 'threshold', 'hard-gumbell', 'no-gumbell'],
                        help='How generated edges are produced: "hard" (binary Gumbel-sigmoid, straight-through), '
                             '"soft" (relaxed Gumbel-sigmoid), "sigmoid" (edge probabilities), "threshold" (binary, no noise). '
                             'Legacy: "hard-gumbell" = hard, "no-gumbell" = sigmoid.')
    parser.add_argument('--normal_label', type=int, default=None,
                        help='Raw graph label treated as the normal class. Defaults to the most frequent label in the dataset.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Compute device. "auto" picks CUDA, then Apple Metal (mps), then CPU.')
    parser.add_argument('--encoder_mode', type=str, default='izi_f+ziz', choices=['izi_f', 'izi_feat', 'ziz', 'izi_f+ziz'],
                        help='Encoder training objective (see train/train_encoder.py).')
    parser.add_argument('--score', type=str, default='recon+critic|size+sizedev', choices=['recon+critic|size+sizedev', 'izi_f', 'feat', 'critic', 'critic+feat', 'critic+feat+z', 'recon', 'recon+critic', 'recon+critic+z',
                                 'feat|size', 'critic|size', 'critic+feat|size', 'critic+feat+z|size', 'recon|size', 'recon+critic|size', 'recon+critic+z|size'],
                        help='Anomaly score used for the reported AUC (all components are logged to bin/aucs.json). '
                             'Default: size-adjusted recon+critic plus a label-free size-deviation term whose weight is chosen on the validation split.')
    parser.add_argument('--no_intermediate_eval', action='store_true',
                        help='Skip encoder training + evaluation at WGAN checkpoints.')
    parser.add_argument('--wgan_steps', type=int, default=10000,
                        help='Train the WGAN for this many critic steps (overrides --wgan_epochs, --epochs_decay and --lr_update_step; 0 = use epochs).')
    parser.add_argument('--encoder_steps', type=int, default=1500,
                        help='Train the encoder for this many steps (overrides --encoder_epochs; 0 = use epochs).')
    parser.add_argument('--info_weight', type=float, default=1.0,
                        help='Weight of the latent-regression loss ||E(G(z)) - z||^2 trained jointly with G (0 = off).')
    parser.add_argument('--canonical', action=argparse.BooleanOptionalAction, default=True,
                        help='Put every graph in a canonical node order (colour refinement + BFS), so adjacency matrices have a fixed layout.')
    parser.add_argument('--order_aware', action=argparse.BooleanOptionalAction, default=True,
                        help='Add an order-aware MLP branch to the critic and encoder (use with --canonical).')
    parser.add_argument('--drift', type=float, default=1e-3, help='Critic drift penalty weight (epsilon * D(x)^2).')
    parser.add_argument('--size_cond', action=argparse.BooleanOptionalAction, default=True,
                        help='Condition the generator on the node count n and mask slots past n (use with --canonical).')
    parser.add_argument('--val_frac', type=float, default=0.2,
                        help='Fraction of the training data (normals and anomalies) held out as a labelled validation set.')
    parser.add_argument('--cv_fold', type=int, default=None, choices=range(5),
                        help='Use fold k of a stratified 5-fold split as the test set (non-Tox21 datasets).')
    parser.add_argument('--cv_seed', type=int, default=42, help='Random state of the 5-fold partition (data split only).')
    parser.add_argument('--z_dim', type=int, default=32, help='Latent dimension.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--log_graphs', action='store_true', help='Log sampled generated graphs to bin/graphs.parquet.')
    parser.add_argument('--n_critic', type=int, default=5, help='Number of discriminator updates per generator update.')

    parser.add_argument('--wgan_pretrained', action='store_true', help='Use pretrained WGAN model.')
    parser.add_argument('--encoder_pretrained', action='store_true', help='Use pretrained encoder model.')

    parser.add_argument('--early_stop', action='store_true', help='Early stop WGAN training.')
    parser.add_argument('--plot_loss', action='store_true', help='Plot discriminator and generator losses over epochs.')
    parser.add_argument('--quiet', action='store_true', help='Do not print to console.')


    return parser.parse_args()
