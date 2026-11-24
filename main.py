import glob
import warnings
import re
import os
import sys
import random
from collections import Counter
from train.train_encoder import train_encoder
from train.train_wgan import train_wgan
from torch.utils.data import DataLoader
from utils.operations import *
from utils.plots import *
from utils.graph_processing import *
from data.graph_sampler import GraphSampler
from data.load_data import read_graphfile
from data.canonical import reorder
from sklearn.model_selection import train_test_split, StratifiedKFold
from models.gan import Generator, Discriminator
from models.encoder import Encoder
from utils.parser import parse_arguments
from train.anomaly_detection import detect_anomalies, select_sizedev_weight

def prepare_data(datadir, args):

    if args.DS.startswith("Tox21"):
        graphs_train = read_graphfile(datadir, args.DS + '_training', args.quiet, max_nodes=0)
        graphs_test = read_graphfile(datadir, args.DS + '_testing', args.quiet, max_nodes=0)
        graphs = graphs_train + graphs_test
    else:
        graphs = read_graphfile(datadir, args.DS, args.quiet, max_nodes=0)
        if args.cv_fold is None:
            graphs_train, graphs_test = train_test_split(graphs, test_size=0.2, random_state=42)
        else:
            # 5-fold stratified CV (the protocol GLocalKD uses for non-Tox21 datasets); fold k is the test set
            folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.cv_seed).split(graphs, [g.graph['label'] for g in graphs]))
            tr_idx, te_idx = folds[args.cv_fold]
            graphs_train, graphs_test = [graphs[i] for i in tr_idx], [graphs[i] for i in te_idx]

    if args.canonical:
        # only use node labels for ordering when the model sees them (keeps 'plain graph' runs label-free)
        reordered = {id(g): reorder(g, use_labels=(args.feat == 'label')) for g in graphs}
        graphs = [reordered[id(g)] for g in graphs]
        graphs_train = [reordered[id(g)] for g in graphs_train]
        graphs_test = [reordered[id(g)] for g in graphs_test]

    # map raw labels to 0 = normal, 1 = anomalous
    label_counts = Counter(int(g.graph['label']) for g in graphs)
    normal_label = args.normal_label if args.normal_label is not None else label_counts.most_common(1)[0][0]
    if normal_label not in label_counts:
        raise ValueError(f"normal_label {normal_label} not in dataset labels {sorted(label_counts)}")
    for g in graphs:
        g.graph['label'] = 0 if int(g.graph['label']) == normal_label else 1

    # labelled validation set carved from the training data (normals + the anomalies that training drops);
    # used for model selection so the test set is only read for reporting
    graphs_val = []
    if args.val_frac > 0:
        graphs_train, graphs_val = train_test_split(graphs_train, test_size=args.val_frac, random_state=0,
                                                    stratify=[g.graph['label'] for g in graphs_train])

    # filtering out abnormal graphs for training
    graphs_train = [graph for graph in graphs_train if graph.graph['label'] == 0]
    n_test_anom = sum(g.graph['label'] for g in graphs_test)
    if not args.quiet:
        print(f"{args.DS}: raw label counts {dict(label_counts)}, normal label = {normal_label}")
        print(f"train: {len(graphs_train)} normal graphs | val: {len(graphs_val)} graphs ({sum(g.graph['label'] for g in graphs_val)} anomalous) "
              f"| test: {len(graphs_test)} graphs ({n_test_anom} anomalous)")

    max_num_nodes = max([G.number_of_nodes() for G in graphs])
    label_dim = None
    if args.feat == 'label':
        if 'label' not in node_dict(graphs[0])[0]:
            raise ValueError(f"{args.DS} has no node labels; use --feat deg or default")
        label_dim = max(len(node_dict(g)[u]['label']) for g in graphs for u in g.nodes())
    dataset_sampler_train = GraphSampler(graphs_train, features=args.feat, normalize=False, max_num_nodes=max_num_nodes, label_dim=label_dim)
    dataset_sampler_test = GraphSampler(graphs_test, features=args.feat, normalize=False, max_num_nodes=max_num_nodes, label_dim=label_dim)

    # drop_last avoids a size-1 batch, which BatchNorm in the generator cannot handle
    data_loader_train = DataLoader(dataset_sampler_train, shuffle=True, batch_size=args.batch_size, drop_last=True)
    data_loader_test = DataLoader(dataset_sampler_test, shuffle=False, batch_size=args.batch_size)
    data_loader_val = None
    if graphs_val:
        dataset_sampler_val = GraphSampler(graphs_val, features=args.feat, normalize=False, max_num_nodes=max_num_nodes, label_dim=label_dim)
        data_loader_val = DataLoader(dataset_sampler_val, shuffle=False, batch_size=args.batch_size)

    return max_num_nodes, dataset_sampler_train, data_loader_train, data_loader_test, data_loader_val

def build_models(args, max_num_nodes, feat_dim, z_dim, device,
                 g_conv_dim=(128, 256, 512), d_linear_dim=(128, 64), d_aggr_dim=128, e_linear_dim=(128, 64)):
    n_flat = max_num_nodes if args.order_aware else None
    G = Generator(list(g_conv_dim), z_dim, max_num_nodes, feat_dim, args.dropout, size_cond=args.size_cond).to(device)
    D = Discriminator([feat_dim, [128, 64]], d_aggr_dim, list(d_linear_dim), device=device, dropout=args.dropout,
                      num_nodes=n_flat).to(device)
    E = Encoder(feat_dim, list(e_linear_dim), z_dim, args.dropout, device=device, num_nodes=n_flat).to(device)
    return G, D, E


def find_latest_checkpoint(checkpoint_dir, prefix):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f'{prefix}_checkpoint_*.pth'))
    
    if not checkpoint_files:
        return None
    
    def extract_epoch(filename):
        match = re.search(r'_checkpoint_(\d+)\.pth', filename)
        return int(match.group(1)) if match else -1
    
    # find file with highest epoch
    latest_file = max(checkpoint_files, key=extract_epoch)
    return latest_file

def load_latest_checkpoints(G, D, g_optimizer, d_optimizer, checkpoint_dir):
    # load generator checkpoint
    g_checkpoint_path = os.path.join(checkpoint_dir, 'G_checkpoint_final.pth')
    if not os.path.exists(g_checkpoint_path):
        g_checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'G')
        if g_checkpoint_path:
            print(f"Loading latest G checkpoint: {g_checkpoint_path}")
        else:
            raise FileNotFoundError("No G checkpoint found!")
    else:
        print(f"Loading final G checkpoint: {g_checkpoint_path}")

    _, _ = load_checkpoint(G, g_optimizer, g_checkpoint_path)

    # load discriminator checkpoint
    d_checkpoint_path = os.path.join(checkpoint_dir, 'D_checkpoint_final.pth')
    if not os.path.exists(d_checkpoint_path):
        d_checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'D')
        if d_checkpoint_path:
            print(f"Loading latest D checkpoint: {d_checkpoint_path}")
        else:
            raise FileNotFoundError("No D checkpoint found!")
    else:
        print(f"Loading final D checkpoint: {d_checkpoint_path}")

    _, _ = load_checkpoint(D, d_optimizer, d_checkpoint_path)

if __name__ == "__main__":

    warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS, but there was no current CUDA context.*")

    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

    DATADIR = os.path.join(repo_dir, 'datasets/')
    CHECKPOINT_DIR = 'checkpoints/'
    PLOTS_DIR = 'plots/'
    LOGS_DIR = 'bin/'
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)
    if not args.quiet:
        print(f"device: {DEVICE}")

    max_num_nodes, dataset_sampler_train, data_loader_train, data_loader_test, data_loader_val = prepare_data(DATADIR, args)

    # step-based budgets make runs comparable across datasets of very different sizes
    if args.wgan_steps:
        args.wgan_epochs = max(1, -(-args.wgan_steps // len(data_loader_train)))
        args.epochs_decay = args.wgan_epochs // 2
        args.lr_update_step = max(1, args.wgan_epochs // 10)
    if args.encoder_steps:
        args.encoder_epochs = max(1, -(-args.encoder_steps // len(data_loader_train)))
    if not args.quiet:
        print(f"{len(data_loader_train)} batches/epoch | WGAN epochs: {args.wgan_epochs} | encoder epochs: {args.encoder_epochs}")

    # pre-defined parameters
    G_CONV_DIM = [128,256,512]
    D_CONV_DIM = [dataset_sampler_train.feat_dim, [128, 64]]
    D_AGGR_DIM = 128
    D_LINEAR_DIM = [128, 64]
    Z_DIM = args.z_dim
    BETA1 = 0.5
    BETA2 = 0.99
    LAMBDA_GP = 10  # penalty coefficient
    ENCODER_SAVE_ITERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    WGAN_SAVE_ITERS = [300, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 100000, 150000]
    E_LINEAR_DIM = [128, 64]
    KAPPA = 1.0

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    
    if not os.path.exists(PLOTS_DIR) and args.plot_loss:
        os.mkdir(PLOTS_DIR)

    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    G, D, E = build_models(args, max_num_nodes, dataset_sampler_train.feat_dim, Z_DIM, DEVICE,
                           G_CONV_DIM, D_LINEAR_DIM, D_AGGR_DIM, E_LINEAR_DIM)
    g_optimizer = torch.optim.Adam(G.parameters(), args.g_lr, [BETA1, BETA2])
    d_optimizer = torch.optim.Adam(D.parameters(), args.d_lr, [BETA1, BETA2])
    e_optimizer =torch.optim.Adam(E.parameters(), args.e_lr, [BETA1, BETA2])

    G.apply(weights_init)
    D.apply(weights_init)
    E.apply(weights_init)
    
    if args.resume_training:
        g_path, d_path = find_latest_checkpoint(CHECKPOINT_DIR, 'G'), find_latest_checkpoint(CHECKPOINT_DIR, 'D')
        if g_path is None or d_path is None:
            raise FileNotFoundError("No numbered G/D checkpoints to resume from")
        start_epoch, _ = load_checkpoint(G, g_optimizer, g_path)
        load_checkpoint(D, d_optimizer, d_path)
    else:
        start_epoch = 0

    losses = None

    if not args.wgan_pretrained:
        losses = train_wgan(G, D, E, g_optimizer, d_optimizer, e_optimizer, start_epoch, args, data_loader_train, 
                   data_loader_test, DEVICE, Z_DIM, WGAN_SAVE_ITERS, CHECKPOINT_DIR, LAMBDA_GP, KAPPA)

    load_latest_checkpoints(G, D, g_optimizer, d_optimizer, CHECKPOINT_DIR)

    joint_e_path = os.path.join(CHECKPOINT_DIR, 'E_checkpoint_wgan.pth')
    if args.info_weight > 0 and os.path.exists(joint_e_path):
        load_checkpoint(E, e_optimizer, joint_e_path)

    if not args.encoder_pretrained:
        train_encoder(G, D, E, e_optimizer, args.encoder_epochs, data_loader_train, DEVICE, KAPPA, CHECKPOINT_DIR, args, Z_DIM)

    _, _ = load_checkpoint(E, e_optimizer, os.path.join(CHECKPOINT_DIR, 'E_checkpoint_final.pth'))

    args.sizedev_w = 0.
    if data_loader_val is not None:
        val_aucs = detect_anomalies(G, E, D, data_loader_val, KAPPA, DEVICE, args, train_loader=data_loader_train, split='val')
        if args.score == 'recon+critic|size+sizedev':
            args.sizedev_w = select_sizedev_weight(val_aucs)   # chosen on validation labels only
    elif args.score == 'recon+critic|size+sizedev':
        print("warning: no validation split (--val_frac 0); size-deviation weight fixed at 0")
    detect_anomalies(G, E, D, data_loader_test, KAPPA, DEVICE, args, train_loader=data_loader_train)

    scores = load_scores_from_file('scores.pkl')

    with open("bin/output.txt", "w") as f:

        f.write("Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write(f"\nFinal score: {scores.get('final')}\n")

    if args.plot_loss and losses is not None:
        plot_losses(losses, PLOTS_DIR)

