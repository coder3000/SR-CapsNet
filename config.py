import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='CapsNet')

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum value')
train_arg.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay value')
train_arg.add_argument('--epochs', type=int, default=350,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.1,
                       help='Initial learning rate value')
train_arg.add_argument('--train_patience', type=int, default=100,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset for training: {mnist, cifar10}')
train_arg.add_argument('--planes', type=int, default=16,
                       help='starting layer width')
train_arg.add_argument('--num_caps', type=int, default=32,
                       help="# of capsules per layer")
train_arg.add_argument('--caps_size', type=int, default=16,
                       help="# of neurons per capsule")
train_arg.add_argument('--depth', type=int, default=1,
                       help="depth of additional layers")


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--name', type=str, default=None,
                      help='Name of model to load / save')
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=2018,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt_cifar10',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')

misc_arg.add_argument('--attack', type=str2bool, default=False,
                      help='Whether to test against attack')
misc_arg.add_argument('--attack_type', type=str, default='fgsm',
                      help='Attack to perform: {fgms, bim}')
misc_arg.add_argument('--attack_eps', type=float, default=0.1,
                      help='eps for adv attack')
misc_arg.add_argument('--targeted', type=str2bool, default=False,
                      help='if true, do targeted attack')
train_arg.add_argument('--exp', type=str, default='',
                       help="viewpoint exp name (NULL, azimuth, elevation, full)")
train_arg.add_argument('--familiar', type=str2bool, default=True,
                       help="viewpoint exp setting (novel, familiar)")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
