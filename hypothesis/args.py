import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.02,
        help='entropy term coefficient (default: 0.02)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU processes to use')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=4,
        help='number of batches for ppo (default: 4)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10000,
        help='number of epochs/updates to train (default: 1e5)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    ####################################
    # Arguments pertaining to MultiKeyDoorEnv
    ####################################
    parser.add_argument(
        '--env-name',
        default='MultiDoorKeyEnv-8x8-2keys-v0',
        help='environment to train on (default: MultiDoorKeyEnv-8x8-2keys-v0)')
    parser.add_argument(
        '--fixed_positions', 
        default=False,
        action='store_true',
        help='Whether to keep key and door positions fixed across resets')

    ####################################
    # Arguments pertaining to ConSpec
    ####################################
    parser.add_argument(
        '--num_prototypes',
        type=int,
        default=8,
        help='number of prototypes used in Conspec')
    parser.add_argument(
        '--intrinsicR_scale',
        type=float,
        default=0.05,
        help='lambda for intrinsic reward from ConSpec')
    parser.add_argument(
        '--save_checkpoint',
        action='store_true',
        default=False,
        help='save checkpoint after training')
    parser.add_argument(
        '--SF_buffer_size',
        type=int,
        default=16,
        help='size of the success and failure memory buffers')
    parser.add_argument(
        '--cos_score_threshold',
        type=float,
        default=0.99,
        help='cosine similarity threshold for ConSpec')
    parser.add_argument(
        '--roundhalf',
        type=int,
        default=3,
        help='window size for rolling average')
    
    ####################################
    # Arguments pertaining to loading pretrained Conspec
    ####################################
    parser.add_argument(
        '--base_path',
        type=str,
        default='/network/scratch/l/lindongy/grid_blickets/conspec_ckpt/MultiDoorKeyEnv-6x6-2keys-v0-conspec-rec-PO-lr0.0006-intrinsR0.1-lrConSpec0.01-entropy0.02-num_mini_batch4-seed1',
        help='path to the pretrained Conspec model')
    parser.add_argument(
        '--ckpt_epi',
        type=int,
        default=3399,
        help='episode number of the pretrained Conspec model')
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
