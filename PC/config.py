#-*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_dimension', type=int, default=2, help='')
net_arg.add_argument('--n_components', type=int, default=5, help='') # Number of components for PCA
net_arg.add_argument('--input_embed', type=int, default=32, help='') # Actor
net_arg.add_argument('--input_embed_c', type=int, default=16, help='') # Critic
net_arg.add_argument('--hidden_dim', type=int, default=128, help='') #256
net_arg.add_argument('--init_min_val', type=float, default=-0.08, help='for uniform random initializer')
net_arg.add_argument('--init_max_val', type=float, default=+0.08, help='for uniform random initializer')
net_arg.add_argument('--init_bias_c', type=float, default=0.7, help='for critic')
net_arg.add_argument('--num_layers', type=int, default=1, help='for actor and critic')  #2 is really good !
net_arg.add_argument('--filter_sizes', type=list, default=[1,2,3,4,5], help='')
net_arg.add_argument('--num_filters', type=int, default=1, help='')
#net_arg.add_argument('--max_enc_length', type=int, default=None, help='')
#net_arg.add_argument('--max_dec_length', type=int, default=None, help='')
#net_arg.add_argument('--num_glimpse', type=int, default=1, help='')
#net_arg.add_argument('--use_terminal_symbol', type=str2bool, default=True, help='')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=32) # 128 ##################################################### (seed 123)
data_arg.add_argument('--max_length', type=int, default=10)  # input sequence length (number of cities)
data_arg.add_argument('--scale', type=float, default=1.0)
#data_arg.add_argument('--task', type=str, default='tsp')
#data_arg.add_argument('--min_data_length', type=int, default=5)
#data_arg.add_argument('--train_num', type=int, default=1000)
#data_arg.add_argument('--valid_num', type=int, default=1000)
#data_arg.add_argument('--test_num', type=int, default=1000)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=60, help='')
train_arg.add_argument('--actor_passes', type=int, default=1, help='')
train_arg.add_argument('--critic_passes', type=int, default=1, help='')
train_arg.add_argument('--init_temperature', type=float, default=1.0 , help='') ##################################################### in [0.3, 3000]
train_arg.add_argument('--temperature_decay', type=float, default=0.9, help='') #####################################################
train_arg.add_argument('--lr1_start', type=float, default=0.0001, help='') # 0.001 #####################################################
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.1, help='') # 0.96 #####################################################
#train_arg.add_argument('--lr1_decay_step', type=int, default=100, help='') # 5000 
train_arg.add_argument('--lr2_start', type=float, default=0.0002, help='') #####################################################
train_arg.add_argument('--lr2_decay_rate', type=float, default=0.1, help='') #####################################################
train_arg.add_argument('--lr2_decay_step', type=int, default=100, help='')
#train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
#train_arg.add_argument('--optimizer', type=str, default='rmsprop', help='')
#train_arg.add_argument('--max_step', type=int, default=1000000, help='')
#train_arg.add_argument('--max_grad_norm', type=float, default=2.0, help='')
#train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')

# Misc
misc_arg = add_argument_group('Misc')
#misc_arg.add_argument('--log_step', type=int, default=50, help='')
#misc_arg.add_argument('--num_log_samples', type=int, default=3, help='')
#misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'], help='')
#misc_arg.add_argument('--log_dir', type=str, default='logs')
#misc_arg.add_argument('--data_dir', type=str, default='data')
#misc_arg.add_argument('--output_dir', type=str, default='outputs')
#misc_arg.add_argument('--load_path', type=str, default='')
#misc_arg.add_argument('--debug', type=str2bool, default=False)
#misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
#misc_arg.add_argument('--random_seed', type=int, default=123, help='')

# parser.add_argument("-v", "--verbosity", help="increase output verbosity",action="count", default=0)


def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


def print_config():
  config, _ = get_config()
  print('\n')
  print('Config:')
  print('* Batch size:',config.batch_size)
  print('* Sequence length:',config.max_length)
  print('* City coordinates:',config.input_dimension)
  print('* City dimension:',config.input_dimension+2*(config.input_dimension+1)+1)
  print('* Input embedding:',config.input_embed)
  print('* Num neurons (Actor & critic):',config.hidden_dim)
  print('\n')



"""
sigma_a=20
sigma_b=24
deltax=-np.log(sigma_a/sigma_b)*sigma_a*sigma_b/(sigma_a+sigma_b)
print(deltax)
"""