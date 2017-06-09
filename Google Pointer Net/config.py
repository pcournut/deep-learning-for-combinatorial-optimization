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

net_arg.add_argument('--seeded_net', type=str2bool, default=False, help='whether or not graph is seeded')
net_arg.add_argument('--seed_net', type=int, default=123, help='graph seed value')

net_arg.add_argument('--input_embed', type=int, default=128, help='actor input embedding') #####################################################
net_arg.add_argument('--hidden_dim', type=int, default=128, help='actor LSTM num_neurons') #####################################################

net_arg.add_argument('--init_min_val', type=float, default=-0.08, help='for uniform random initializer') #####################################################
net_arg.add_argument('--init_max_val', type=float, default=+0.08, help='for uniform random initializer') #####################################################

#net_arg.add_argument('--max_enc_length', type=int, default=None, help='')
#net_arg.add_argument('--max_dec_length', type=int, default=None, help='')
#net_arg.add_argument('--num_glimpse', type=int, default=1, help='')
#net_arg.add_argument('--use_terminal_symbol', type=str2bool, default=True, help='')


# Data
data_arg = add_argument_group('Data')

data_arg.add_argument('--seed_data', type=int, default=0, help='data generator seed value') # Default 0 means no seed

data_arg.add_argument('--batch_size', type=int, default=128, help='batch size')
data_arg.add_argument('--input_dimension', type=int, default=2, help='city dimension')
data_arg.add_argument('--max_length', type=int, default=20, help='input sequence length')

#data_arg.add_argument('--task', type=str, default='tsp')
#data_arg.add_argument('--min_data_length', type=int, default=5)
#data_arg.add_argument('--train_num', type=int, default=1000)
#data_arg.add_argument('--valid_num', type=int, default=1000)
#data_arg.add_argument('--test_num', type=int, default=1000)


# Training / test parameters
train_arg = add_argument_group('Training')

train_arg.add_argument('--nb_epoch', type=int, default=10000, help='nb epoch')

train_arg.add_argument('--lr1_start', type=float, default=0.001, help='actor learning rate') 		# 0.001 during training
train_arg.add_argument('--lr1_decay_step', type=int, default=5000, help='lr1 decay step') # 5000
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate') # 0.96

train_arg.add_argument('--alpha', type=float, default=0.99, help='update factor moving average baseline')

train_arg.add_argument('--inference_mode', type=str2bool, default=False, help='switch to inference mode when model is trained')

train_arg.add_argument('--init_temperature', type=float, default=1.0 , help='pointer_net initial temperature') ##################################################### in [0.3, 3000]
train_arg.add_argument('--T_decay_step', type=int, default=1000, help='temperature decay step')
train_arg.add_argument('--T_decay_rate', type=float, default=1.0, help='temperature decay rate')

train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping') #####################################################

#train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
#train_arg.add_argument('--optimizer', type=str, default='rmsprop', help='')
#train_arg.add_argument('--max_step', type=int, default=1000000, help='')
#train_arg.add_argument('--max_grad_norm', type=float, default=2.0, help='')
#train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')


# Misc
misc_arg = add_argument_group('Misc')

misc_arg.add_argument('--save_to', type=str, default='20/1', help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='20/1', help='loader sub directory')
misc_arg.add_argument('--restore_model', type=str2bool, default=False, help='whether or not model is retrieved')
misc_arg.add_argument('--log_dir', type=str, default='summary/10.t', help='summary writer log directory') 

#misc_arg.add_argument('--log_step', type=int, default=50, help='')
#misc_arg.add_argument('--num_log_samples', type=int, default=3, help='')
#misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'], help='')
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
  print('Data Config:')
  if config.seed_data!=0:
  	print('* Seeded data: True ( seed',config.seed_data,')')
  else:
  	print('* Seeded data: False')
  print('* Batch size:',config.batch_size)
  print('* Sequence length:',config.max_length)
  print('* City coordinates:',config.input_dimension)
  print('\n')
  print('Network Config:')
  print('* Restored model:',config.restore_model)
  if config.restore_model==False:
  	if config.seeded_net==True:
  		print('* Seeded network: True ( seed',config.seed_net,')')
  	else:
  		print('* Seeded network: False')
  print('* Actor input embedding:',config.input_embed)
  print('* Actor hidden_dim (num neurons):',config.hidden_dim)
  print('* Actor tan clipping:',config.C)
  if config.restore_model==False:
  	print('* Uniform random initializer:',config.init_min_val,':',config.init_max_val)
  print('\n')
  if config.inference_mode==False:
  	print('Training Config:')
  	print('* Nb epoch:',config.nb_epoch)
  	print('* Temperature (init,decay_step,decay_rate):',config.init_temperature,config.T_decay_step,config.T_decay_rate)
  	print('* Actor learning rate (init,decay_step,decay_rate):',config.lr1_start,config.lr1_decay_step,config.lr1_decay_rate)
  else:
  	print('Testing Config:')
  print('* Summary writer log dir:',config.log_dir)
  print('\n')