#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

#from tsp_with_ortools import Solver
from dataset import DataGenerator
from actor import Actor
from tqdm import tqdm

from config import get_config, print_config

# TODO

"""
- 20 cities TSP (more training)

- Paper: Learning Combinatorial Optimization on graph ***

- CalculQuebec Helios (ssh MobaXterm + WinSCP) - Script.pbs (OR tools ??) ***

- DO EXACTLY AS IN GOOGLE PAPER (pdf) - Git
    
    - Inference mode - GOOGLE ACTIVE SEARCH proc ! ### ###

    - Attention improvements (C.tanh(.) vs ./T)

    - (Actor-) Critic (Encoder + 3 process steps + 2 FFN) for RL pretrain

_________________________CONFIG_____________________________________________

- Parameter tuning [config.py] *** ***

- SYNTHESE OVERLEAF !!! / Git *** ***
_____________________________________________________________________
_____________________________________________________________________
_____________________________________________________________________

- Variable seq length (padding)

------ Speed up learning on GPU !!!!!!!!
- Parallelize (GPU) for parameter search, C++... (Thread Coordinator / Dataset QueueRunner + Supervisor)

--- Mash up Alex Notebook (start from noise and Active Search with attention / conditional probability / Translation conv. attention / Gumbel Softmax)

- Improve with edit. MD
"""


def main():
    # Get config
    config, _ = get_config()
    print_config()

    # Build Actor, Critic and Reward from config
    print("Initializing the actor...")
    if config.seeded_net==True:
        tf.set_random_seed(config.seed_net)
    actor = Actor(config)

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.trainable_variables()]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)  

    print("Starting training...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model==True:
            saver.restore(sess, "save/"+config.restore_from+"/actor.ckpt")
            print("Model restored.")
    
        # Summary writer
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        # Initialize data generator
        solver = []     #Solver(actor.max_length)
        training_set = DataGenerator(solver)

        # Get feed_dict (single graph)
        coord_batch = training_set.single_shuffled_batch(actor.batch_size, actor.max_length, actor.input_dimension, seed=75) #training_set.next_batch(actor.batch_size, actor.max_length, actor.input_dimension,seed=config.seed_data)
        feed = {actor.input_coordinates: coord_batch}

        # Solve instance
        #opt_length_batch = training_set.solve_batch([coord_batch[0]])
        #opt_length = opt_length_batch[0][0]
        #print('\n Optimal tour length: ',opt_length)

        for i in tqdm(range(config.nb_epoch)): # epoch i

            coord_batch = training_set.shuffle_batch(coord_batch)
            feed = {actor.input_coordinates: coord_batch}

            # Forward pass & Train step
            summary, base_op, train_step1 = sess.run([actor.merged, actor.base_op, actor.train_step1],feed_dict=feed)

            if i % 100 == 0:
                writer.add_summary(summary,i)
                baseline = sess.run(actor.avg_baseline,feed_dict=feed)
                print('\n Baseline:',baseline)

            # Save the variables to disk
            if i % max(1,int(config.nb_epoch/5)) == 0 and i!=0 :
                save_path = saver.save(sess,"save/"+config.save_to+"/tmp.ckpt", global_step=i)
                print("\n Model saved in file: %s" % save_path)

        # Print last epoch
        permutation, trip, circuit_length, reward = sess.run([actor.positions, actor.trip, actor.distances, actor.reward], feed_dict=feed)
        print('\n Permutations: \n',permutation[:10])
        for k in range(10):
            training_set.visualize_2D_trip(trip[k])

    
        print("Training is COMPLETE!")
        saver.save(sess,"save/"+config.save_to+"/actor.ckpt")





if __name__ == "__main__":
    main()