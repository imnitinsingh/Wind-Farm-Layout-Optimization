# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:21:13 2020

@author: Nitin.N.Singh
"""

import math
import numpy as np
import numpy.random as npr
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from datetime import datetime
import pickle
from tqdm import tqdm

#import warnings
#warnings.filterwarnings("ignore")

from Farm_Evaluator_Vec2 import farmEvaluator, loadPowerCurve, binWindResourceData, preProcessing

class WindFarmGenetic:
    r"""parameters for genetic algorithms"""
    elite_rate  = 0.2
    cross_rate  = 0.6
    random_rate = 0.5
    mutate_rate = 0.1

    turbine = None

    pop_size = 0         # number of individuals in a population
    N = 0                # number of wind turbines
    rows = 0             # number of rows : cells
    cols = 0             # number of columns : cells
    cell_width = 0       # cell width
    cell_width_half = 0  # half cell width
    iteration = 0        # number of iterations of genetic algorithm

    r"""constructor of class WindFarmGenetic"""
    def __init__(self, elite_rate=0.2, cross_rate=0.6, random_rate=0.5, 
                 mutate_rate=0.1, pop_size=0, N=0, rows=0, cols=0, cell_width=0,
                 iteration=0):
        # self.turbine = GE_1_5_sleTurbine() ------>> need to figure this out but I think only dia is going in
        self.elite_rate     = elite_rate
        self.cross_rate     = cross_rate
        self.random_rate    = random_rate
        self.mutate_rate    = mutate_rate
        self.pop_size       = pop_size
        self.N              = N
        self.rows           = rows
        self.cols           = cols
        self.cell_width     = cell_width
        self.cell_width_half= cell_width * 0.5
        self.iteration      = iteration
        
        self.init_pop                  = None
        self.init_pop_nonezero_indices = None

        return 
    
    
    def gen_init_pop(self):
        #np.random.seed(seed=int(time.time()))
        layouts = np.zeros((self.pop_size, self.rows*self.cols), dtype=np.int32)
        dummy_solution = np.array([0]*int(self.rows*self.cols-self.N) + [1]*self.N)
        
        for i in range(self.pop_size):
            np.random.shuffle(dummy_solution)
            layouts[i] = dummy_solution
            
        self.init_pop = layouts
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for i in range(self.pop_size):
            idx = 0
            for j in range(self.rows*self.cols):
                if self.init_pop[i, j] == 1:
                    self.init_pop_nonezero_indices[i, idx] = j
                    idx += 1
                    
        # Artifically imputing some good populaton members
#        test_array = np.zeros((10,2))
#        test_array[:,0] = [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]
#        test_array[:,1] = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]
#        test_array = np.column_stack([test_array]*5).flatten()
#        test_array[0] = 0; test_array[1] = 1 
#        layouts[0:1] = test_array
        
            
        return #(self.init_pop, self.init_pop_nonezero_indices)
    
    
    def conventional_genetic_alg(self):  # conventional genetic algorithm
        print("conventional genetic algorithm starts....")
        
        # best fitness value in each generation
        fitness_generations         =  np.zeros(self.iteration, dtype=np.float32)  
        
        # best layout in each generation
        best_layout_generations     =  np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32) 
        
        fitness_values_generations =   np.zeros((self.iteration, self.pop_size),
                                           dtype=np.int32)
        
        # each row is a layout cell indices. in each layout, order turbine power from least to largest
        # power_order = np.zeros((self.pop_size, self.N), dtype=np.int32)  
        self.gen_init_pop()
        pop         = np.copy(self.init_pop)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.
        
        # Do the preprocessing from module Farm_Evaluator_Vec2
        power_curve   =  loadPowerCurve(r'.\Data\power_curve.csv')
        wind_inst_freq     =    binWindResourceData(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Final Datasets\Benbecula\Filtered\wind_data_2005.csv')
        n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)
        
        for gen in tqdm(range(self.iteration)):
            print("generation {}...".format(gen))
            fitness_value = np.zeros((self.pop_size,), dtype=np.float32) 
            
            for i in range(self.pop_size):
                fitness_value[i] = farmEvaluator(pop[i].reshape(10,10), power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
                # print('=======>>>', fitness_value[i])
            fitness_values_generations[gen,:] = fitness_value
            
            # fitness value descending from largest to least
            # returns indices that would sort an array
            sorted_index = np.argsort(-fitness_value)  
            
            # you have sorted the pop
            pop = pop[sorted_index, :]
            # power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]
            fitness_value = fitness_value[sorted_index]
            
            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
            
            n_parents, parent_layouts, parent_pop_indices = self.conventional_select(pop=pop, pop_indices=pop_indices,
                                                                                     pop_size=self.pop_size,
                                                                                     elite_rate=self.elite_rate,
                                                                                     random_rate=self.random_rate, fitness_value = fitness_value)
            pop, pop_indices = self.conventional_crossover(N=self.N, pop=pop, pop_indices=pop_indices, 
                                                           pop_size=self.pop_size, n_parents=n_parents,
                                                           parent_layouts=parent_layouts, 
                                                           parent_pop_indices=parent_pop_indices, 
                                                           cross_rate = self.cross_rate)
            pop, pop_indices = self.conventional_mutation(rows=self.rows, cols=self.cols, N=self.N, pop=pop, 
                                                          pop_indices=pop_indices, pop_size=self.pop_size,
                                                          mutation_rate=self.mutate_rate)
        
            
            print('Best fitness after {0} generations'.format(gen), fitness_generations[gen])
            
        print("conventional genetic algorithm ends.")
        return (fitness_generations, best_layout_generations, fitness_values_generations)#run_time, eta_generations[self.iteration - 1]
    
    
    def conventional_select(self, pop, pop_indices, pop_size, elite_rate, random_rate, fitness_value):
        r"""Roulette Wheel Selection"""
        
        def selectOne(fitness_value):
            max_ = sum(fitness_value)
            selection_probs = [f/max_ for f in fitness_value]
            return (npr.choice(len(fitness_value), p=selection_probs))
        
        r'''
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        #parents_ind = []
        
        for i in range(n_elite, pop_size):
            if np.random.randn() < random_rate:
                parents_ind.append(i)
        '''
        parents_ind = []
        for i in range(pop_size):
            parents_ind.append(selectOne(fitness_value))
        
        # print('*****--->', parents_ind)
        parent_layouts = pop[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        
        return len(parent_pop_indices), parent_layouts, parent_pop_indices
    
    
    def conventional_crossover(self, N, pop, pop_indices, pop_size, n_parents,
                               parent_layouts, parent_pop_indices, cross_rate):
        n_counter = int(pop_size * self.elite_rate)
        
        while n_counter < pop_size:
            
            male   = np.random.randint(0, n_parents)
            female = np.random.randint(0, n_parents)
            
            cross_point = np.random.randint(1, N)
            
            if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                # print('success')
                pop[n_counter, :] = 0
                pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                 :parent_pop_indices[
                                                                                      male, cross_point - 1] + 1]
                pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                           parent_pop_indices[female, cross_point]:]
                pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                n_counter += 1
            #else:
                # print('failure')
               # if np.sum(pop[male, 0:cross_point]) == np.sum(pop[female, 0:cross_point]):
            # print('****', n_counter)
        return pop, pop_indices
    
    
    def conventional_mutation(self, rows, cols, N, pop, pop_indices, pop_size, mutation_rate):
        #np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            if np.random.randn() > mutation_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, cols * rows)
                if pop[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1
            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])
        return pop, pop_indices


if __name__ == "__main__":
    
    r'''main function'''
    
    # parameters for the genetic algorithm
    elite_rate  = 0.25
    cross_rate  = 0.6
    random_rate = 0.5
    mutate_rate = 0.15
    
    # wind farm size, cells
    rows = 10
    cols = 10
    cell_width = 150.*2 # unit : m
    
    N = 50  # number of wind turbines
    pop_size = 100  # population size, number of inidividuals in a population
    iteration = 100  # number of genetic algorithm iterations/generations
    
    # make return the 50 by 50 turbine
    wfg = WindFarmGenetic(elite_rate=elite_rate, cross_rate=cross_rate, 
                          random_rate=random_rate, mutate_rate=mutate_rate,
                          pop_size=pop_size,N=N,rows=rows, cols=cols,  
                          cell_width=cell_width, iteration=iteration)
    best_fitness, best_layout, population_fitness = wfg.conventional_genetic_alg()
