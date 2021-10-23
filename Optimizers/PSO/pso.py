# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:17:43 2020

@author: Nitin.N.Singh
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plt
from Farm_Evaluator_Vec import farmEvaluator, loadPowerCurve, binWindResourceData, preProcessing


power_curve    =  loadPowerCurve(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\GA\Data\power_curve.csv')
wind_inst_freq = binWindResourceData(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Final Datasets\Benbecula\Filtered\wind_data_2005.csv')
n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)

popSize      = 100
nGenerations = 100

r'''Create initial Population'''
population = np.zeros((popSize, 50, 2), dtype = np.float32)
dirPath = r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\Data'
for i in range(popSize):
    #file = pd.read_csv(dirPath + r'\turbine_loc_reg1.csv')
    
    #population[i] = file.values + 150*np.random.uniform(low=-1, high=1, size=(50,2))
    population[i] = test_sol + 5*np.random.uniform(low=-1, high=1, size=(50,2))

population[-1] = test_sol

        
r'''Impart initial velocity to each of the particle'''
velocities =  np.random.uniform(low=0, high=1, size=(popSize, 50,2))

    
r'''define PSO parameters'''
w  = 0.10
c1 = 0.25
c2 = 2.75


r'''For recording the local best and global best'''
globalBestFitness = 0
localBestFitness  = np.zeros((popSize,))    
localBestPositions = np.zeros((popSize, 50, 2), dtype = np.float32)

bestSolutions = np.zeros((nGenerations, 50, 2), dtype = np.float32)
        
r'''Do the PSO Algorithm'''
for n in range(nGenerations):
    
    r'''record local and global bests'''
    for i in range(popSize):
        fitness = -1*farmEvaluator(population[i], power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
    
        if fitness > globalBestFitness:
            #if n != 0:
            #    print('updated')
            globalBestFitness = np.copy(fitness)
            globalBestIdx     = i
            globalBestPosition = np.copy(population[i]) 
            
        if fitness > localBestFitness[i]:
            localBestFitness[i] = np.copy(fitness)
            localBestPositions[i] = np.copy(population[i])
            
    r'''Update the velocities of the particles once have global best captured'''
    for i in range(popSize):
        r1 = np.random.uniform(low=0, high=1, size=(50,2))
        r2 = np.random.uniform(low=0, high=1, size=(50,2))                
        velocities[i] = w*velocities[i] + c1*r1*(localBestPositions[i] - population[i]) + \
                        c2*r2*(globalBestPosition - population[i]) 
    population += velocities 
    
    bestSolutions[n] = globalBestPosition
    print('Best Fitness about {0} generations is {1}'. format(n, globalBestFitness))
    

for n in range(nGenerations):
    plt.figure()
    plt.plot(bestSolutions[n,:,0], bestSolutions[n,:,1], 'o')
    plt.savefig(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\PSO\Images\fig{0}.png'.format(n))
    plt.close()   


r'''Try Scipy Opt bro'''                
            




