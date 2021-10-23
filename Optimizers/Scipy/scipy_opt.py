# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:00:41 2020

@author: Nitin.N.Singh
"""

# A bit similar to what fmincon does in Matlab

import os; os.chdir(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\Scipy')

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy
from scipy import optimize
from scipy.optimize import basinhopping

import cma
opts = cma.CMAOptions()
opts.set("bounds", [[50, 3950]*100])


from matplotlib import pyplot as plt
from Farm_Evaluator_Vec import farmEvaluator, loadPowerCurve, binWindResourceData, preProcessing


power_curve    =  loadPowerCurve(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\GA\Data\power_curve.csv')
wind_inst_freq = binWindResourceData(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Final Datasets\Benbecula\Filtered\wind_data_2019.csv')
n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)


dirPath = r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\Data'
file = pd.read_csv(dirPath + r'\turbine_loc_reg1.csv')

# initial_solution = file.values + 150*np.random.uniform(low=-1, high=1, size=(50,2))
#initial_solution = pd.read_csv(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\Data\turbine_loc_reg0.csv').values
#initial_solution = np.load(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\CMA-ES\a-good-solution.npy')
#initial_solution = initial_solution 
initial_solution = initial_solution  = agoodsolution2
initial_obj_fun_val = farmEvaluator(initial_solution.flatten(),power_curve, wind_inst_freq, n_wind_instances, 
                                     cos_dir, sin_dir, wind_sped_stacked, C_t)

# print('Intial Objective function is: ', -1*initial_obj_fun_val)

#minimizer_kwargs = { "method": "Nelder-Mead", 'args':(power_curve, wind_inst_freq, n_wind_instances, 
#                                     cos_dir, sin_dir, wind_sped_stacked, C_t), 'tol':1e-4}
#res= basinhopping(farmEvaluator, initial_solution.flatten(), 
#                             minimizer_kwargs=minimizer_kwargs,niter=10)

#res= scipy.optimize.differential_evolution(farmEvaluator, bounds = [(50, 3950)]*100,
#                             args=(power_curve, wind_inst_freq, n_wind_instances, 
#                                     cos_dir, sin_dir, wind_sped_stacked, C_t), disp = True, 
#                                   init = de_init_solutions1, maxiter = 1000, tol=0.0001,
#                                   strategy = 'randtobest1exp', mutation = 0.10, seed  = 1
#                                   , recombination=0.9)
 
res = cma.fmin(farmEvaluator, args=(power_curve, wind_inst_freq, n_wind_instances, 
                                     cos_dir, sin_dir, wind_sped_stacked, C_t), 
                                       x0 = initial_solution.flatten(), 
                                       sigma0 =200, restarts =5, restart_from_best= True,
                                       options={'ftarget':-520, 'bounds':[50, 3950]#, 'popSize':1000
                                                ,'CMA_diagonal':1000,'tolfun':1e-4})#, 'AdaptSigma':True,
                                                #'CMA_cmean':0.30, 'CMA_mirrormethod':0})

# beat 5.77 at 1000 fevals                            
#print('Optimized Objective Function: ', -1*res['fun'])

#plt.plot(initial_solution[:,0], initial_solution[:,1], 'o')
#plt.plot(res['x'].reshape(50,2)[:,0], res['x'].reshape(50,2)[:,1], 'o')


# To do
# I can let things run for a while after identofying correct params. 
# CMA_mirrormethod 0 


                             