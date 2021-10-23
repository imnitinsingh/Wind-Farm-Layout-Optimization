# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:36:59 2020

@author: Nitin.N.Singh
"""

from Farm_Evaluator import main

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import progressbar
progress = progressbar.ProgressBar()
from tqdm import tqdm

from pyomo.opt import TerminationCondition
from pyomo.opt import SolverStatus
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
from coopr.pyomo import ConcreteModel, Set, Var, Constraint, Objective


def turb_coords_idxRu(m):
    """
        Function that returns ordered tuple...
            
    """

    index_ = []
    for i in range(50):
        index_.append(('p{0}'.format(i),'x'))
        index_.append(('p{0}'.format(i),'y'))
    return index_ 


def turb_loc_initRu(m, a, b):
    """
       Initializer rule var
    """
    turbine_loc_init = pd.read_csv(r'C:\Users\Nitin.N.Singh\Desktop\HACK\Optimization\Data\turbine_loc_init.csv')
    
    if b=='x':
        turb_coords = turbine_loc_init['xc'].iloc[int(a[1:])]
    else:
        turb_coords = turbine_loc_init['yc'].iloc[int(a[1:])]
    return(turb_coords)

# %%
# Create SolverFactory using ipopt solver plugin and ASL interface
# SolverFactory to instance a solver
solver    = 'glpk'                # Interior Point Optimizer
solver_io = 'nl'                   # nl (AMPL style) solver interface
opt       = SolverFactory(solver)#, solver_io=solver_io)
#opt.options['max_iter'] = 50

# For Fancy Stream Output
stream_solver = True
keepfiles     = False
# %%

#main()

if __name__ == "__main__":
    
    farm_peri = [[0,0],[0,4000],[4000,4000],[4000,0]]
    
    model = ConcreteModel()
    
    model.turb_coords_idx = Set( dimen=2, rule=turb_coords_idxRu, ordered=True )
    
    # define variables
    model.turb_coords     = Var(
                                  model.turb_coords_idx, 
                                  initialize=turb_loc_initRu,
                                  bounds= (0, 2000)
                               )
    
    
    
    # define constraints
    # define that now rwo points can be close to each other then 4*d_rotor
    
    
    
    # define objective function
    model.objective_fun = Objective(rule=main)
    # model.objective_fun = Objective()
    results = opt.solve(model, tee=stream_solver)
    
    
    
    # optimize
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    