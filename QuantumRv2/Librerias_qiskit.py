# useful additional packages
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
# import networkx as nx

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit.optimization.applications.ising import max_cut, tsp
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua import aqua_globals
from qiskit.aqua import QuantumInstance
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.algorithms import MinimumEigenOptimizer,GroverOptimizer
from qiskit.optimization.problems import QuadraticProgram

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
# set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram


from qiskit import BasicAer
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer

from qiskit.optimization.converters import InequalityToEquality
from qiskit.optimization.converters import IntegerToBinary
from qiskit.optimization.converters import LinearEqualityToPenalty

from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
from qiskit.optimization import QuadraticProgram

import docplex as dc
from docplex.mp.model import Model as dcModel
from docplex.mp.model_reader import ModelReader

import docplex.mp.quad

from qiskit import IBMQ
import time
import pandas as pd

import numpy as np
from gurobipy import *



class q_model_file:
    bin_model='none'
    name='none'

    def __init__(self, qp_eq_bin, name):
        
        self.qp_eq_bin = qp_eq_bin
        self.name=name
        

    def write_file(self):        

        lista=self.qp_eq_bin.variables_index
        quadra=Model("hola")

        for k in lista:
            b=str(k)
            a=b.replace("@", "_")
            t=a+'=quadra.addVar(vtype=GRB.BINARY,name='+str('"'+a+'"')+')'
        #     print(t)
            exec(t)

        texto=self.qp_eq_bin.export_as_lp_string()
        texto=texto.replace("@","_")
        texto=texto.replace("*"," * ")
        texto=texto.replace("^"," ^")
        texto=texto.replace("]","] ")

        archivo=open(self.name+".lp","w")
        archivo.write(texto)
        archivo.close()

        return quadra

class QAOA_solve:
    
    name='none'
    qubo='none'
    times='none'
    p='none'
    init_p='none'
    seed='none'
    
    
    def __init__(self, 
                name='none',
                qubo='none',
                times='none',
                p='none',
                init_p='none',
                seed='none'
                ):
        
        self.name=name
        self.qubo=qubo
        self.times=times
        self.p=p
        self.init_p=init_p
        self.seed=seed 

        
    def solve(self):
            
        extended_stabilizer_simulator = QasmSimulator(method='matrix_product_state')
        aqua_globals.random_seed = self.seed
        quantum_instance = QuantumInstance(extended_stabilizer_simulator,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=self.init_p, p=self.p)
        qaoa = MinimumEigenOptimizer(qaoa_mes)   # using QAOA

        dicc={}

        for i in range(1,self.times+1):
            print("First iteration")
            inicio = time.time()
            qaoa_result = qaoa.solve(self.qubo)
            fin = time.time()
            tiempo=fin-inicio

            dicc[i]=qaoa_result.variables_dict
            dicc[i]['valor']=qaoa_result.fval
            dicc[i]['tiempo']=tiempo
        
        self.df = pd.DataFrame(dicc)
        self.df.to_excel(f'{self.name}-t_{self.times}-p_{self.p}.xlsx', sheet_name='QAOA')
        
        return self.df, dicc





