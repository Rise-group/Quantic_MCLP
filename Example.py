##EXPERIMENT NUMBER 1

from QuantumRv2 import mclp as mp
import numpy as np


Exp_ID=1

modelo = mp.MCLP()
modelo.readFile("EXPERIMENTS.xls",'MCLP',Exp_ID)

modelo.MCLPmin()
modelo.linear_quadratic()
modelo.quadratic_gurobi()
modelo.Annealer()

# seed=np.random.randint(5000,10000)
seed=10000
i=np.pi/2
j=np.pi/3

## QAOA EXECUTION
## NOTE: TO TEST, FIRST EXECUTE WITHOUT THIS INSTRUCTION
# modelo.QAOA_Optim(deep   = 1,
#                   times  = 1,
#                   init_p = [i,j],
#                   seed   = seed )


