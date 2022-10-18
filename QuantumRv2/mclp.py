import os
import errno
import traceback

from datetime import datetime
import numpy as np
#from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

from gurobipy import *

from Librerias_qiskit import *

from QuantumRv2.Region import Region


#import Region


class MCLP(Region):
    
    """
    This class implements MCLP model
    Classical : Gurobipy
    Quadratic : Gurobipy
    Quantum   : Qiskit - QAOA
    Annealing : Dwave
    """
    
    prefix='MCLP'
    solution_vector=[]
    optim=False
    folder="Region"
    
    
    def __init__(self, 
                 prefix='None',
                 rows='None',
                 columns='None',
                 I='None',
                 J='None',
                 p='None',
                 S='None',
                 a='None', #Clientes
                 N='None'): #Vecinos, calculo interno
        
        self.exp_id='None'
        self.prefix=prefix
        self.rows=rows
        self.columns=columns
        self.I=I
        self.J=J
        self.p=p
        self.S=S
#         self.sites=sites
        self.a=a
#         self.N = N 
        
        self.rows    = rows
        self.columns = columns
        self.p       = p
        self.folder  = f"{self.prefix}_{self.rows}x{self.columns}_{self.p}"
        self.maindir = os.getcwd() 
        self.model   = None
        self.varGL   = {}
        self.varGQ   = {}
        self.Q_mod   = None
        self.qp_eq_bin = None
        self.QAOA_ban  = False
        self.MOD_Q     = None
        self.QAOA_dicc = None
        self.dataf     = None
        self.DF_Full   = None
        self.data      = None
    
    def distance(self,site_i,site_y):
        distance = np.sqrt((site_i[0]-site_y[0])**2+(site_i[1]-site_y[1])**2)
        return distance

    def Neighbors(self):
        self.N = {}        
        for i in self.I:
            for j in self.J:
                if self.distance(self.sites[i],self.sites[j]) <= self.S:
                    try:
                        self.N[i].append(j)
                    except:
                        self.N[i] = [j]
#         return N
    
    def createInstance(self):
        self.sites = {}
        # self.a = [] #population
        counter = 0
        for x in range(self.rows):
            for y in range(self.columns):
                self.sites[counter] = (x,y)
                # self.a.append(1)
                counter += 1
#         return sites,a

    

    def MCLPmin(self):
        
        self.execution_folder(self.folder)
        
        try:
            self.createInstance()
            self.Neighbors() 
            
            # Tolerance to non-integer solutions
            tol = 1e-5 #min value: 1e-9
    
            m = Model("MCLPmin")
    
            #y_i
            y = {}
            for i in self.I:
                y[i] = m.addVar(vtype=GRB.BINARY,name="y_"+str(i))
    
            self.temporal=y
            #x_j
            x = {}
            for j in self.J:
                x[j] = m.addVar(vtype=GRB.BINARY,name="x_"+str(j))
    
            m.update()
    
            # Objective function
            m.setObjective(quicksum(self.a[i]*y[i] for i in self.I), GRB.MINIMIZE) 
    
            # Constraints 1
            for i in self.I:
                temp = []
                if i in self.N.keys():
                    for j in self.N[i]:
                        temp.append(x[j])
                    if len(temp) >= 1:
                        m.addConstr(quicksum(temp)+ y[i] >= 1, "c1a_"+str(i))
                else:
                    m.addConstr(y[i] >= 1, "c1b_"+str(i))
    
            # Constraints 2
            temp = []
            for j in self.J:
                temp.append(x[j])
            m.addConstr(quicksum(temp) == self.p, "c2_"+str(i))
    
            m.update()
    
            #Writes the .lp file format of the model
            m.write(self.prefix+".lp" ) #self.prefix+"_MCLPmin_"+str(self.rows*self.columns)+"_"+str(self.S)+".lp")
            #m.write(prefix+"_MCLP_"+str(rows*columns)+"_"+str(S)+".mps")
    
            # To set the tolerance to non-integer solutions
            m.setParam('IntFeasTol', tol)
            # m.setParam('LogFile', self.prefix+"min_MCLP_"+str(self.rows*self.columns)+"_"+str(self.S))
            m.setParam('LogFile', self.prefix+str(self.S))
            m.params.timeLimit = 1800
            m.optimize()
            
            self.m=m
            self.model=m
            
            
            n = self.rows*self.columns
            nonCoverVector = np.zeros(n)
            selecFacilityVector = np.zeros(n)
            
            nonCover = {}
            for r in y.keys():
                if 'y_' in y[r].VarName:
                    if y[r].X == 1:
                        nonCover[r] = 1
                    else:
                        nonCover[r] = 0
                    

            for i in range(nonCoverVector.size): # range(n)
                nonCoverVector[i] = nonCover[i]

            self.nonCoverVector = list(nonCoverVector)
            
            
            selectedFacility = {}
            for r in x.keys():
                if 'x_' in x[r].VarName:
                    if x[r].X == 1:
                        selecFacilityVector[r] = 1
                    else:
                        selecFacilityVector[r] = 0              

            self.selecFacilityVector = list(selecFacilityVector)
            
            self.optim= True
            
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the main execution folder
        
        except:
            
            self.nonCoverVector = None
            self.selecFacilityVector = None
            m = None
            self.optim= False
            
            os.chdir(self.maindir) #Return to the main execution folder
            
            print("\n\nSE PRESENTÓ UN ERROR DURANTE LA EJECUCION DEL MÉTODO \n REVISE EL INFORME A CONTINUCIACÓN:\n")
            var = traceback.format_exc()
            print(var)   
            
        return nonCoverVector,selecFacilityVector,m
    
    
    

    def Display_mod(self):
        print("++++ FORMULACIÓN DEL MODELO ++++")
        self.m.display()
        print("++++ EJECUCIÓN DEL MODELO ++++")
        self.m.optimize()
        print("++++ solución DEL MODELO ++++")
        self.m.printAttr('x')
        
    def var_result(self):
        self.m.printAttr('x')

    def name_file(self):
        name=self.prefix+".lp" #+"_MCLPmin_"+str(self.rows*self.columns)+"_"+str(self.S)+".lp"
        return name
    
    def readFile(self,file,sheet,exp_id):
        """
        Reads an Excel file with:
        
        file   = file name
        sheet  = sheet name
        exp_id = number of the experiment to be run.
        
        Excel file should have this columns format: 
           |name|p|fil|col|DATA CELLS
        
        name = Name of the experiment
        I    = Set of demand nodes
        p    = Number of p facilities
        S    = service distance
        J    = set of facilitie sites
        clientes = set of clients in each node
        fil  = Number of rows
        col  = Number of columns
        
        J         = |x0|     |      |        |x4|   | ... 
        DATACELLS = |value_1|value_2|value_3|...|value_(fil*col)|
        """
        
        self.maindir = os.getcwd()   
        
        df            = pd.read_excel(file,sheet_name=sheet)   
        self.exp_id   = exp_id
        tipo          = sheet
        
        self.rows     = df.iloc[exp_id]['Fil']
        self.columns  = df.iloc[exp_id]['Col']
        self.prefix   = str(tipo+
                            str(df.iloc[exp_id]['num'])+ "_"+
                            df.iloc[exp_id]['Nombre']+
                            f"_{self.rows}x{self.columns}-"+
                            str(df.iloc[exp_id]['p']))

        self.I        = range(df.iloc[exp_id]['Fil']*df.iloc[exp_id]['Col']) # Set of demand nodes        
        self.p        = df.iloc[exp_id]['p']    #number of facilities
        self.S        = df.iloc[exp_id]['s']    #service distance
    
        self.J        = [(j) for j in range(0,(self.columns*self.rows),1) 
                         if str(df.iloc[exp_id-1][j])!='nan']           #set of facility sites
        
        self.clientes = list(int(df.iloc[exp_id][j]) 
                             for j in range(0,(self.columns*self.rows),1))
        
        self.folder   = f"{exp_id}_{self.prefix}"
        
        
        self.a        = self.clientes
        
        FacilityMap = [0 for i in range(0,(self.columns*self.rows),1)] 
        for i in range(0,(self.columns*self.rows),1):
            if i in self.J :
                FacilityMap[i]=1
       
        
        self.facilityMap = FacilityMap
        
        return self.I, self.J, self.clientes
    

    def graphGrid(self):
        """
        Creates the png files through the create_grid2 method
        
        """
        self.execution_folder(self.folder)
        
        try:    
            
            if self.optim==False:
                print("No se ha optimizado el modelo")
                self.create_grid2("Problem clients",
                                  self.clientes,
                                  output_path_and_filename=(f"Problem-clients_ID-{self.exp_id}"))
                
                
                self.create_grid2("Problem Facilities",
                                  self.facilityMap,
                                  output_path_and_filename=(f"Problem-Facilities_ID-{self.exp_id}"))
                
                
            else:
                self.create_grid2("Problem clients",
                                  self.clientes,
                                  output_path_and_filename=(f"Problem-clients_ID-{self.exp_id}"))
                
                
                self.create_grid2("Problem Facilities",
                                  self.facilityMap,
                                  output_path_and_filename=(f"Problem-Facilities_ID-{self.exp_id}"))
                
                self.create_grid2("Selected Facilities-Gurobi",
                                  [int(i) for i in self.selecFacilityVector],
                                  output_path_and_filename=(f"Selected_Facilities-Gurobi_ID-{self.exp_id}"))

                self.create_grid2("Non cover sites-Gurobi",
                                  [int(i) for i in self.nonCoverVector],
                                  output_path_and_filename=(f"NonCover_sites-Gurobi_ID-{self.exp_id}"))            
                
                
            if self.QAOA_ban == True:
                
                for k in range(1,self.qaoaTimes+1,1):
                
                    self.qaoaVectorSol=[0 for i in range(0,(self.columns*self.rows),1)] 
                    
                    for i in self.QAOA_dicc[k]:
                        if "x_" in i:
                            if self.QAOA_dicc[k][i]==1:
                                self.qaoaVectorSol[int(i[2:])]=1
                               
                    self.create_grid2(f"QAOA Solution No {k}",
                                      [int(i) for i in self.qaoaVectorSol],
                                      output_path_and_filename=(f"QAOA_No_{k}_Solution_ID-{self.exp_id}"))            

           
            if self.annealBan ==True:
                
                     
                for k in range(1,self.annealTimes,1):
                
                    annealVectorSol=[0 for i in range(0,(self.columns*self.rows),1)] 
                
                    for i in self.annealDicc[k]:
                        if "x_" in i:
                            if self.annealDicc[k][i]==1:
                                annealVectorSol[int(i[2:])]=1
                
                
                    self.create_grid2(f"Anneal Solution No {k}",
                                      [int(i) for i in annealVectorSol],
                                      output_path_and_filename=(f"Anneal_No_{k}_Solution_ID-{self.exp_id}"))       
                                
                 
                
                
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the main execution folder
            
        except:
            
            
            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)    
    
    
    
    
    
    
    