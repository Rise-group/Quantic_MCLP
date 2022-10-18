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



class Region():
    
    """
    This class implements region model
    Classical: Gurobipy
    Quadratic : Gurobipy
    Quantum  : Qiskit - QAOA
    Annealing: Dwave
    """
   
    prefix="Region"
    solution_vector=[]
    optim=False
    folder="Region"
    
    def __init__(self,
                 rows    = None,
                 columns = None,
                 p       = None,
                 data    = None,
                 prefix  = None
                ):
        
        self.rows    = rows
        self.columns = columns
        self.p       = p
        self.data    = data
        self.prefix  = prefix
        self.folder  = f"{self.prefix}_{self.rows}x{self.columns}_{self.p}"
        self.maindir = os.getcwd() 
        self.model   = None
        self.varGL   = {}
        self.varGQ   = {}
        self.Q_mod   = None
        self.qp_eq_bin = None
        self.QAOA_ban = False
        self.MOD_Q = None
        self.QAOA_dicc = None
        self.dataf = None
        self.DF_Full =None
        self.annealBan=False
    
        """
        Create the neighboring links for the model
        """
        
        change = 0
        for link in links:
            if len(link.intersection(tree)) > 0:
                tree = tree.union(link)
                links.remove(link)
                change = 1
        return tree, links, change

    def graphGrid(self):
        """
        Creates the png files through the create_grid2 method
        
        """
        self.execution_folder(self.folder)
        
        try:    
            print("AnnealBan graph=",self.annealBan)
            if self.optim==False:
                print("No se ha optimizado el modelo")
                self.create_grid2("Problem",self.data,output_path_and_filename=("Problema"))
            else:
                self.create_grid2("Problem",self.data,output_path_and_filename=("Problema"))
                self.create_grid2("Gurobi Lineal Solution",self.solution_vector,output_path_and_filename=("Lineal Gurobi Solution"))
            
            if self.QAOA_ban == True:
                
                for k in range(1,self.qaoaTimes+1,1):
                
                    self.qaoaVectorSol=[0 for i in range(0,(self.columns*self.rows),1)] 
                    
                    for i in self.QAOA_dicc[k]:
                        if "x_" in i:
                            if self.QAOA_dicc[k][i]==1:
                                self.qaoaVectorSol[int(i[-1:])]=1
                               
                    
                    self.create_grid2(f"QAOA Solution No {k}",
                                      [int(i) for i in self.qaoaVectorSol],
                                      output_path_and_filename=(f"QAOA_No_{k}_Solution_ID-{self.exp_id}"))            
            
           
            
            if self.annealBan ==True:
                
                     
                for k in range(1,self.annealTimes,1):
                
                    annealVectorSol=[0 for i in range(0,(self.columns*self.rows),1)] 
                
                    for i in self.annealDicc[k]:
                        if "x_" in i:
                            if self.annealDicc[k][i]==1:
                                annealVectorSol[int(i[-1:])]=1
                
                
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
    
    def create_grid2(self, title,values,output_path_and_filename = None, plot = True):
        """
        Creates the png files
        
        """
        
        X = [values[i:i + self.columns] for i in range(0, len(values), self.columns)] 
        X = np.array(X)

        fig, ax = plt.subplots()
        ax.imshow(X)

        rows, columns = X.shape


        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if 0 <= col < columns and 0 <= row < rows:
                z = X[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        
        for r in range(self.rows):
            for c in range(self.columns):
                plt.text(c,r,X[r,c],fontsize=15,bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 5})

        ax.format_coord = format_coord
        plt.axis('off')
        plt.suptitle(title)
        plt.savefig(output_path_and_filename, dpi=300, bbox_inches='tight')
        plt.show()

    
      
    def execution_folder(self,folder):
        #### We create the folder for the experiment
        try:
            os.mkdir(f"{folder}")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        ### We change the execution folder to save the files inside it
        mydir = os.getcwd() # would be the MAIN folder
        mydir_tmp = mydir + f"/{folder}" # add the test folder name## mydir + f"\\{folder}"
        mydir_new = os.chdir(mydir_tmp) # change the current working directory
        mydir_2 = os.getcwd() # set the main directory again, now it calls the new test directory
        #####
    
    def linear_quadratic(self):
        archivo=self.prefix+".lp"
        
        self.execution_folder(self.folder)
        
        try:    
                    
            mod2 = QuadraticProgram()
            ineq2eq = InequalityToEquality()
            lineq2penalty = LinearEqualityToPenalty()
            int2bin = IntegerToBinary()
        
            mod2.read_from_lp_file(archivo)
            qp_eq = ineq2eq.convert(mod2)
            qubo = lineq2penalty.convert(qp_eq)
            qp_eq_bin = int2bin.convert(qubo)
        
            ## Changing variables name that has @, to avoid errors in other functions
            texto=qp_eq_bin.export_as_lp_string()
            texto=texto.replace("@","_")
        
            archivo_qubo=open("qubo.lp","w")
            archivo_qubo.write(texto)
            archivo_qubo.close()
        
            qp_eq_bin.read_from_lp_file("qubo.lp")
            self.qp_eq_bin = qp_eq_bin
        
        
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder
        
        except:
            
            self.qp_eq_bin = None
            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)
        
        
        return self.qp_eq_bin

    def get_vars(self):
        
        allvars = self.model.getVars() 
        varGL={}
        for i in allvars: varGL[i.varName]=i.x 
        self.varGL=varGL
        
        return self.varGL

    def vars_GQ(self):
        
        allvars = self.Q_mod.getVars()  
        varGQ={}
    
        try:         
            for i in allvars: varGQ[i.varName]=i.x
            #varGQ
            self.varGQ=varGQ
        
        except:
            
            print("----------\n"+
                  "\nSe obtuvieron las variables del modelo sin embargo\n"+
                  "no se puede obtener el VALOR de las mismas,\n" + 
                  "porque no se cuenta con la licencia de \n" +
                  "GurobyPy para trabajos grandes\n"+ 
                  "Todos los valores fueron asignados a 0\n"+
                  "y puede continuar con otros procesos como el Annealing\n\n"+
                  "----------\n")


            variables=self.Q_mod.getVars()
            DvarGQ={}
            
            for i in variables:
                var=str(i)
                var = var.replace("<","")
                var = var.replace(">","")
                var = var.replace("gurobi.Var","")
                var = var.replace(" ","")
                if (var.isnumeric()):
                    pass
                else:
                    DvarGQ[var]=0
  
            varGQ=DvarGQ
            self.varGQ=varGQ
            
        return self.varGQ



    def quadratic_gurobi(self):
        
        self.execution_folder(self.folder)
        
        try:
            qp_eq_bin = self.qp_eq_bin
            
            quadra=q_model_file(qp_eq_bin,"QUBO")
            quadra.write_file()
            Q_mod=read("QUBO.lp")
            
            try:
                Q_mod.optimize()
            except:
                print("Se necesita la licencia de GurobiPy para casos con gran numero de variables")
            
            self.Q_mod = Q_mod
        
        
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder
            
        except:
            
            self.Q_mod = None
            os.chdir(self.maindir) #Return to the main execution folder
            
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)             ### Return to the original execution folder
        
        return self.Q_mod 
              

    def Annealer(self):
        
        variables = self.vars_GQ()
        Ann_objective = self.Q_mod.getObjective()
        
        self.execution_folder(self.folder)
        
        try:
            ### Eliminate some integers that gurobi delivers in the vars dictionary, and that raise in errors
            list_varGQ=list(variables)
            for i in list_varGQ:
                try:
                    i=int(i)
                    list_varGQ.remove(str(i))
                except:
                    continue
            # list_varGQ
            
            ### Creating the Excel file that is going to be used in the Annealer
            df_varGQ=pd.DataFrame(list_varGQ,columns=["Variables"])
            df_varGQ.to_excel("Anneal_varGQ.xls","Quad_vars")
            
            ### Write the variables in a txt file to pass them to the annealer plattform
            var_text=str(list_varGQ)
            arch_var_text=open("Anneal_var.txt","w")
            arch_var_text.write(var_text)
            arch_var_text.close()
            
            ### We get the Quadratic Objective function to pass it to he annealer plattform
            ### This is necesary due to the different notation between platforms
            quad_eq=str(Ann_objective)[17:-1]
        
            quad_eq=quad_eq.replace(".0","")
            quad_eq=quad_eq.replace(" ","")
            quad_eq=quad_eq.replace("+-","-")
            quad_eq=quad_eq.replace("[","(")
            quad_eq=quad_eq.replace("]",")")
            quad_eq=quad_eq.replace("^","**")
        
            for i in range(0,10,1):
                quad_eq=quad_eq.replace(f"{i}y",f"{i}*y")
                quad_eq=quad_eq.replace(f"{i}x",f"{i}*x")
                quad_eq=quad_eq.replace(f"{i}c",f"{i}*c")
                quad_eq=quad_eq.replace(f"{i}t",f"{i}*t")
            # quad_eq
        
            quad_text=str(quad_eq)
            arch_quad_text=open("Anneal_quad.txt","w")
            arch_quad_text.write(quad_text)
            arch_quad_text.close()
           
            
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder
 
        except:

            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)
        
        
 
    def QAOA_Optim(self,
                   deep   = 1,
                   name   = 'PREGION',
                   times  = 1,
                   init_p = [np.pi/2,np.pi/3],
                   seed   = 10000
                   ):
        
        self.execution_folder(self.folder)
        self.qaoaTimes=times
        
        try:
    
            MOD_Q  = QAOA_solve(name, self.qp_eq_bin, times, deep, init_p, seed)
            dataf,QAOA_dicc = MOD_Q.solve()
                          
            for i in range(0,len(init_p),1):
                QAOA_dicc[1][f"Parameter {i}"] = init_p[i]
            
            QAOA_dicc[1]["Seed"]=seed
            
            
            self.MOD_Q = MOD_Q
            self.QAOA_dicc = QAOA_dicc
            self.dataf = dataf
            self.QAOA_ban = True
   
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder
            
        except:
            
            self.QAOA_ban = False
            self.MOD_Q = None
            self.QAOA_dicc = None
            self.dataf = None

            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)             ### Return to the original execution folder
         
 
    
 
    def save_experiments(self):
        
        self.execution_folder(self.folder)
        
        try:
            
            file_Full = {}
            self.get_vars()
            
            file_Full["Lineal Gurobi"]    = self.varGL
            file_Full["Quadratic Gurobi"] = self.varGQ
            
            if self.QAOA_ban == True:
            
                for i in self.QAOA_dicc:    
                    file_Full[f"QAOA {i}"] = self.QAOA_dicc[i]
            
            if self.annealBan ==True:
                for i in self.annealDicc:
                    file_Full[f"Annealer {i}"] = self.annealDicc[i]
                
            
            self.DF_Full=pd.DataFrame(file_Full)
            
                       
            now = datetime.now()
            
            self.DF_Full.to_excel(f'{self.prefix}-F({now.day}_{now.month}_h{now.hour}).xlsx',
                             sheet_name='Resultados')
            
            print(f'{self.prefix}-F({now.day}_{now.month}_h{now.hour}).xlsx')

            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder
        
        except:
            
            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)             ### Return to the original execution folder
            
    
           
    def configAnneal(self,
                     my_bucket,
                     my_prefix,
                     device  
                     ):
        """
        Import libraries to run in Amazon braket Dwave's system
        my_bucket = "amazon-braket-experimentos" # the name of the bucket
        my_prefix = "DWAVE" # the name of the folder in the bucket
        device = AwsDevice("arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6")
        Atributes:
        s3_folder
        annealBQM
        annealConstraint
        annealDevice
        annealDicc
        annealEquation
        annealModel
        annealSampler
        annealSamples
        annealVarList
        annealTimes
        
        """

       
        # Libraries to execute in AWS Amazon Dwave Annealer
        import json
        from braket.aws import AwsDevice
        from braket.ocean_plugin import BraketSampler, BraketDWaveSampler
        import networkx as nx
        import dwave_networkx as dnx
        from dimod.binary_quadratic_model import BinaryQuadraticModel
        from dwave.system.composites import EmbeddingComposite
        from pyqubo import Binary, Constraint, Spin
        from dimod import ExactSolver
        import time
        from dwave.system.composites import EmbeddingComposite
 
        # Folder of the experiment
        self.execution_folder(self.folder)
        
        try:
            
            # Enter the S3 bucket you created during onboarding in the code below
            self.s3_folder = (my_bucket, my_prefix)
            self.annealDevice = AwsDevice(device)
            
            # Annealer variables    
            variables=open("Anneal_var.txt")
            var_Quad=str(variables.readlines())
            var_Quad=var_Quad[2:-2]
            # exec(f"list_var={var_Quad}")
            exec(f"self.list_var={var_Quad}")
            
            self.annealVarList=self.list_var
            
            for i in self.annealVarList: 
                a=str(i)
                t=a+'= Binary('+str('"'+a+'"')+')'
                #t=a+'= Spin('+str('"'+a+'"')+')'
                exec(t)
            
            # Read de quadratic file
            archivo=open("Anneal_quad.txt")
            
            annealEquation=str(archivo.readlines())
            annealEquation=annealEquation[2:-2]
            self.annealEquation=annealEquation
            
            # Creates de constraint model
            instruccion= f'self.H = Constraint({self.annealEquation}, "const1")'
            exec(instruccion)
            
            self.annealConstraint=self.H
            
            # Select de Dwave Annealer
            sampler1 = BraketDWaveSampler(self.s3_folder,'arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6')
            self.annealSampler = EmbeddingComposite(sampler1)
            
            #Creates de model for the annealer
            modelAnn = self.H.compile()
            bqm = modelAnn.to_bqm()            
            self.annealModel=modelAnn
            self.annealBQM=bqm
            
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder

        except:
            
            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)             ### Return to the original execution folder
            
            

    def execAnneal(self,
                   times,
                   numReads=10000
                   ):
        
        #----GO TO THE EXPERIMENT'S FOLDER----
        self.execution_folder(self.folder)
        
        try:
            
            self.annealDicc={}
            self.annealTimes=times
            
            for i in range(1,times):
                
                inicio = time.time()
            
                sampleset = self.annealSampler.sample(self.annealBQM, num_reads=numReads)
                decoded_samples = self.annealModel.decode_sampleset(sampleset)
                best_sample = min(decoded_samples, key=lambda x: x.energy)
                ind_var=best_sample.sample # doctest: +SKIP
                
                self.annealSamples = decoded_samples
            
                fin = time.time()
                tiempo=fin-inicio
                
                self.annealDicc[i]=ind_var
                self.annealDicc[i]['valor']=best_sample.energy
                self.annealDicc[i]['tiempo']=tiempo
            
            self.annealBan=True
            #print("AnnealBan=",self.annealBan)
            
            #---RETURN TO THE MAIN FOLDER---
            os.chdir(self.maindir) #Return to the original execution folder
            
        except:
            
            os.chdir(self.maindir) #Return to the main execution folder
            
            print("Se presentó un error durante la ejecución del método \n Puede revisar el informe a continuación:\n")
            var = traceback.format_exc()
            print(var)             ### Return to the original execution folder
            
    
         
         
         