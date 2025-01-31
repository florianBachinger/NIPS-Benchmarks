import SCRBenchmark.base as base
import sympy
import numpy as np
import pandas as pd
import warnings
import os
import jax
import SCRBenchmark.Constants.StringKeys as sk
from SCRBenchmark.Data.feynman_srsdf_constraint_info import SRSD_EQUATION_CONSTRAINTS as SRSDFConstraints
CONSTRAINT_SAMPLING_SIZE = 100_000

class Benchmark(object):
    _eq_name = None
    

    def __init__(self, equation, initialize_constraint_checking_datasets = True):
        super().__init__()
        assert issubclass(equation ,base.KnownEquation)

        self.equation = equation()
        self.constraints = self.get_constraints()
        self.datasets = None
        if(initialize_constraint_checking_datasets):
          self.read_datasets_for_constraint_checking()

    def read_datasets_for_constraint_checking(self):
      constraints = [c for c in self.constraints if c[sk.EQUATION_CONSTRAINTS_DESCRIPTOR_KEY]!=sk.EQUATION_CONSTRAINTS_DESCRIPTOR_NO_CONSTRAINT]
      if(len(constraints) == 0):
          warnings.warn( f"equation {self.equation._eq_name} has to have constraints to be checked. all checks will return true.")
      self.datasets = {}
      for constraint in constraints:
          sample_space = constraint[sk.EQUATION_CONSTRAINTS_SAMPLE_SPACE_KEY]
          lows = [ space['low'] for space in sample_space]
          highs = [ space['high'] for space in sample_space]

          self.datasets[constraint[sk.EQUATION_CONSTRAINTS_ID_KEY]] = np.random.uniform(lows,highs,(CONSTRAINT_SAMPLING_SIZE,self.equation.get_var_count()))

    def read_test_dataframe(self):
        file = os.path.join(os.path.dirname(__file__),f'Data/Test/{self.equation.get_eq_name()}.csv')
        return pd.read_csv(file)
    
    def create_dataset(self, sample_size,  noise_level = 0, seed = None, patience = 10 ):
        assert (0<=noise_level and noise_level<=1), f'noise_level must be in [0,1]'

        if(not (seed is None)):
          np.random.seed(seed)

        xs = self.equation.create_dataset(sample_size,patience)

        if(noise_level>0):
          std_dev = np.std(xs[:,-1])
          xs[:,-1] = xs[:,-1] + np.random.normal(0,std_dev*np.sqrt(noise_level),len(xs))

        return (xs, self.read_test_dataframe().to_numpy())
    
    def create_dataframe(self,sample_size, noise_level = 0, seed = None, patience = 10, use_display_name = False ):
       (train, test) = self.create_dataset(sample_size=sample_size,
                                           noise_level=noise_level,
                                           seed= seed,
                                           patience = patience)
       train_df = self.equation.to_dataframe(train,use_display_name)
       test_df = self.equation.to_dataframe(test,use_display_name)
       return (train_df,test_df)

    def get_constraints (self):
      if(self.equation.get_eq_source() == sk.SRSDF_SOURCE_QUALIFIER):
          return next(x[sk.EQUATION_CONSTRAINTS_CONSTRAINTS_KEY] for x in SRSDFConstraints if x[sk.EQUATION_EQUATION_NAME_KEY] == self.equation.get_eq_name())
          
    def check_constraints (self, f, Library = "SymPy", use_display_names = False):
      if(Library == "SymPy"):
        return self.check_constraints_SymPy (f, use_display_names)
      elif(Library == "JAX"):
        return self.check_constraints_JAX (f, use_display_names)
      else:
        raise RuntimeError(f"Specified library '{Library}' is not supported.")
       
    def check_constraints_SymPy (self, f, use_display_names = False):
      constraints = self.get_constraints()

      constraints = [c for c in constraints if c[sk.EQUATION_CONSTRAINTS_DESCRIPTOR_KEY]!=sk.EQUATION_CONSTRAINTS_DESCRIPTOR_NO_CONSTRAINT]
      if(len(constraints) == 0):
          return (True, []) #no constraints to check
      
      if(self.datasets is None):
          self.read_datasets_for_constraint_checking()

      # replace the sympy local dictionary with the display names of variables if specified
      local_dict = self.equation.get_sympy_eq_local_dict()
      if(use_display_names):
          local_dict = { c : sympy.Symbol(c) for c in self.equation.get_var_names()}

      # parse the provided candidate expression
      # will use display names if specified
      expr = sympy.parse_expr(f, evaluate=False, local_dict= local_dict)

      #calculate all first order partial derivatives of the expression 
      f_primes = [(sympy.Derivative(expr, var).doit(),var.name, 1) 
                 for var
                 in local_dict.values()]
      
      #calculate all second order partial derivatives of the expression (every possible combination [Hessian])
      f_prime_mat = [[ (sympy.Derivative(f_prime, var).doit(), [prime_var_name,var.name], 2 ) 
                        for var
                        in local_dict.values()] 
                     for (f_prime, prime_var_name, _) 
                     in f_primes]
      
      #flatten 2d Hessian to 1d list and combine them 
      f_prime_mat_flattened = [item for sublist in f_prime_mat for item in sublist]
      derviatives = f_primes+f_prime_mat_flattened
      
      
      violated_constraints = []
      #check for all existing constraints if they are met
      for constraint in constraints:
        #every constraint has a specific input range in which they apply
        xs = self.datasets[constraint[sk.EQUATION_CONSTRAINTS_ID_KEY]]

        #the current constraint to be checked matches only one of derivatives (all possible combinations are derived)
        if(use_display_names):
          matches = [ derivative for (derivative, var, _) in derviatives if var == constraint[sk.EQUATION_CONSTRAINTS_VAR_DISPLAY_NAME_KEY]]
        else:
          matches = [ derivative for (derivative, var, _) in derviatives if var == constraint[sk.EQUATION_CONSTRAINTS_VAR_NAME_KEY]]

        if(len(matches)>1):
           raise "derivatives are not names uniquely"
        if(len(matches)==0):
           raise "derivative not available"
        derivative = matches[0]

        #does the calculated (sampled) gradient for the current derivative match the constraint description
        descriptor = base.get_constraint_descriptor(derivative, local_dict.keys(), xs)
        if(descriptor != constraint[sk.EQUATION_CONSTRAINTS_DESCRIPTOR_KEY]):
            violated_constraints.append(constraint)

      return (len(violated_constraints) == 0, violated_constraints)
    
    def check_constraints_JAX (self, f, use_display_names = False):
      constraints = self.get_constraints()

      constraints = [c for c in constraints if c[sk.EQUATION_CONSTRAINTS_DESCRIPTOR_KEY]!=sk.EQUATION_CONSTRAINTS_DESCRIPTOR_NO_CONSTRAINT]
      if(len(constraints) == 0):
          return (True, []) #no constraints to check
      
      if(self.datasets is None):
          self.read_datasets_for_constraint_checking()

      # replace the sympy local dictionary with the display names of variables if specified
      var_names = [v.name for v in self.equation.get_vars()]

      g = jax.jit(jax.grad(f))
      hessian = jax.jit(jax.hessian(f))
      
      violated_constraints = []
      #check for all existing constraints if they are met
      for constraint in constraints:
        #every constraint has a specific input range in which they apply
        xs = self.datasets[constraint[sk.EQUATION_CONSTRAINTS_ID_KEY]]

        var_name_constraint = constraint[sk.EQUATION_CONSTRAINTS_VAR_NAME_KEY]
        descriptor = sk.EQUATION_CONSTRAINTS_DESCRIPTOR_UNKOWN_CONSTRAINT

        # checking the different types of constraints supported
        if(constraint[sk.EQUATION_CONSTRAINTS_ORDER_DERIVATIVE_KEY] == 1):
          #constraint is defined for the first order derivative
          # the signs of the functions gradient are to be checked for the input domain
          var_index = var_names.index(var_name_constraint)
          gradients = y = jax.vmap(g)(xs) 
          var_gradients = gradients[:,var_index]
          descriptor = base.get_constraint_descriptor_for_gradients(var_gradients)

        elif(constraint[sk.EQUATION_CONSTRAINTS_ORDER_DERIVATIVE_KEY] == 2):
          var1_index = var_names.index(var_name_constraint[0])
          var2_index = var_names.index(var_name_constraint[1])
          hessian_gradients = jax.vmap(hessian)(xs) 
          var_gradients = hessian_gradients[:,var1_index,var2_index]
          descriptor = base.get_constraint_descriptor_for_gradients(var_gradients)

        else:
          raise "constraint was available but it was not handled/checked"


        if(descriptor != constraint[sk.EQUATION_CONSTRAINTS_DESCRIPTOR_KEY]):
            violated_constraints.append(constraint)

      return (len(violated_constraints) == 0, violated_constraints)
