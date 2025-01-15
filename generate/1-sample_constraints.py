import sympy
import copy

import SCRBenchmark.SRSDFeynman as srsdf
import SCRBenchmark.Constants.StringKeys as sk
from SCRBenchmark import create_dataset_from_sampling_objectives, get_constraint_descriptor
from SCRBenchmark import Benchmark

CONSTRAINT_SAMPLING_SIZE = 1_000_000

def CalculateEquation(equation_class, text_file):
  print(f"{equation_class}")
  
  scr = Benchmark(equation_class, initialize_constraint_checking_datasets=False)

  f_primes = [(sympy.Derivative(scr.equation.sympy_eq, var).doit(),var.name, var_display_name, 1) 
                 for (var,var_display_name) 
                 in list(zip(scr.equation.x, scr.equation.get_var_names()))]
  
  f_seconds = [(sympy.Derivative(scr.equation.sympy_eq, var).doit(),var.name, var_display_name, 2) 
                 for (var,var_display_name) 
                 in list(zip(scr.equation.x, scr.equation.get_var_names()))]
  
  derviatives = f_primes+f_seconds

  split_objectives = []
  constraints = []

  objectives= list(
      zip(
      scr.equation.x,
      [obj.to_uniform_sampling() for obj in scr.equation.sampling_objs]
      ))

  split_objectives.append(objectives)

  for (var_name,objective ) in objectives:

    if(objective.uses_positive and objective.uses_negative):
      positive_copy = copy.deepcopy(split_objectives)
      negative_copy = copy.deepcopy(split_objectives)
      
      for objs in positive_copy:
        for (var, obj) in objs:
          if(var == var_name):
            obj.uses_negative = False
      for objs in negative_copy:
        for (var, obj) in objs:
          if(var == var_name):
            obj.uses_positive = False
      split_objectives = positive_copy + negative_copy

  constraint_id = 1 
  for split_objective in split_objectives:
    for (derivative, var_name, var_display_name, order_derivative) in derviatives:
      sampling_space = [{ 'name':var.name,
                         'low': obj.get_value_range()[0],
                         'high':obj.get_value_range()[1]} for (var,obj) in split_objective]

      sampling_objectives = [obj for (var, obj) in split_objective]
      xs = scr.equation.get_inputs_from_dataset(
              create_dataset_from_sampling_objectives(sampling_objectives, 
                                                      scr.equation.sympy_eq, 
                                                      scr.equation.eq_func, 
                                                      scr.equation.check_if_valid, 
                                                      CONSTRAINT_SAMPLING_SIZE,
                                                      patience = CONSTRAINT_SAMPLING_SIZE)
            )

      descriptor =  get_constraint_descriptor(derivative, scr.equation.x, xs)
      print(f'> partial derivative over ({var_name}) with space {sampling_space} with constraint ({descriptor})')
      if(descriptor != sk.EQUATION_CONSTRAINTS_DESCRIPTOR_NO_CONSTRAINT):
        constraints.append({
          sk.EQUATION_CONSTRAINTS_VAR_NAME_KEY: var_name,
          sk.EQUATION_CONSTRAINTS_VAR_DISPLAY_NAME_KEY:var_display_name,
          sk.EQUATION_CONSTRAINTS_ORDER_DERIVATIVE_KEY:order_derivative,
          sk.EQUATION_CONSTRAINTS_DESCRIPTOR_KEY: descriptor,
          sk.EQUATION_CONSTRAINTS_DERIVATIVE_KEY: str(derivative),
          sk.EQUATION_CONSTRAINTS_SAMPLE_SPACE_KEY: sampling_space,
          sk.EQUATION_CONSTRAINTS_ID_KEY: constraint_id  
          })
        constraint_id = constraint_id + 1

  text_file.write(str( {sk.EQUATION_EQUATION_NAME_KEY:scr.equation.get_eq_name(),
        sk.EQUATION_CONSTRAINTS_CONSTRAINTS_KEY: constraints
          }).replace('\'','\"'))
  text_file.write(',')


with open("SCRBenchmark/Data/feynman_srsdf_constraint_info.json", "w") as text_file:
  text_file.write('SRSD_EQUATION_CONSTRAINTS = [')
  #iterate over all equations
  for dictEntry in srsdf.AllEquations:
    CalculateEquation(srsdf.AllEquations[dictEntry], text_file)
    text_file.flush()
  text_file.write(']')