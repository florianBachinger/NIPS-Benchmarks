"""
  Copied April 2023 from https://github.com/omron-sinicx/srsd-benchmark under MIT licence
  Credit Matsubara et al.:
  @article{matsubara2022rethinking,
    title={Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery},
    author={Matsubara, Yoshitomo and Chiba, Naoya and Igarashi, Ryo and Tatsunori, Taniai and Ushiku, Yoshitaka},
    journal={arXiv preprint arXiv:2206.10540},
    year={2022}
  }
"""

import warnings

import numpy as np
import pandas as pd
import sympy
from sympy import Derivative, Matrix, Symbol, simplify, solve, lambdify
import SCRBenchmark.Constants.StringKeys as sk

from SCRBenchmark.Data.feynman_srsd_info import SRSD_EQUATION_CONFIG_DICT as SRSDConfig


FLOAT32_MAX = np.finfo(np.float32).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_TINY = np.finfo(np.float32).tiny


FLOAT64_MAX = np.finfo(np.float64).max
FLOAT64_MIN = np.finfo(np.float64).min
FLOAT64_TINY = np.finfo(np.float64).tiny



def create_dataset_from_sampling_objectives(sampling_objs, sympy_eq,eq_func,check_if_valid, sample_size, patience=10, ):
    warnings.filterwarnings('ignore')
    assert len(sampling_objs) > 0, f'There should be at least one variable provided in `{sympy_eq}`'
    xs = [sampling_func(sample_size) for sampling_func in sampling_objs]
    y = eq_func(xs)
    # Check if y contains NaN, Infinity, etc
    valid_sample_flags = check_if_valid(y)
    valid_sample_size = sum(valid_sample_flags)
    if valid_sample_size == sample_size:
        return np.array([*xs, y]).T

    valid_xs = [x[valid_sample_flags] for x in xs]
    valid_y = y[valid_sample_flags]
    missed_sample_size = sample_size - valid_sample_size
    for i in range(patience):
        # print(f'patience {i}/{patience} remaining size {missed_sample_size}')
        xs = [sampling_func(missed_sample_size * 5) for sampling_func in sampling_objs]
        y = eq_func(xs)
        valid_sample_flags = check_if_valid(y)
        valid_xs = [np.concatenate([xs[i][valid_sample_flags], valid_xs[i]]) for i in range(len(xs))]
        valid_y = np.concatenate([y[valid_sample_flags], valid_y])
        valid_sample_size = len(valid_y)
        if valid_sample_size >= sample_size:
            xs = [x[:sample_size] for x in valid_xs]
            y = valid_y[:sample_size]
            return np.array([*xs, y]).T
    raise TimeoutError(f'number of valid samples (`{len(valid_y)}`) did not reach to '
                        f'{sample_size} within {patience} trials')


def get_constraint_descriptor( eq, local_dict, xs):
    f = sympy.lambdify(local_dict, eq,"numpy")
    #calculate gradient per data point
    # gradients = np.array([ f(*row) for row in xs ])
    # speedup of 5:
    f_v = np.vectorize(f)
    gradients = f_v(*(xs.T))
    return get_constraint_descriptor_for_gradients(gradients)

def get_constraint_descriptor_for_gradients( gradients):
    unique_gradient_signs = set(np.unique(np.sign(gradients)))
    if((unique_gradient_signs ==  set([-1])) or (unique_gradient_signs ==  set([-1, 0]))):
      descriptor = sk.EQUATION_CONSTRAINTS_DESCRIPTOR_NEGATIVE
    elif ((unique_gradient_signs ==  set([1])) or (unique_gradient_signs ==  set([0, 1]))):
        descriptor = sk.EQUATION_CONSTRAINTS_DESCRIPTOR_POSITIVE
    elif ((unique_gradient_signs ==  set([-1, 1])) or (unique_gradient_signs ==  set([-1, 0, 1]))):
        descriptor = sk.EQUATION_CONSTRAINTS_DESCRIPTOR_NO_CONSTRAINT
    elif (unique_gradient_signs ==  set([0])):
        descriptor = sk.EQUATION_CONSTRAINTS_DESCRIPTOR_ZERO
    else:
      raise "Unforseen sign values!"
    return descriptor

class KnownEquation(object):
    _eq_name = None

    def __init__(self, num_vars, sampling_objs, kwargs_list=None):
        super().__init__()
        if kwargs_list is None:
            kwargs_list = [{'real': True} for _ in range(num_vars)]

        assert len(sampling_objs) == num_vars
        assert len(kwargs_list) == num_vars
        self.sampling_objs = sampling_objs
        self.x = [Symbol(f'x{i}', **kwargs) for i, kwargs in enumerate(kwargs_list)]
        self.sympy_eq = None

    def calculate(self,xs):
        return self.eq_func(xs)

    def get_eq_name(self, prefix=None, suffix=None):
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        return prefix + self._eq_name + suffix

    def get_var_count(self):
        return len(self.x)

    def get_op_count(self):
        return self.sympy_eq.count_ops()

    def check_num_vars_consistency(self, debug=False):
        num_vars = self.get_var_count()
        num_vars_used = len(self.sympy_eq.atoms(Symbol))
        consistent = num_vars == num_vars_used
        if debug and not consistent:
            print(f'\tnumber of variables (`{num_vars}`) is not consistent with '
                  f'number of those used in sympy_eq (`{num_vars_used}`)')
        return consistent

    def get_domain_range(self):
        min_value = None
        max_value = None
        for sampling_objs in self.sampling_objs:
            sub_min_value = sampling_objs.min_value
            sub_max_value = sampling_objs.max_value
            if min_value is None:
                min_value = sub_min_value
                max_value = sub_max_value
            elif sub_min_value < min_value:
                min_value = sub_min_value
            elif sub_max_value > max_value:
                max_value = sub_max_value
        return np.abs(np.log10(np.abs(max_value - min_value)))

    def get_domain_ranges(self):
        return [{"name":var.name, 
                "low":sampling_objs.get_value_range()[0],
                "high":sampling_objs.get_value_range()[1]}
                
                for (var,sampling_objs) 
                in zip(self.get_vars(), self.sampling_objs) ]

    def eq_func(self, x):
        raise NotImplementedError()

    def check_if_valid(self, values):
        return ~np.isnan(values) * ~np.isinf(values) * \
               (FLOAT64_MIN <= values) * (values <= FLOAT64_MAX) * (np.abs(values) >= FLOAT64_TINY)

    def create_dataset(self, sample_size, patience=10):
        return create_dataset_from_sampling_objectives(self.sampling_objs, self.sympy_eq, self.eq_func, self.check_if_valid, sample_size,patience)

    def find_stationary_points(self, excludes_saddle_points=False):
        if self.sympy_eq is None:
            raise ValueError('`sympy_eq` is None and should be initialized with sympy object')

        # 1st-order partial derivative
        f_primes = [Derivative(self.sympy_eq, var).doit() for var in self.x]

        # Find stationary points
        try:
            stationary_points = solve(f_primes, self.x)
            stationary_points = [sp for sp in map(lambda sp: simplify(sp), stationary_points)
                                 if isinstance(sp, sympy.core.containers.Tuple) and all([s.is_real for s in sp])]
            if len(stationary_points) == 0 or not excludes_saddle_points:
                return stationary_points
        except Exception as e:
            print(f'====={e}=====')
            return []

        # 2nd-order partial derivative
        f_prime_mat = [[Derivative(f_prime, var).doit() for var in self.x] for f_prime in f_primes]

        # Hesse matrix
        hesse_mat = Matrix(f_prime_mat)
        det_hessian = hesse_mat.det()

        # Find saddle points
        saddle_point_list = list()
        diff_stationary_point_list = list()
        for sp in stationary_points:
            pairs = [(var, sp_value) for var, sp_value in zip(self.x, sp)]
            sign_det_hessian = det_hessian.subs(pairs).evalf()
            if sign_det_hessian < 0:
                saddle_point_list.append(sp)
            else:
                diff_stationary_point_list.append(sp)
        return diff_stationary_point_list

    @classmethod
    def from_sympy_eq(cls, sympy_eq, sampling_objs, reindexes=True):
        warnings.filterwarnings('ignore')
        variables = tuple(sympy_eq.free_symbols)
        if reindexes:
            new_variables = tuple([Symbol(f'x{i}') for i in range(len(variables))])
            for old_variable, new_variable in zip(variables, new_variables):
                sympy_eq = sympy_eq.subs(old_variable, new_variable)
            variables = new_variables

        assert len(sampling_objs) == len(variables)
        ds = cls(len(variables), sampling_objs)
        ds.sympy_eq = sympy_eq
        eq_func = lambdify(variables, sympy_eq, modules='numpy')
        ds.eq_func = lambda x: eq_func(*x).T
        return ds
    
    def to_dataframe(self, data,use_display_name = False):
        if(use_display_name):
          return pd.DataFrame(data, columns= self.get_var_names() + [self.get_output_name()])
        else:
          return pd.DataFrame(data, columns= self.get_vars() + [self.get_output_name()])
    
    def create_dataframe(self, sample_size, patience=10 ):
        data = self.create_dataset(sample_size, patience)
        return self.to_dataframe(data)
    
    def create_input_dataset (self, sample_size, patience=10):
      dataset = self.create_dataset(sample_size, patience)
      return dataset[:,:-1]
    
    def get_inputs_from_dataset (self, dataset = None):
      if(dataset is None):
          dataset = self.create_dataset()
      return dataset[:,:-1]
    
    def get_eq_name (self):
      return self.__class__.__name__
    
    def get_eq_source (self):
      return self._eq_source
    
    def get_eq_raw (self):
      if(self.get_eq_source() == sk.SRSDF_SOURCE_QUALIFIER):
          return SRSDConfig[self.get_eq_name()][sk.EQUATION_CONFIG_DICT_RAW_EXPRESSION_KEY]
      raise "no equation source specified, or equation is not supported"
    
    def get_sympy_eq_local_dict (self):
      return { v.name:v for v in self.x}
    
    def get_vars (self):
      return self.x
    
    def get_var_names (self):
      if(self.get_eq_source() == sk.SRSDF_SOURCE_QUALIFIER):
          return list(SRSDConfig[self.get_eq_name()][sk.EQUATION_CONFIG_DICT_VARIABLE_KEY])
      raise "no equation source specified, or equation is not supported"
    
    def get_output_name (self):
      if(self.get_eq_source() == sk.SRSDF_SOURCE_QUALIFIER):
          return SRSDConfig[self.get_eq_name()][sk.EQUATION_CONFIG_DICT_OUTPUT_KEY]
      raise "no equation source specified, or equation is not supported"
    


          





    