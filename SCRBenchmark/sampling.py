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

import numpy as np

from .registry import get_sampling_obj, register_sampling_class, register_sampling_func


@register_sampling_func
def default_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    # x ~ either U(0.1, 10.0) or U(-10.0, -0.1) with 50% chance
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    pos_samples = 10.0 ** np.random.uniform(log10_min, log10_max, size=num_positives)
    neg_samples = -10.0 ** np.random.uniform(log10_min, log10_max, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


@register_sampling_func
def default_positive_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    # x ~ U(0.1, 10.0)
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    return 10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)


@register_sampling_func
def default_negative_sampling(sample_size, min_value=-1.0e1, max_value=-1.0e-1):
    # x ~ U(-10.0, -0.1)
    log10_min = -np.log10(abs(min_value))
    log10_max = -np.log10(abs(max_value))
    return -10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)


@register_sampling_func
def simple_sampling(sample_size, min_value=0.0, max_value=1.0):
    # x ~ either U(0.0, 1.0) or U(-1.0, 0.) with 50% chance
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    if num_negatives>2:
        neg_samples = -np.random.uniform(min_value, max_value, size=num_negatives-2)
        neg_samples = np.concat([neg_samples, [-min_value, -max_value]])
    else:
        neg_samples = -np.random.uniform(min_value, max_value, size=num_negatives)

    if num_positives>2:
        pos_samples = np.random.uniform(min_value, max_value, size=num_positives-2)
        pos_samples = np.concat([pos_samples, [min_value, max_value]])
    else:
        pos_samples = np.random.uniform(min_value, max_value, size=num_positives)
    
    all_samples = np.concat([neg_samples, pos_samples])
    np.random.shuffle(all_samples)
    return all_samples


@register_sampling_func
def simple_positive_sampling(sample_size, min_value=0.0, max_value=1.0):
    # x ~ U(0.0, 1.0)
    return np.random.uniform(min_value, max_value, size=sample_size)


@register_sampling_func
def simple_negative_sampling(sample_size, min_value=-1.0, max_value=0.0):
    # x ~ U(-1, 0.0)
    return -np.random.uniform(min_value, max_value, size=sample_size)


@register_sampling_func
def integer_sampling(sample_size, min_value=1, max_value=100):
    # x ~ either U(1, 100) or U(-100, -1) with 50% chance
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    pos_samples = np.random.randint(min_value, max_value, size=num_positives)
    neg_samples = -np.random.randint(min_value, max_value, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


@register_sampling_func
def integer_positive_sampling(sample_size, min_value=1, max_value=100):
    # x ~ U(1, 100)
    return np.random.randint(min_value, max_value, size=sample_size)


@register_sampling_func
def integer_negative_sampling(sample_size, min_value=-100, max_value=-1):
    # x ~ U(-100, -1)
    return -np.random.randint(min_value, max_value, size=sample_size)
    

@register_sampling_class
class DefaultSampling(object):
    def __init__(self, min_value, max_value, uses_positive=True, uses_negative=True):
        self.min_value = min_value
        self.max_value = max_value
        assert uses_positive or uses_negative
        self.uses_positive = uses_positive
        self.uses_negative = uses_negative
    
    def __call__(self, sample_size):
        if self.uses_positive and self.uses_negative:
            return default_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return default_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return default_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(f'Either self.uses_positive ({self.uses_positive}) or '
                             f'self.uses_negative({self.uses_negative}) must be True')
    
    def get_value_range(self):
        if self.uses_positive and self.uses_negative:
            return (-self.max_value, self.max_value)
        elif self.uses_positive:
            return (self.min_value, self.max_value)
        elif self.uses_negative:
            return (-self.max_value, -self.min_value)
        raise AttributeError(f'Either self.uses_positive ({self.uses_positive}) or '
                             f'self.uses_negative({self.uses_negative}) must be True')
    
    def to_uniform_sampling(self):
        return SimpleSampling(self.min_value,self.max_value,self.uses_positive,self.uses_negative)



@register_sampling_class
class SimpleSampling(object):
    def __init__(self, min_value, max_value, uses_positive=True, uses_negative=True):
        self.min_value = min_value
        self.max_value = max_value
        assert uses_positive or uses_negative
        self.uses_positive = uses_positive
        self.uses_negative = uses_negative
    
    def __call__(self, sample_size):
        if self.uses_positive and self.uses_negative:
            return simple_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return simple_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return simple_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(f'Either self.uses_positive ({self.uses_positive}) or '
                             f'self.uses_negative({self.uses_negative}) must be True')

    def get_value_range(self):
        if self.uses_positive and self.uses_negative:
            return (-self.max_value, self.max_value)
        elif self.uses_positive:
            return (self.min_value, self.max_value)
        elif self.uses_negative:
            return (-self.max_value, -self.min_value)
        raise AttributeError(f'Either self.uses_positive ({self.uses_positive}) or '
                             f'self.uses_negative({self.uses_negative}) must be True')
    
    def to_uniform_sampling(self):
        return self

@register_sampling_class
class IntegerSampling(object):
    def __init__(self, min_value, max_value, uses_positive=True, uses_negative=True):
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        assert uses_positive or uses_negative
        self.uses_positive = uses_positive
        self.uses_negative = uses_negative
    
    def __call__(self, sample_size):
        if self.uses_positive and self.uses_negative:
            return integer_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return integer_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return integer_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(f'Either self.uses_positive ({self.uses_positive}) or '
                             f'self.uses_negative({self.uses_negative}) must be True')

    def get_value_range(self):
        if self.uses_positive and self.uses_negative:
            return (-self.max_value, self.max_value)
        elif self.uses_positive:
            return (self.min_value, self.max_value)
        elif self.uses_negative:
            return (-self.max_value, -self.min_value)
        raise AttributeError(f'Either self.uses_positive ({self.uses_positive}) or '
                             f'self.uses_negative({self.uses_negative}) must be True')

    def to_uniform_sampling(self):
        return self

def build_sampling_objs(sampling_obj_configs):
    sampling_obj_list = list()
    for sampling_obj_config in sampling_obj_configs:
        sampling_type = sampling_obj_config['type']
        sampling_kwargs = sampling_obj_config.get('kwargs', dict())
        sampling_obj = get_sampling_obj(sampling_type, **sampling_kwargs)
        sampling_obj_list.append(sampling_obj)
    return sampling_obj_list


def to_string(SamplingObj):
    if SamplingObj.uses_positive and SamplingObj.uses_negative:
        return f"[{-SamplingObj.max_value}, {-SamplingObj.min_value}] ∪ [{SamplingObj.min_value}, {SamplingObj.max_value}] ⊆ R"
    elif SamplingObj.uses_positive:
        return f"[{SamplingObj.min_value}, {SamplingObj.max_value}] ⊆ R"
    elif SamplingObj.uses_negative:
        return f"[{-SamplingObj.max_value}, {-SamplingObj.min_value}] ⊆ R"
    raise AttributeError(f'Either SamplingObj.uses_positive ({SamplingObj.uses_positive}) or '
                          f'SamplingObj.uses_negative({SamplingObj.uses_negative}) must be True')

def to_latex_string(SamplingObj):
    if SamplingObj.uses_positive and SamplingObj.uses_negative:
        return f"\mathbb{{R}}^{{\leq {-SamplingObj.min_value}}}_{{\geq {-SamplingObj.max_value}}} \cup \mathbb{{R}}^{{\leq {SamplingObj.max_value}}}_{{\geq {SamplingObj.min_value}}}"
    elif SamplingObj.uses_positive:
        return f"\mathbb{{R}}^{{\leq {SamplingObj.max_value}}}_{{\geq {SamplingObj.min_value}}}"
    elif SamplingObj.uses_negative:
        return f"\mathbb{{R}}^{{\leq {-SamplingObj.min_value}}}_{{\geq {-SamplingObj.max_value}}}"
    raise AttributeError(f'Either SamplingObj.uses_positive ({SamplingObj.uses_positive}) or '
                          f'SamplingObj.uses_negative({SamplingObj.uses_negative}) must be True')