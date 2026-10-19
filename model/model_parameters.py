#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fitted rate constants for the three-state receptor model."""

DEFAULT_PARAMS = {
    'R_t': 1,    # Total receptor concentration
    'kon': 1.1670e+06 ,     # Association rate constant
    'koff': 7.1525e-02,   # Dissociation rate constant
    
    'kdes': 1.1845e-03,  # Desensitization rate constant
    'kres': 5.4944e-04,  # Resensitization rate constant 
    
    'ksyn_cAMP': 2e-8,
    'kdeg_cAMP': 3e-2,
    
    'ksyn_R': 9.6e-6,
    'kdeg_mem': 9.6e-6,
    'kdeg_des': 6.4e-5
}


DEFAULT_PARAMS_HEK = {
    'R_t': 1,    # Total receptor concentration
    'kon': 1.1670e+06,     # Association rate constant
    'koff': 7.1525e-02,   # Dissociation rate constant
    
    'kdes': 2.3e-03,  # Desensitization rate constant
    'kres': 2.6e-04,  # Resensitization rate constant 
    
    'ksyn_cAMP': 2e-8,
    'kdeg_cAMP': 3e-2,
    
    'ksyn_R': 9.6e-6,
    'kdeg_mem': 9.6e-6,
    'kdeg_des': 6.4e-5
}


def get_params(**kwargs):
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    return params