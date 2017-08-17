#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:02:48 2017

@author: gfleishman
"""

# valid inputs
models = ['elastic',
          'fluid',
          'lddmm',
          'syn',
          'gsid',
          'sigsid',
          'grid']

optimizers = ['static',
              'secant',
              'bb']

matchers = ['ssd',
            'gcc',
            'lcc',
            'mi']

regularizers = ['gaussian',
                'differential']

diffeo_models = models[2:]
