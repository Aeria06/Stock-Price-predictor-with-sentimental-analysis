import sys, os
sys.path.insert(0, r'C:\Users\Hitanshi Arora\Downloads\stock-safe-pro')
from src.models import RFReg
import numpy as np

class D:
    pass

r = RFReg()
d = D()
d.feature_importances_ = np.array([0.123, 0.877])
r.m = d
cols = ['feat1', ('a','b')]
imp = r.feature_importances(cols)
print('index types:', [type(x) for x in imp.index])
print('index values:', imp.index.tolist())
print('importances:', imp.tolist())
