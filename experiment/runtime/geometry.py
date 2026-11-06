"""Origin-anchored rational grids with exact integer exclusion tests."""
from fractions import Fraction
import numpy as np

TX = (0.0, 0.0)
RECEIVERS = [(-20.0, -20.0, 4.0), (-20.0, 20.0, 4.0),
             (20.0, -20.0, 4.0), (20.0, 20.0, 4.0)]

def integer_grid(step=0.4):
    spacing = Fraction(str(step))
    limit = Fraction(20)/spacing
    if spacing<=0 or limit.denominator!=1:
        raise ValueError('The grid must divide the 20 um half width exactly.')
    p,q = spacing.numerator, spacing.denominator
    index = np.arange(-int(limit),int(limit)+1,dtype=np.int64)
    i,j = np.meshgrid(index,index,indexing='ij')
    x,y = p*i,p*j
    feasible = x*x+y*y > (4*q)**2
    for rx,ry in [(-20*q,-20*q),(-20*q,20*q),(20*q,-20*q),(20*q,20*q)]:
        feasible &= (x-rx)**2+(y-ry)**2 > (8*q)**2
    return np.column_stack((i[feasible],j[feasible]))

def make_grid(step):
    spacing = Fraction(str(step))
    indices = integer_grid(step)
    points = indices.astype(np.float64)*spacing.numerator/spacing.denominator
    return points,(indices[:,0]+indices[:,1])%5
