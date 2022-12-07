import numpy as np

def unwrap(t, delta):
    """ Unwrapping "classique" d'un signal 1D """
    ns=t.shape[-1]
    tt=t.copy()
    dt=tt[...,1:]-tt[...,:-1]
    # Quantification de la dérivée par pas de delta
    udt= np.round(dt/delta).cumsum(-1)*delta
    tt[...,1:] -= udt
    return tt

def sync( t2, t1 ):
    """ 
    Search a linear drift of t2 from t1
    We search a and b such that t2 = a*t1+b
    """

    # Association des deux horloges par moindres carrés
    # ##################################################
    n = t2.shape[0]

    t1a = (t1 ** 2).sum()
    t1b = t1.sum()
    t2a = (t1 * t2).sum()
    t2b = t2.sum()

    d = n * t1a - t1b * t1b
    a = (n * t2a - t1b * t2b) / d
    b = (t1a * t2b - t1b * t2a)  /d
    return a, b

