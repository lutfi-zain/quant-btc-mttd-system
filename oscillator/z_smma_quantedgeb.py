import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indicators_helper import *

def z_smma_quantedgeb(
    df: pd.DataFrame,
    lu: float = 0.1,
    su: float = -0.1,
    len_smma: int = 12,
    len_z: int = 30
) -> pd.DataFrame:
    src = df['close']
    smma = rma(src, len_smma)
    mean_val = ema(smma, len_z)
    sd_val = stdev(smma, len_z)
    smma_z = (smma - mean_val) / sd_val
    
    long_c = smma_z > lu
    short_c = smma_z < su
    
    qb = 0
    qb_vals = np.zeros(len(df))
    for i in range(len(df)):
        if long_c.iloc[i] and not short_c.iloc[i]:
            qb = 1
        elif short_c.iloc[i]:
            qb = -1
        qb_vals[i] = qb
        
    return pd.DataFrame({
        'smma_z': smma_z,
        'qb': qb_vals
    }, index=df.index)
