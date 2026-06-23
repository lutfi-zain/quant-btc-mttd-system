import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import indicators_helper
sys.path.append(str(Path(__file__).resolve().parents[1]))
from indicators_helper import *

def irs_elder_force_volume_index(df: pd.DataFrame,
                                 length: int = 40) -> pd.DataFrame:
    
    n = len(df)
    
    hl2 = (df['high'] + df['low']) / 2.0
    hl2_change = hl2 - hl2.shift(1)
    
    # efi = ta.ema(ta.change(hl2) * volume, length)
    efi_input = hl2_change * df['volume']
    efi_val = ema(efi_input, length)
    
    # Stateful trend vii
    vii = np.zeros(n)
    curr_vii = 0.0
    
    efi_vals = efi_val.values
    
    for i in range(n):
        if i == 0 or np.isnan(efi_vals[i]) or np.isnan(efi_vals[i-1]):
            vii[i] = 0.0
            continue
            
        L = (efi_vals[i] > 0.0) and (efi_vals[i-1] <= 0.0)
        S = (efi_vals[i] < 0.0) and (efi_vals[i-1] >= 0.0)
        
        if L and not S:
            curr_vii = 1.0
        elif S:
            curr_vii = -1.0
            
        vii[i] = curr_vii
        
    hma_val = hma(df['close'], 45)
    
    results = pd.DataFrame(index=df.index)
    results['hma'] = hma_val
    results['efi'] = efi_val
    results['vii'] = vii
    
    return results
