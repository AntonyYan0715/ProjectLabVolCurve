import os
import numpy as np
import pandas as pd
from functools import cache
import pandas_market_calendars as mcal # NYSE Calendar


@cache
def read_h5(data_path=os.path.join("..", "data"), col=None, start="1970-01-01", end="2038-01-19"):
    """Reads HDF5 files containing option data.
    data_path - directory of the data folder (containing the h5 files)
    col   - (optional) default None meaning no slicing for dates
    start - (optional) default includes data starting at the beginning of the series
    end   - (optional) default includes data through the end of the series non-inclusive
    """
    raw = pd.concat([pd.read_hdf(os.path.join(data_path, "vol1.h5"), key="onetick"), 
                    pd.read_hdf(os.path.join(data_path, "vol2.h5"), key="onetick")])
    
    if not(col): return raw
    else:        return raw[(start<=raw[col]) & (raw[col]<end)].reset_index(drop=True)

def add_attributes(df):
    df["DIFF"] = np.abs(df["STRIKE"] - df["UNDERLYING_PRICE"])
    df["RANK"] = df.groupby(["TS", "TYPE"])["DIFF"].rank(method="min") - 1

    holidays = mcal.get_calendar('NYSE').holidays()
    holidays = list(holidays.holidays) # NYSE Holiday
    df["BUS_DAYS"] = np.busday_count([d.date() for d in df["TS"]], 
                                      [d.date() for d in df["EXP"]], 
                                      holidays=holidays)
    df["CAL_DAYS"] = np.busday_count([d.date() for d in df["TS"]], 
                                      [d.date() for d in df["EXP"]], 
                                      weekmask="1111111")
    return df