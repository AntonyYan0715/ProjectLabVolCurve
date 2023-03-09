import os, sys, time, pickle
sys.path.append('..') # Parent directory in path 
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 4)
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pylab as pl
import statsmodels.api as sm

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (16,10)
plt.show()
plt.close()
plt.rcParams["figure.figsize"] = (16,10)
from concurrent.futures import ProcessPoolExecutor # Concurrency for data processing



MODEL      = "hyperbola"
NCPU       = 4
PLOT_PATH  = "curve_charts"
FIT_BOUNDS = ((1e-6, 0, -1, -4, -3), (2, 3, 1, 4, 3))
X_BOUNDS   = np.linspace(-0.2,0.2,101)

class Model:

    def __init__(self):
        """Instantiate fitted parameters as None"""
        self.params = None

    def function(self, x, *args) -> np.ndarray:
        """The function the model represents"""
        pass

    def function_jac(self, x, *args) -> np.ndarray:
        """The Jacobian matrix used to speed up fitting"""
        pass
    
    def fit(self, x, y) -> None:
        """Fits the model on the x, y data and updates params"""
        self.params = []
    
    def visualize(self, ax, X, params):
        """Do some plotting stuff with the fitted model"""
        return ax

class HyperbolaModel(Model):
    
    def __init__(self, bounds: tuple):
        super().__init__()
        self.bounds = bounds
    
    def function(self, x, a, b, c, d, e):
        """Tiltable hyperbola for scipy.optimize.curve_fit
        Asymptotes at 
        y=(-b/a+d)(x-c)+e
        y=( b/a+d)(x-c)+e"""
        return (b**2*(1+(x-c)**2/a**2))**0.5+d*(x-c)+e
    
    def function_jac(self, x, a, b, c, d, e) -> np.ndarray:
        """Jacobian of the hyperbola with respect to each parameter"""
        ru = np.sqrt(b**2*(1+(x-c)**2/a**2))
        da = -(b**2 * (x-c)**2) / (a**3 * ru)
        db = np.sqrt(1+(x-c)**2/a**2)
        dc = -(b**2 * (x-c)) / (a**2 * ru) - d
        dd = x-c
        de = np.ones(len(x))
        return np.array([da, db, dc, dd, de]).T
    
    def function_minimizer(self, a, b, c, d, e):
        """Returns the minimizer (x such that y is the minimum) of a hyperbola"""
        return ((a**2 * c * d**2 + d*(a**4 * b**2 - a**6 * d**2)**0.5 - b**2 * c)
                /(a**2 * d**2 - b**2))
    
    def fit(self, x, y, p0=None) -> np.ndarray:
        """Fits a hyperbola for a day's IV single curve
        Using hyperbola and scipy.curve_fit()"""
        popt, pcov = curve_fit(self.function, x, y, p0=p0, bounds=self.bounds, 
                               jac=self.function_jac, max_nfev=100*25)
        self.params = popt
    
    def visualize(self, ax, X, params):
        """Draws the hyperbola according to params with relevant details
        including asymptotes and minima"""
        a, b, c, d, e = params

        slpos = (b/a +d)*(X-c)+e
        slneg = (-b/a+d)*(X-c)+e
        foci  = (b**2 + a**2)**0.5 + e
        locl  = self.function_minimizer(a, b, c, d, e)
        lowest= self.function(locl, a, b, c, d, e)
        skew  = d*(X-c) + b + e

        ax.plot(X, self.function(X, *params), label="Predicted IV Curve")

        # ax.plot(locl, foci, "go", alpha=0.3, label="Focal Point")

        ax.plot(X, slpos, color='g', alpha=0.3, label="Asymptote")
        ax.plot(X, slneg, color='g', alpha=0.3)

        ax.plot(X, skew, color='b', alpha=0.3, linestyle="--", label=f"Skew={d:.3f}")

        ax.axvline(locl, color='g', alpha=0.3, linestyle="--", 
                   label=f"X of Lowest Point={locl:.3f}")
        ax.axhline(lowest, color='g', alpha=0.3, linestyle="--", 
                   label=f"Y of Lowest Point={lowest:.3f}")

        return ax

class SingleExp:
    
    def __init__(self, exp, ivslice: pd.DataFrame, model: Model, xcol: str, ycol: str, xbounds: np.ndarray) -> None:
        self.exp     = exp
        self.ivslice = ivslice
        self.dayu    = self.ivslice["TS"].unique()
        self.model   = model

        self.xcol = xcol
        self.ycol = ycol

        self.xbounds = xbounds

        self.hyperbola_med_chg = pd.read_pickle("hyperbola_med_chg.pkl")
        
    def ts_fit(self):
        """Iterates through the days and builds frame of fitted parameters"""
        p0   = None
        fits = []
        for d in self.dayu:
            ivday  = self.ivslice[self.ivslice["TS"]==d]
            try:
                self.model.fit(ivday[self.xcol], ivday[self.ycol], p0=p0)
                p0 = params = self.model.params
            except RuntimeError: # Optimal parameters not found
                params = np.array([0,0,0,0,0])
            fits.append(params)

        self.fits = np.array(fits)
    
    def predict(self, indexer, method: str):
        """Uses data up to indexer to predict values for indexer+1"""
        if method.lower()=="naive":
            # Naive implementation. Assume today's model fits tomorrow's curve.
            return self.fits[indexer,:]
        elif method.lower()=="hyperbola_med_chg":
            p  = self.fits[indexer,:]
            dp = self.hyperbola_med_chg.iloc[indexer,:].values
            return p * (1 + dp)
        elif method.lower()=="last_change":
            p  = self.fits[indexer,:]
            dp = p / self.fits[indexer-1,:]
            return p * (dp)

    def loss(self, pred, actl, method: str):
        """Computes the loss function when comparing predictions versus actual"""
        if method.lower()=="mse":
            return np.mean((pred-actl)**2)
        else:
            raise ValueError("method must be mse")

    def ts_validate(self, start_indexer, method: str):
        """Iterates through the days and builds an array of loss based on the predict function"""
        losses = -np.ones(len(self.dayu)) # instantiate losses as -1; any valid loss will be >= 0

        # use i to predict i+1
        for i in range(start_indexer, len(self.dayu)-1):
            pred_params = self.predict(i, method)
            actl_params = self.fits[i+1,:]

            # Patch in case day failed to fit
            if (pred_params[0]==0) | (actl_params[0]==0):
                pass
            else:
                pred_y = self.model.function(self.xbounds, *pred_params)
                actl_y = self.model.function(self.xbounds, *actl_params)

                losses[i+1] = self.loss(pred_y, actl_y, "mse")

        self.losses = losses

    def plot_day(self, indexer: int, save=False):
        if type(indexer)==int:
            date = self.dayu[indexer]

        fig, ax = plt.subplots()
        ax = self.model.visualize(ax, self.xbounds, self.fits[indexer])

        to_plot = self.ivslice[self.ivslice["TS"]==date]

        ax.scatter(to_plot[self.xcol], to_plot[self.ycol], alpha=0.5, label="Empirical Data")

        ax.set_ylim([0, (to_plot[self.ycol].max())*1.1])
        ax.set_title(f"Hyperbolic Fit for EXP={str(self.exp)[:10]} DATE={str(date)[:10]}")
        ax.set_xlabel(self.xcol)
        ax.set_ylabel(self.ycol)
        ax.legend()

        if save: 
            plt.savefig(os.path.join(PLOT_PATH, f"{str(self.exp)[:10]}_{str(date)[:10]}.png"))
            plt.close()
        else: plt.show(block=True)

class AllExp:
    
    def __init__(self, ivts: pd.DataFrame, model: Model, xcol: str, ycol: str, xbounds: np.ndarray) -> None:
        self.ivts  = ivts
        self.expu  = self.ivts["EXP"].unique()
        self.model = model

        self.xcol = xcol
        self.ycol = ycol

        self.xbounds = xbounds
    
    def execute(self, exp) -> SingleExp:
        """Create a process for each expiry and fit respective curves"""
        print(f"EXP={str(exp)[:10]} Starting...")
        st = time.time()

        single = SingleExp(exp, self.ivts[self.ivts["EXP"]==exp], self.model, self.xcol, self.ycol, self.xbounds)
        single.ts_fit()
        single.ts_validate(1, "hyperbola_med_chg")

        el = time.time() - st
        print(f"EXP={str(exp)[:10]} Completed. Elapsed {str(dt.timedelta(seconds=el))[:-7]}")

        # single.plot_day(0)

        return single
        
    def by_exp(self):
        """Multiprocessed function to fit all the data we have"""

        print("Starting ProcessPoolExecutor")
        with ProcessPoolExecutor(max_workers=NCPU) as exec:
            # exec processes the interpolation using n processes.
            results = exec.map(self.execute, self.expu)
            res = [r for r in results]
        
        self.exps = res
    
    def clear_data(self) -> None:
        """Clears data from objects so we can dump a small pickle file"""
        self.ivts = None
        for single in self.exps:
            single.ivslice = None

    def reload_data(self, ivts):
        """Loads in data used to construct the class"""
        self.ivts = ivts
        for single in self.exps:
            single.ivslice = self.ivts[self.ivts["EXP"]==single.exp]

def main():
    st = time.time()

    print("Reading in data")
    ivts = pd.concat([pd.read_hdf(os.path.join("..", "data", "spx_iv_db_1.h5")), 
                      pd.read_hdf(os.path.join("..", "data", "spx_iv_db_2.h5"))])

    otm_mask = (  ((ivts["TYPE"]=='C') & (ivts["STRIKE"]> ivts["UNDERLYING_PRICE"])) 
                | ((ivts["TYPE"]=='P') & (ivts["STRIKE"]<=ivts["UNDERLYING_PRICE"])))

    otm  = ivts[otm_mask].dropna().reset_index(drop=True)

    otm["LOG_MONEYNESS_F"] = np.log(otm["STRIKE"]/otm["F_T"])
    otm["IVAR_MID"] = otm["IV_MID"]**2
    
    print(f"Instantiating Model and Framework using MODEL={MODEL}")

    if MODEL=="hyperbola":
        hyperbola = HyperbolaModel(bounds=FIT_BOUNDS)
        framework = AllExp(otm, hyperbola, "LOG_MONEYNESS_F", "IVAR_MID", X_BOUNDS)
    else:
        raise ValueError("MODEL must be hyperbola, ...")

    framework.by_exp()
    framework.clear_data()

    with open(f"{MODEL}_fits_med.pkl", "wb") as f:
        pickle.dump(framework, f)

    el = time.time() - st
    print(f"Completed prediction_engine.py. Elapsed {str(dt.timedelta(seconds=el))[:-7]}")

if __name__=="__main__":
    main()
