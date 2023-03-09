from numpy import log, exp, linspace, ones, array, round
from scipy.stats import norm


class Option:
    
    def __init__(self, S, K, T, sig, r, q, type="c", calc=False):
        self.S, self.K, self.T, self.sig = S, K, T, sig
        self.r, self.q = r, q
        
        if type.lower() in ['c', 'p', "call", "put"]:
            self.type = type.lower()[0]
        else: raise ValueError(f"Option type must be c, call, p, or put, not {self.type}")
        
        self.d1 = self.calc_d1()
        self.d2 = self.calc_d2()
        
        if calc:
            self.V = self.calc_fair_value()
    
    def calc_d1(self):
        return (log(self.S/self.K)+(self.r-self.q+self.sig**2/2)*self.T)/(self.sig*self.T**0.5)
    
    def calc_d2(self):
        return self.d1 - self.sig*self.T**0.5
    
    def calc_fair_value(self):
        if self.type=='c':
            return self.S*exp(-self.q*self.T)*norm.cdf(self.d1) - self.K*exp(-self.r*self.T)*norm.cdf(self.d2)
        else:
            return self.K*exp(-self.r*self.T)*norm.cdf(-self.d2) - self.S*exp(-self.q*self.T)*norm.cdf(-self.d1)
    
    def calc_delta(self):
        if self.type=='c':
            return exp(-self.q*self.T)*norm.cdf(self.d1)
        else:
            return -exp(-self.q*self.T)*norm.cdf(-self.d1)
    
    def calc_gamma(self):
        return exp(-self.q*self.T)*norm.pdf(self.d1) / (self.S*self.sig*self.T**0.5)
    
    def calc_theta(self):
        if self.type=='c':
            return (-self.S*exp(-self.q*self.T)*norm.pdf(self.d1)*self.sig/(2*self.T**0.5) 
                    - self.r*self.K*exp(-self.r*self.T)*norm.cdf(self.d2) 
                    + self.q*self.S*exp(-self.q*self.T)*norm.cdf(self.d1))
        else:
            return (-self.S*exp(-self.q*self.T)*norm.pdf(self.d1)*self.sig/(2*self.T**0.5) 
                    + self.r*self.K*exp(-self.r*self.T)*norm.cdf(-self.d2) 
                    - self.q*self.S*exp(-self.q*self.T)*norm.cdf(-self.d1))
    
    def calc_vega(self):
        return self.S*exp(-self.q*self.T)*norm.pdf(self.d1)*self.T**0.5


def d1(S, K, T, sig, r, q):
    return (log(S/K)+(r-q+sig**2/2)*T)/(sig*T**0.5)

def d2(S, K, T, sig, r, q):
    return d1(S, K, T, sig, r, q) - sig*T**0.5

def fair_value(S, K, T, sig, r, q, type="c"):
    d1 = d1(S, K, T, sig, r, q)
    d1 = d1(S, K, T, sig, r, q)
    if type=='c': return S*exp(-q*T)*norm.cdf(d1)  - K*exp(-r*T)*norm.cdf(d2)
    else:         return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)
