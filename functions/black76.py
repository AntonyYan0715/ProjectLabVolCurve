from numpy import log, exp, linspace, ones, array, round
from scipy.stats import norm



class Black76:
    
    def __init__(self, F, K, T, sig, r, type, calc=False):
        self.F = F
        self.K = K
        self.T = T
        self.sig = sig
        self.r = r
        self.type = type

        self.d1 = self.calc_d1()
        self.d2 = self.calc_d2()
        
        if calc:
            self.V     = self.calc_fair_value()
            self.delta = self.calc_delta()
            self.gamma = self.calc_gamma()
            self.theta = self.calc_theta()
            self.vega  = self.calc_vega()
    
    def calc_d1(self):
        return (log(self.F/self.K)+(self.sig**2/2)*self.T)/(self.sig*self.T**0.5)
    
    def calc_d2(self):
        return self.d1 - self.sig*self.T**0.5
    
    def calc_fair_value(self):
        calls = (self.F*exp(-self.r*self.T)*norm.cdf(self.d1) 
                 - self.K*exp(-self.r*self.T)*norm.cdf(self.d2))
        puts  = (self.K*exp(-self.r*self.T)*norm.cdf(-self.d2)
                 - self.F*exp(-self.r*self.T)*norm.cdf(-self.d1))
        return (self.type=='C')*calls + (self.type=='P')*puts
    
    def calc_delta(self):
        calls =  exp(-self.r*self.T)*norm.cdf(self.d1)
        puts  = -exp(-self.r*self.T)*(norm.cdf(-self.d1))
        return (self.type=='C')*calls + (self.type=='P')*puts
    
    def calc_gamma(self):
        return exp(-self.r*self.T)*norm.pdf(self.d1) / (self.F*self.sig*self.T**0.5)
    
    def calc_theta(self):
        calls = (-self.F*exp(-self.r*self.T)*norm.pdf(self.d1)*self.sig/(2*self.T**0.5) 
                    - self.r*self.K*exp(-self.r*self.T)*norm.cdf(self.d2) 
                    + self.r*self.F*exp(-self.r*self.T)*norm.cdf(self.d1))
        puts  = (-self.F*exp(-self.r*self.T)*norm.pdf(self.d1)*self.sig/(2*self.T**0.5) 
                    + self.r*self.K*exp(-self.r*self.T)*norm.cdf(-self.d2) 
                    - self.r*self.F*exp(-self.r*self.T)*norm.cdf(-self.d1))
        return (self.type=='C')*calls + (self.type=='P')*puts
    
    def calc_vega(self):
        return self.F*exp(-self.r*self.T)*norm.pdf(self.d1)*self.T**0.5