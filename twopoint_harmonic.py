import headers
from universe import *
from kernels import *

class TwoPointHarmonic:
    def __init__(self, Universe, Kernel, lmax):
        self.Universe = Universe
        self.Kernel = Kernel
        
        self.lmax = lmax
        self.ells = np.logspace(-2, np.log10(self.lmax), 3000)
        
    def genC_kk(self):
        
        self.C_kk = np.zeros(shape=(len(self.Kernel.q_kappa), len(self.Kernel.q_kappa), len(self.ells)))
                             
        for i, q_kappa_i in enumerate(self.Kernel.q_kappa):
            for j, q_kappa_j in enumerate(self.Kernel.q_kappa):
                C_kk_integrand = np.zeros(shape=(len(self.Universe.chi), len(self.ells)))
                             
                for d, chi_val in enumerate(self.Universe.chi):
                    k = (self.ells + 1/2)/chi_val
                    if chi_val==0:
                        continue
                             
                    Pmm_interp = interp.interp1d(self.Universe.k_range, self.Universe.Pmm[:, d], fill_value=self.Universe.Pmm[-1, d], bounds_error=False)
                    C_kk_integrand[d, :] = q_kappa_i[d] * q_kappa_j[d] / chi_val**2 * Pmm_interp(k)
                    
                self.C_kk[i, j, :] = integrate.simps(C_kk_integrand[1::,:], x=self.Universe.chi[1::], axis=0)
                
        return self.C_kk

    def genC_dk(self):
        
        self.C_dk = np.zeros(shape=(len(self.Kernel.q_delta), len(self.Kernel.q_kappa), len(self.ells)))
                             
        for i, q_delta_i in enumerate(self.Kernel.q_delta):
            for j, q_kappa_j in enumerate(self.Kernel.q_kappa):
                C_dk_integrand = np.zeros(shape=(len(self.Universe.chi), len(self.ells)))
                             
                for d, chi_val in enumerate(self.Universe.chi):
                    k = (self.ells + 1/2)/chi_val
                    if chi_val==0:
                        continue
                             
                    Pmg_interp = interp.interp1d(self.Universe.k_range, self.Universe.Pmg[:, d], fill_value=self.Universe.Pmg[-1, d], bounds_error=False)
                    C_dk_integrand[d, :] = q_delta_i[d] * q_kappa_j[d] / chi_val**2 * Pmg_interp(k)
                    
                self.C_dk[i, j, :] = integrate.simps(C_dk_integrand[1::,:], x=self.Universe.chi[1::], axis=0)
                
        return self.C_dk

    def genC_dd(self):
        
        self.C_dd = np.zeros(shape=(len(self.Kernel.q_delta), len(self.Kernel.q_delta), len(self.ells)))
                             
        for i, q_delta_i in enumerate(self.Kernel.q_delta):
            for j, q_delta_j in enumerate(self.Kernel.q_delta):
                C_dd_integrand = np.zeros(shape=(len(self.Universe.chi), len(self.ells)))
                             
                for d, chi_val in enumerate(self.Universe.chi):
                    k = (self.ells + 1/2)/chi_val
                    if chi_val==0:
                        continue
                             
                    Pgg_interp = interp.interp1d(self.Universe.k_range, self.Universe.Pgg[:, d], fill_value=self.Universe.Pgg[-1, d], bounds_error=False)
                    C_dd_integrand[d, :] = q_delta_i[d] * q_delta_j[d] / chi_val**2 * Pgg_interp(k)
                    
                self.C_dd[i, j, :] = integrate.simps(C_dd_integrand[1::,:], x=self.Universe.chi[1::], axis=0)
                
        return self.C_dd
                    
                    #Pgg_interp = interp.interp1d(k_range, self.Universe.Pgg[:, d], fill_value=self.Universe.Pgg[-1, d], bounds_error=False)
                    #Pmg_interp = interp.interp1d(k_range, self.Universe.Pmg[:, d], fill_value=self.Universe.Pmg[-1, d], bounds_error=False)               