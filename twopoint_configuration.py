from twopoint_harmonic import *

class TwoPointConfiguration:
    def __init__(self, TwoPointHarmonic, theta_max, theta_density):
        self.Universe = TwoPointHarmonic.Universe
        self.Kernel = TwoPointHarmonic.Kernel
        self.TwoPointHarmonic = TwoPointHarmonic
        
        self.theta_density = theta_density
        self.theta_min =  2 * np.pi / TwoPointHarmonic.lmax #gives smallest theta in radians
        self.theta_max = theta_max
        self.theta_range = np.logspace(np.log10(self.theta_min), np.log10(self.theta_max), self.theta_density)
        
    def genLegendre(self):
        
        x = np.cos(self.theta_range)
        
        P_0 = special.legendre(0)(x)
        P_1 = special.legendre(1)(x)

        self.LegendrePolyMatrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))

        for ell in range(self.TwoPointHarmonic.lmax):
            if ell==0:
                self.LegendrePolyMatrix[:,ell] = (2*ell+1)/(4*np.pi) * P_0
            elif ell==1:
                self.LegendrePolyMatrix[:,ell] = (2*ell+1)/(4*np.pi) * P_1
            else:
                P_2 = ((2*(ell-1)+1)*x*P_1 - (ell-1)*P_0)/(ell)
                P_0 = P_1
                P_1 = P_2
                self.LegendrePolyMatrix[:,ell] = (2*ell+1)/(4*np.pi) * P_2
                
        return self.LegendrePolyMatrix
                
    def gen_w(self, R=0, shot_noise=0):
        self.w = np.zeros(shape=(len(self.Kernel.source_nzs), len(self.theta_range)))
        for i in range(len(self.Kernel.source_nzs)):
            C_dd_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_dd[i,i,:], fill_value=self.TwoPointHarmonic.C_dd[i,i,0], bounds_error=False)

            C_dd_temp = C_dd_interp(np.arange(self.TwoPointHarmonic.lmax)) **  np.exp(-np.arange(self.TwoPointHarmonic.lmax)**2 * R**2) + shot_noise

            self.w[i,:] = np.matmul(self.LegendrePolyMatrix, C_dd_temp)
        
        return self.w