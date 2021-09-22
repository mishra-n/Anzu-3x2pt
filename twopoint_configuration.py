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
        
#    def gen_P_ell(self):
#        m=2
#        
#        x = np.cos(self.theta_range)
#        
#        P_0 = special.legendre(0)(x)
#        P_1 = special.legendre(1)(x)        
#
#        self.P_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))
#
#        for ell in range(self.TwoPointHarmonic.lmax):
#            if ell==0:
#                self.P_ell_matrix[:,ell] = P_0
#            elif ell==1:
#                self.P_ell_matrix[:,ell] = P_1
#            else:
#                P_2 = ((2*(ell-1)+1)*x*P_1 - (ell-1)*P_0)/(ell)
#                P_0 = P_1
#                P_1 = P_2
#                self.P_ell_matrix[:,ell] =  P_2
#                
#        return self.P_ell_matrix
    
    def gen_P_ell(self, load_matrix=False):
        if load_matrix:
            self.P_ell_matrix = np.load('P_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy')
        else:
            x = np.cos(self.theta_range)
            self.P_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))

            for ell in range(self.TwoPointHarmonic.lmax):
                self.P_ell_matrix[:,ell] = special.lpmv(0, ell, x)
                if ell % 1000 ==0:
                    print(ell)
            np.save('P_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy', self.P_ell_matrix)
            
        return self.P_ell_matrix
        
    def gen_P2_ell(self, load_matrix=False):
        if load_matrix:
            self.P2_ell_matrix = np.load('P2_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy')
        else:
            x = np.cos(self.theta_range)
            self.P2_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))

            for ell in range(self.TwoPointHarmonic.lmax):
                self.P2_ell_matrix[:,ell] = special.lpmv(2, ell, x)        
                if ell % 1000 ==0:
                    print(ell)
            np.save('P2_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy', self.P2_ell_matrix)
        return self.P2_ell_matrix
                
    def gen_wtheta(self, R=0, shot_noise=0):
        self.wtheta = np.zeros(shape=(len(self.Kernel.lens_nzs), len(self.theta_range)))
        
        full_ells = np.arange(self.TwoPointHarmonic.lmax)
        for i in range(len(self.Kernel.lens_nzs)):
            C_dd_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_dd[i,i,:], fill_value=0, bounds_error=False)

            C_dd_temp = (2*full_ells+1)/(4*np.pi) * C_dd_interp(full_ells) *  np.exp(-full_ells**2 * R**2) + shot_noise

            self.wtheta[i,:] = np.matmul(self.P_ell_matrix, C_dd_temp)
        
        return self.wtheta
    
    def test_wtheta(self, ell_test, data_test, R=0.0002, shot_noise=0):
        #self.wtheta = np.zeros(shape=(len(self.Kernel.source_nzs), len(self.theta_range)))
        
        full_ells = np.arange(self.TwoPointHarmonic.lmax)

        C_dd_interp = interp.interp1d(ell_test, data_test, fill_value=0, bounds_error=False)

        C_dd_temp = (2*full_ells+1)/(4*np.pi) * C_dd_interp(full_ells) *  np.exp(-full_ells**2 * R**2) + shot_noise

        wtheta_test = np.matmul(self.P_ell_matrix, C_dd_temp)
        
        return wtheta_test
    
    def gen_gammat(self, R=0, shot_noise=0):
        self.gammat = np.zeros(shape=(len(self.Kernel.lens_nzs), len(self.Kernel.source_nzs), len(self.theta_range)))
        
        full_ells = np.arange(self.TwoPointHarmonic.lmax)
        for i in range(len(self.Kernel.lens_nzs)):
            for j in range(len(self.Kernel.source_nzs)):
                C_dk_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_dk[i,j,:], fill_value=self.TwoPointHarmonic.C_dk[i,j,0], bounds_error=False)
                C_dk_temp = (2*full_ells+1)/(4*np.pi*full_ells*(full_ells+1)) * C_dk_interp(full_ells) *  np.exp(-full_ells**2 * R**2) + shot_noise
                C_dk_temp[C_dk_temp == np.inf] = 0
                C_dk_temp[C_dk_temp == -np.inf] = 0
                self.gammat[i,j,:] = np.matmul(self.P2_ell_matrix, C_dk_temp)
        
        return self.gammat
    
    def gen_G_ell(self):
        self.Gp_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))
        self.Gm_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))
        
        x = np.cos(self.theta_range)
        m = 2
        
        for ell in range(self.TwoPointHarmonic.lmax):
            if ell == 0:
                continue
            else:
                P_1 = self.P2_ell_matrix[:,ell]
                P_0 = self.P2_ell_matrix[:,ell-1]
                
                self.Gp_ell_matrix[:,ell] = -((1-m**2)/(1-x**2) + 0.5*ell*(ell-1))*P_1 + (ell+m)*(x/(1-x**2))*P_0
                self.Gm_ell_matrix[:,ell] = m*((ell-1)*x/(1-x**2)*P_1 - (ell+m)*(1/(1-x**2))*P_0)
        
        return self.Gp_ell_matrix, self.Gm_ell_matrix
    
    def gen_xi(self, R=0, shot_noise=0):
        self.xip = np.zeros(shape=(len(self.Kernel.source_nzs), len(self.Kernel.source_nzs), len(self.theta_range)))
        self.xim = np.zeros(shape=(len(self.Kernel.source_nzs), len(self.Kernel.source_nzs), len(self.theta_range)))
        
        full_ells = np.arange(self.TwoPointHarmonic.lmax)
        for i in range(len(self.Kernel.source_nzs)):
            for j in range(len(self.Kernel.source_nzs)):
                C_kk_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_kk[i,j,:], fill_value=self.TwoPointHarmonic.C_kk[i,j,0], bounds_error=False)
                C_kk_temp = (2*full_ells+1)/(2*np.pi*full_ells**2 *(full_ells+1)**2) * C_kk_interp(full_ells) *  np.exp(-full_ells**2 * R**2) + shot_noise
                C_kk_temp[C_kk_temp == np.inf] = 0
                C_kk_temp[C_kk_temp == -np.inf] = 0
                self.xip[i,j,:] = np.matmul((self.Gp_ell_matrix + self.Gm_ell_matrix), C_kk_temp)
                self.xim[i,j,:] = np.matmul((self.Gp_ell_matrix - self.Gm_ell_matrix), C_kk_temp)
                
        return self.xip, self.xim    