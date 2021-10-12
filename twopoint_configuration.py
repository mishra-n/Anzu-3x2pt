from twopoint_harmonic import *
from legendre import *

class SpectrumInterp(object):
    """class for 1d interpolation of spectra"""

    def __init__(self, angle, spec, kind='cubic'):
        assert np.all(angle>=0.)
        starts_with_zero=False
        self.spec0 = 0.
        if angle[0]<1.e-9:
            starts_with_zero=True
            self.spec0 = spec[0]
            angle, spec = angle[1:], spec[1:]

        if np.all(spec > 0):
            self.interp_func = interp.interp1d(np.log(angle), np.log(spec), bounds_error=False,
                                               kind=kind)
            self.interp_type = 'loglog'
        elif np.all(spec < 0):
            self.interp_func = interp.interp1d(np.log(angle), np.log(-spec), bounds_error=False,
                                               kind=kind)
            self.interp_type = 'minus_loglog'
        else:
            self.interp_func = interp.interp1d(np.log(angle), spec, bounds_error=False, fill_value=0.,
                                               kind=kind)
            self.interp_type = "log_ang"

    def __call__(self, angle):
        non_zero = angle>1.e-12
        if self.interp_type == 'loglog':
            spec = np.exp(self.interp_func(np.log(angle)))
        elif self.interp_type == 'minus_loglog':
            spec = -np.exp(self.interp_func(np.log(angle)))
        else:
            assert self.interp_type == "log_ang"
            spec = self.interp_func(np.log(angle))
        return np.where(non_zero, spec, self.spec0)

class TwoPointConfiguration:
    def __init__(self, TwoPointHarmonic, theta_max, theta_density):
        self.Universe = TwoPointHarmonic.Universe
        self.Kernel = TwoPointHarmonic.Kernel
        self.TwoPointHarmonic = TwoPointHarmonic
        
        self.theta_density = theta_density
        self.theta_min =  2 * np.pi / TwoPointHarmonic.lmax #gives smallest theta in radians
        self.theta_max = theta_max
        self.theta_range = np.logspace(np.log10(self.theta_min), np.log10(self.theta_max), self.theta_density)
        
        
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
            cos_legfac2 = get_legfactors_02(np.arange(self.TwoPointHarmonic.lmax), self.theta_range)
            flegfac2 = apply_filter(self.TwoPointHarmonic.lmax-1, 1, cos_legfac2)
            
            self.P2_ell_matrix = flegfac2
            
       #     x = np.cos(self.theta_range)
       #     self.P2_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))

       #     for ell in range(self.TwoPointHarmonic.lmax):
       #         self.P2_ell_matrix[:,ell] = special.lpmv(2, ell, x)        
       #         if ell % 1000 ==0:
       #             print(ell)
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

        C_dd_temp = C_dd_interp(full_ells) *  np.exp(-full_ells**2 * R**2) + shot_noise

        wtheta_test = np.matmul(self.P_ell_matrix, C_dd_temp)
        
        return wtheta_test
    
    def gen_gammat(self, R=0, shot_noise=0):
        self.gammat = np.zeros(shape=(len(self.Kernel.lens_nzs), len(self.Kernel.source_nzs), len(self.theta_range)))
        
        full_ells = np.arange(self.TwoPointHarmonic.lmax)
        for i in range(len(self.Kernel.lens_nzs)):
            for j in range(len(self.Kernel.source_nzs)):
                #C_dk_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_dk[i,j,:], fill_value=self.TwoPointHarmonic.C_dk[i,j,0], bounds_error=False)
                C_dk_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_dk[i,j,:], fill_value=0, bounds_error=False)
                C_dk_temp = C_dk_interp(full_ells) *  np.exp(-full_ells**2 * R**2) + shot_noise
                #C_dk_temp[C_dk_temp == np.inf] = 0
                #C_dk_temp[C_dk_temp == -np.inf] = 0
                self.gammat[i,j,:] = np.matmul(self.P2_ell_matrix, C_dk_temp)
        
        return self.gammat
    
    def gen_G_ell(self, load_matrix=False):
        self.Gp_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))
        self.Gm_ell_matrix = np.zeros(shape=(len(self.theta_range), self.TwoPointHarmonic.lmax))
        
        if load_matrix:
            self.Gp_ell_matrix = np.load('Gp_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy')
            self.Gm_ell_matrix = np.load('Gm_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy')
            
        else:
            cos_legfac22 = get_legfactors_22(np.arange(self.TwoPointHarmonic.lmax), self.theta_range)
            flegfac22_0 = apply_filter(self.TwoPointHarmonic.lmax-1, 1, cos_legfac22[0])
            flegfac22_1 = apply_filter(self.TwoPointHarmonic.lmax-1, 1, cos_legfac22[1])
            
            self.Gp_ell_matrix = (flegfac22_0 + flegfac22_1)/2
            self.Gm_ell_matrix = (flegfac22_0 - flegfac22_1)/2
            
            np.save('Gp_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy', self.Gp_ell_matrix)
            np.save('Gm_lmax=' + str(self.TwoPointHarmonic.lmax) + '_thetadensity='+ str(self.theta_density) + '.npy', self.Gm_ell_matrix)
            
        return self.Gp_ell_matrix, self.Gm_ell_matrix

            
            
        #x = np.cos(self.theta_range)
        #m = 2
        
        #for ell in range(self.TwoPointHarmonic.lmax):
        #    if ell == 0:
        #        continue
        #    else:
        #        P_1 = self.P2_ell_matrix[:,ell]
        #        P_0 = self.P2_ell_matrix[:,ell-1]
        #        
        #        self.Gp_ell_matrix[:,ell] = -((1-m**2)/(1-x**2) + 0.5*ell*(ell-1))*P_1 + (ell+m)*(x/(1-x**2))*P_0
        #        self.Gm_ell_matrix[:,ell] = m*((ell-1)*x/(1-x**2)*P_1 - (ell+m)*(1/(1-x**2))*P_0)
        
    
    def gen_xi(self, R=0, shot_noise=0):
        self.xip = np.zeros(shape=(len(self.Kernel.source_nzs), len(self.Kernel.source_nzs), len(self.theta_range)))
        self.xim = np.zeros(shape=(len(self.Kernel.source_nzs), len(self.Kernel.source_nzs), len(self.theta_range)))
        
        full_ells = np.arange(self.TwoPointHarmonic.lmax)
        for i in range(len(self.Kernel.source_nzs)):
            for j in range(len(self.Kernel.source_nzs)):
                C_kk_interp = interp.interp1d(self.TwoPointHarmonic.ells, self.TwoPointHarmonic.C_kk[i,j,:], fill_value=self.TwoPointHarmonic.C_kk[i,j,0], bounds_error=False)
                C_kk_temp =  C_kk_interp(full_ells) *  np.exp(-full_ells**2 * R**2)# * (2*full_ells+1)/(2*np.pi*full_ells**2 *(full_ells+1)**2) + shot_noise
               # C_kk_temp[C_kk_temp == np.inf] = 0
               # C_kk_temp[C_kk_temp == -np.inf] = 0
                self.xip[i,j,:] = np.matmul((self.Gp_ell_matrix + self.Gm_ell_matrix), C_kk_temp)
                self.xim[i,j,:] = np.matmul((self.Gp_ell_matrix - self.Gm_ell_matrix), C_kk_temp)
                
        return self.xip, self.xim
    
