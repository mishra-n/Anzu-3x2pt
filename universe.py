from headers import *

class Universe:
    def __init__(self, h=0.7, omega_b=0.046, omega_m=0.286, sigma8=0.82, n_s=0.96, w=-1, tau=0.07):
        self.c = 3e5 #km/s 
        
        self.h = h
        self.omega_b = omega_b
        self.omega_m = omega_m

        self.sigma8 = sigma8
        self.n_s = n_s
        self.w = w
        self.tau = tau
        
        #derived parameters
        self.H_0 = h*100 #km/s/Mpc
        self.omega_cdm = self.omega_m - self.omega_b
        
        #little h parameters
        self.ombh2 = self.omega_b * self.h**2
        self.omch2 = self.omega_cdm*self.h**2
        
    def runCAMB(self):
        
        #Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()

        #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
        pars.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=0.06, omk=0, tau=self.tau)
        pars.InitPower.set_params(As=2e-9, ns=self.n_s, r=0)
        pars.set_for_lmax(2500, lens_potential_accuracy=0);

        #calculate results for these parameters
        results = camb.get_results(pars)
        
        self.CAMB_results = results
        
    def redshift_dependence(self, z):
        self.z = z
        self.a = 1/(1+self.z)
        self.H_z = self.CAMB_results.hubble_parameter(self.z)
        comoving_radial_distance = np.vectorize(self.CAMB_results.comoving_radial_distance)
        self.chi = comoving_radial_distance(self.z)
        
    def linear_growth(self):
        #default growth rate, we can set our own structure growth using Anzu bias parameters
        self.D_z = self.a * special.hyp2f1(1./3,1,11./6,-self.a**3/self.omega_m*(1-self.omega_m)) / special.hyp2f1(1./3,1,11./6,-1/self.omega_m*(1-self.omega_m))
        
    def set_biasParams_Anzu(self, b1, b2, bs2, bnabla2, SN):
        self.bvec = [b1, b2, bs2, bnabla2, SN]
    
    def genPowerSpectrum(self, k_range, sigma8=None, runtime=False):
        if sigma8 is not None:
            self.sigma8 = sigma8
            
        self.k_range = k_range
        emu = LPTEmulator(kecleft=True)
        
        wave_density = len(self.k_range)
        
        self.Pmm = np.zeros(shape=(len(self.k_range), len(self.z)))
        self.Pgg = np.zeros(shape=(len(self.k_range), len(self.z)))
        self.Pmg = np.zeros(shape=(len(self.k_range), len(self.z)))
        
        for i, a in enumerate(self.a):
            cosmo_vec = np.atleast_2d([self.ombh2, self.omch2, self.w, self.n_s, self.sigma8,  self.H_0,  3.046,  a])
            emu_spec = emu.predict(self.k_range/self.h, cosmo_vec)
            biased_spec = emu.basis_to_full(self.k_range/self.h, self.bvec, emu_spec[0])
            
            self.Pmm[:,i] = emu_spec[0,0,:]/self.h**3
            self.Pgg[:,i] = biased_spec[0:wave_density]/self.h**3
            self.Pmg[:,i] = biased_spec[wave_density::]/self.h**3
            
        return self.z, (self.Pmm, self.Pgg, self.Pmg)
    
    def extrapolatePowerSpectrum(self, new_kmax):
        pixels_to_extrapolate = 5
        k_end = self.k_range[-pixels_to_extrapolate: -1]
        new_ks = np.logspace(np.log10(k_end[-1]), np.log10(new_kmax), 100)
        
        self.k_range = np.append(self.k_range, new_ks[1::])
        
        temp_Pmm = np.zeros(shape=(len(self.k_range), len(self.z)))
        temp_Pgg = np.zeros(shape=(len(self.k_range), len(self.z)))
        temp_Pmg = np.zeros(shape=(len(self.k_range), len(self.z)))
        for i, z in enumerate(self.z):
            Pmm_end = self.Pmm[-pixels_to_extrapolate: -1, i]
            Pgg_end = self.Pgg[-pixels_to_extrapolate: -1, i]
            Pmg_end = self.Pmg[-pixels_to_extrapolate: -1, i]

            results = stats.linregress(np.log10(k_end), np.log10(Pmm_end))
            new_logP = np.log10(new_ks)*results.slope + results.intercept
            new_Pmm = 10**new_logP

            results = stats.linregress(np.log10(k_end), np.log10(Pgg_end))
            new_logP = np.log10(new_ks)*results.slope + results.intercept
            new_Pgg = 10**new_logP

            results = stats.linregress(np.log10(k_end), np.log10(Pmg_end))
            new_logP = np.log10(new_ks)*results.slope + results.intercept
            new_Pmg = 10**new_logP

            temp_Pmm[:,i] = np.append(self.Pmm[:,i], new_Pmm[1::])
            temp_Pgg[:,i] = np.append(self.Pgg[:,i], new_Pgg[1::])
            temp_Pmg[:,i] = np.append(self.Pmg[:,i], new_Pmg[1::])
            
        self.Pmm = temp_Pmm
        self.Pgg = temp_Pgg
        self.Pmg = temp_Pmg

        return self.z, (self.Pmm, self.Pgg, self.Pmg)

    def addNoisePowerSpectrum(self, A=1, n=2):
        
        self.Pmm = self.Pmm + A*self.k_range[:, None]**n
        self.Pgg = self.Pgg + A*self.k_range[:, None]**n
        self.Pmg = self.Pmg + A*self.k_range[:, None]**n

        return self.z, (self.Pmm, self.Pgg, self.Pmg)
    
    def multiplyNoisePowerSpectrum(self, A=1, n=2):
        
        self.Pmm = self.Pmm*(1 + A*self.k_range[:, None]**n)
        self.Pgg = self.Pgg*(1 + A*self.k_range[:, None]**n)
        self.Pmg = self.Pmg*(1 + A*self.k_range[:, None]**n)

        return self.z, (self.Pmm, self.Pgg, self.Pmg)

    def divideNoisePowerSpectrum(self, A=1, n=2):
        
        self.Pmm = self.Pmm/(1 + A*self.k_range[:, None]**n)
        self.Pgg = self.Pgg/(1 + A*self.k_range[:, None]**n)
        self.Pmg = self.Pmg/(1 + A*self.k_range[:, None]**n)

        return self.z, (self.Pmm, self.Pgg, self.Pmg)
    
    def multiplyNoisePowerSpectrum2(self, A=1, n=2):
        
        self.Pmm = self.Pmm*(1 + (A*self.k_range[:, None])**n / (A**n + self.k_range[:, None]**n))
        self.Pgg = self.Pgg*(1 + (A*self.k_range[:, None])**n / (A**n + self.k_range[:, None]**n))
        self.Pmg = self.Pmg*(1 + (A*self.k_range[:, None])**n / (A**n + self.k_range[:, None]**n))

        return self.z, (self.Pmm, self.Pgg, self.Pmg)

    def divideNoisePowerSpectrum2(self, A=1, n=2):
        
        self.Pmm = self.Pmm/(1 + (A*self.k_range[:, None])**n / (A**n + self.k_range[:, None]**n))
        self.Pgg = self.Pgg/(1 + (A*self.k_range[:, None])**n / (A**n + self.k_range[:, None]**n))
        self.Pmg = self.Pmg/(1 + (A*self.k_range[:, None])**n / (A**n + self.k_range[:, None]**n))

        return self.z, (self.Pmm, self.Pgg, self.Pmg)


            
            