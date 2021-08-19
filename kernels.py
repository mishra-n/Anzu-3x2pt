from universe import *

class Kernels:
    def __init__(self, Universe):
        self.Universe = Universe     
        
########################################################################
#nz functions
########################################################################

    def loadDES_nzs(self, fits_file):
        data = twopoint.TwoPointFile.from_fits(fits_file + '.fits')
        
        z = data.kernels[0].z
        
        self.source_nzs = data.kernels[0].nzs
        self.lens_nzs = data.kernels[1].nzs
        self.Universe.redshift_dependence(z)

    def plot_source_nzs(self, save=False, name='source_nzs.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        colormap='viridis'
        for i, nz in enumerate(self.source_nzs):
            color = plt.get_cmap(colormap)(0.2*i)
            ax.plot(self.Universe.z, nz, lw=1.5, color=color)
            ax.fill_between(self.Universe.z, 0, nz, color=color, alpha=0.2)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Normalized counts')
        ax.set_xlim(0, 2)
        ax.set_ylim(bottom=0)
        if save:
            plt.savefig(name, dpi=200, bbox_inches='tight')
        plt.show()
        plt.clf()

    def plot_lens_nzs(self, save=False, name='lens_nzs.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        colormap='viridis'
        for i, nz in enumerate(self.lens_nzs):
            color = plt.get_cmap(colormap)(0.2*i)
            ax.plot(self.Universe.z, nz, lw=1.5, color=color)
            ax.fill_between(self.Universe.z, 0, nz, color=color, alpha=0.2)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Normalized counts')
        ax.set_xlim(0, 2)
        ax.set_ylim(bottom=0)
        if save:
            plt.savefig(name, dpi=200, bbox_inches='tight')
        plt.show()
        plt.clf()

########################################################################
#kernel functions
########################################################################

        
    def lens_kernel(self):        
        dz_div_dchi = self.Universe.H_z/self.Universe.c
        
        self.q_delta = []
        
        for i, nz in enumerate(self.lens_nzs):
            nz_bar = integrate.simps(nz, x=self.Universe.z)
            self.q_delta.append(nz/nz_bar * dz_div_dchi)
            
    def source_kernel(self, ia=False, A_0=0.44, gamma=0):
        prefix = 3 * self.Universe.H_0**2 * self.Universe.omega_m / (2 * self.Universe.c**2) * (self.Universe.chi / self.Universe.a)
        dz_div_dchi = self.Universe.H_z/self.Universe.c
        self.Universe.linear_growth()
        self.q_kappa = []
        
        for i, nz in enumerate(self.source_nzs):
            nz_bar = integrate.simps(nz, x=self.Universe.z)
            q_kappa_temp = np.zeros(shape=self.Universe.chi.shape[0])
            ia_temp = np.zeros(shape=self.Universe.chi.shape[0])
            for n, chi_val in enumerate(self.Universe.chi):
                integrand = nz/nz_bar * dz_div_dchi * ((self.Universe.chi) - chi_val)/self.Universe.chi
                q_kappa_temp[n] = integrate.simps(integrand[n::], x=self.Universe.chi[n::])
                
            if ia:
                z_0 = 0.62
                C = 0.0134
                A_0 = A_0#.44
                alpha = gamma
                D_z = self.Universe.D_z
                A = A_0 * ((1+self.Universe.z)/(1+z_0))**alpha * C / D_z
                print('doing')
                ia_temp = - A * nz/nz_bar * dz_div_dchi
        
            self.q_kappa.append(prefix * q_kappa_temp + ia_temp)
            
            
    def plot_lens_kernels(self, test=True, save=False, name='lens_kernels.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        colormap='viridis'
        
        for i, q_delta in enumerate(self.q_delta):
            color = plt.get_cmap(colormap)(0.2*i)
            ax.plot(self.Universe.chi, q_delta, color=color)
            if test:
                stuff = np.loadtxt('lens_kernels.txt')
                ax.plot(stuff[:,0], stuff[:,i+1],'--', color=color)
        ax.set_ylabel('$q_{\delta}$')
        ax.set_xlabel('Co-moving distance [Mpc]')
        ax.set_xlim(0, 4500)
        if save:
            plt.savefig(name, dpi=200, bbox_inches='tight')
        plt.show()
        plt.clf()

    def plot_source_kernels(self, test=True, save=False, name='source_kernels.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        colormap='viridis'
        
        for i, q_kappa in enumerate(self.q_kappa):
            color = plt.get_cmap(colormap)(0.2*i)
            ax.plot(self.Universe.chi, q_kappa, color=color)
            if test:
                stuff = np.loadtxt('source_kernels.txt')
                ax.plot(stuff[:,0], stuff[:,i+1],'--', color=color)
        ax.set_ylabel('$q_{\kappa}$')
        ax.set_xlabel('Co-moving distance [Mpc]')
        ax.set_xlim(0, 4500)
        if save:
            plt.savefig(name, dpi=200, bbox_inches='tight')
        plt.show()
        plt.clf()



