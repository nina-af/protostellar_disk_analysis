#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy import stats

import pytreegrav as pg


class Snapshot_TURBSPHERE:
    """
    Class for analyzing turbulence properties in TURBSPHERE snapshots.
    Default code units: 1 Msun, 1 pc, 1 m/s, 1 T.
    """

    def __init__(self, M0, R0, alpha0, bturb0, fname,
                 rho_min=None, G_code=4300.71, B_unit=1e4, cs0=200.0,
                 verbose=False):

        self.fname   = fname
        self.snapdir = self.get_snapdir()

        # Physical constants.
        self.PROTONMASS_CGS     = 1.6726e-24
        self.ELECTRONMASS_CGS   = 9.10953e-28
        self.BOLTZMANN_CGS      = 1.38066e-16
        self.HYDROGEN_MASSFRAC  = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS        = 2.9979e10
        self.HYDROGEN_MASSFRAC  = 0.76

        # Initial cloud mass, radius, turbulent virial parameter, and ratio of
        # magnetic to gravitational energy.
        self.M0 = M0
        self.R0 = R0
        self.L0 = (4.0 * np.pi * self.R0**3 / 3.0)**(1.0/3.0) # Volume-equivalent box length.
        self.alpha0 = alpha0
        self.bturb0 = bturb0
        self.G_code = G_code
        self.B_unit = B_unit
        self.cs0    = cs0

        # Minimum density threshold for selecting gas cells.
        self.rho_min = rho_min

        # Open HDF5 file.
        with h5py.File(fname, 'r') as f:
            header = f['Header']
            p0     = f['PartType0']

            # Header attributes.
            self.box_size = header.attrs['BoxSize']
            self.num_p0   = header.attrs['NumPart_Total'][0]
            self.t        = header.attrs['Time']

            # Unit conversions to cgs; note typo in header for G_code.
            self.G_code      = header.attrs['Gravitational_Constant_In_Code_Inits']
            if 'Internal_UnitB_In_Gauss' in header.attrs:
                self.B_code = header.attrs['Internal_UnitB_In_Gauss']
            else:
                self.B_code = 2.916731267922059e-09
            self.l_unit      = header.attrs['UnitLength_In_CGS']
            self.m_unit      = header.attrs['UnitMass_In_CGS']
            self.v_unit      = header.attrs['UnitVelocity_In_CGS']
            self.B_unit      = B_unit                           # Magnetic field unit in Gauss.
            self.t_unit      = self.l_unit / self.v_unit
            self.t_unit_myr  = self.t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
            self.rho_unit    = self.m_unit / self.l_unit**3
            self.nH_unit     = self.rho_unit/self.PROTONMASS_CGS
            self.P_unit      = self.m_unit / self.l_unit / self.t_unit**2
            self.spec_L_unit = self.l_unit * self.v_unit        # Specific angular momentum (get_net_ang_mom).
            self.L_unit      = self.spec_L_unit * self.m_unit   # Angular momentum.
            self.E_unit      = self.l_unit**2 / self.t_unit**2  # Energy [erg].
            self.eta_unit    = self.l_unit**2 / self.t_unit     # Nonideal MHD diffusivities.
            # Convert internal energy to temperature units.
            self.u_to_temp_units = (self.PROTONMASS_CGS/self.BOLTZMANN_CGS)*self.E_unit

            # PartType0 data.
            self.all_ids  = p0['ParticleIDs'][()]
            self.all_rho  = p0['Density'][()]
            self.gas_ids  = self.get_gas_ids_above_density_cut()
            self.idx_g    = np.isin(self.all_ids, self.gas_ids)
            self.p0_ids   = p0['ParticleIDs'][()][self.idx_g]         # Particle IDs.
            self.p0_m     = p0['Masses'][()][self.idx_g]              # Masses.
            self.p0_rho   = p0['Density'][()][self.idx_g]             # Density.
            self.p0_hsml  = p0['SmoothingLength'][()][self.idx_g]     # Particle smoothing length.
            self.p0_E_int = p0['InternalEnergy'][()][self.idx_g]      # Internal energy.
            self.p0_P     = p0['Pressure'][()][self.idx_g]            # Pressure.
            self.p0_cs    = p0['SoundSpeed'][()][self.idx_g]          # Sound speed.
            self.p0_x     = p0['Coordinates'][()][self.idx_g, 0]      # Coordinates.
            self.p0_y     = p0['Coordinates'][()][self.idx_g, 1]
            self.p0_z     = p0['Coordinates'][()][self.idx_g, 2]
            self.p0_u     = p0['Velocities'][()][self.idx_g, 0]       # Velocities.
            self.p0_v     = p0['Velocities'][()][self.idx_g, 1]
            self.p0_w     = p0['Velocities'][()][self.idx_g, 2]
            self.p0_Ne    = p0['ElectronAbundance'][()][self.idx_g]   # Electron abundance.
            if 'MagneticField' in p0.keys():
                self.p0_Bx    = p0['MagneticField'][()][self.idx_g, 0]
                self.p0_By    = p0['MagneticField'][()][self.idx_g, 1]
                self.p0_Bz    = p0['MagneticField'][()][self.idx_g, 2]
                self.p0_B_mag = np.sqrt(self.p0_Bx**2 + self.p0_By**2 + self.p0_Bz**2)
                if 'NonidealDiffusivities' in p0.keys():
                    if verbose:
                        print('Reading NMHD coefficients from snapshot...', flush=True)
                    self.p0_eta_O = p0['NonidealDiffusivities'][()][:, 0]
                    self.p0_eta_H = p0['NonidealDiffusivities'][()][:, 1]
                    self.p0_eta_A = p0['NonidealDiffusivities'][()][:, 2]
            else:
                self.p0_Bx    = np.zeros(len(self.p0_ids))
                self.p0_By    = np.zeros(len(self.p0_ids))
                self.p0_Bz    = np.zeros(len(self.p0_ids))
                self.p0_B_mag = np.zeros(len(self.p0_ids))
            self.p0_pot = p0['Potential'][()][self.idx_g]             # Gravitational potential.
            # Hydrogen number density and total metallicity.
            self.p0_n_H  = (1.0 / self.PROTONMASS_CGS) * \
                            np.multiply(self.p0_rho * self.rho_unit, 1.0 - p0['Metallicity'][()][self.idx_g, 0])
            self.p0_total_metallicity = p0['Metallicity'][()][self.idx_g, 0]
            # Calculate mean molecular weight.
            self.p0_mean_molecular_weight = self.get_mean_molecular_weight()
            # Neutral hydrogen abundance, molecular mass fraction.
            self.p0_neutral_H_abundance = p0['NeutralHydrogenAbundance'][()][self.idx_g]
            self.p0_molecular_mass_frac = p0['MolecularMassFraction'][()][self.idx_g]

            # Calculate gas adiabatic index and temperature.
            fH, f, xe            = self.HYDROGEN_MASSFRAC, self.p0_molecular_mass_frac, self.p0_Ne
            f_mono, f_di         = fH*(xe + 1.-f) + (1.-fH)/4., fH*f/2.
            gamma_mono, gamma_di = 5./3., 7./5.
            gamma                = 1. + (f_mono + f_di) / (f_mono/(gamma_mono-1.) + f_di/(gamma_di-1.))
            self.p0_temperature  = (gamma - 1.) * self.p0_mean_molecular_weight * \
                                    self.u_to_temp_units * self.p0_E_int
            self.gamma           = gamma
            
            # Temperature stored by GIZMO
            if 'Temperature' in p0.keys():
                self.p0_temperature_GIZMO = p0['Temperature'][()]
                                
            # Dust temperature.
            if 'Dust_Temperature' in p0.keys():
                self.p0_dust_temp = p0['Dust_Temperature'][()]

            # For convenience, coordinates and velocities in a (n_gas, 3) array.
            self.p0_coord = np.vstack((self.p0_x, self.p0_y, self.p0_z)).T
            self.p0_vel   = np.vstack((self.p0_u, self.p0_v, self.p0_w)).T
            self.p0_mag   = np.vstack((self.p0_Bx, self.p0_By, self.p0_Bz)).T

        # Get 3D gas velocity dispersion, Mach number based on initial cloud parameters.
        self.sigma_3D0 = np.sqrt((3.0 * self.alpha0 * self.G_code * self.M0) / (5.0 * self.R0))
        self.Mach_3D0  = self.sigma_3D0 / self.cs0

        # Get 3D rms gas velocity dispersion, Mach number from gas cell data.
        self.sigma_3D = self.get_sigma_3D()
        self.Mach_3D  = self.get_Mach_3D()

    # Get snapshot number from filename.
    def get_i(self):
        return int(self.fname.split('snapshot_')[1].split('.hdf5')[0])

    # Get snapshot datadir from filename.
    def get_snapdir(self):
        return self.fname.split('snapshot_')[0]

    # Return particle IDs of gas above density threshhold.
    def get_gas_ids_above_density_cut(self):
        if self.rho_min is None:
            return self.all_ids
        cut = (self.all_rho * self.rho_unit) > self.rho_min
        return self.all_ids[cut]

    # Get 3D gas velocity dispersion.
    def get_sigma_3D(self):
        m        = self.p0_m
        u, v, w  = self.p0_u, self.p0_v, self.p0_w
        sigma_3D = np.sqrt(self.weight_std(u, m)**2.0 + self.weight_std(v, m)**2.0 + \
                           self.weight_std(w, m)**2.0)
        return sigma_3D

    # Get 3D gas Mach number.
    def get_Mach_3D(self, cs_const=None):
        if cs_const is not None:
            print('Using constant cs = {0:.2f} m/s'.format(cs_const))
            cs = cs_const
        else:
            print('Using gas cell cs')
            cs = self.p0_cs
        m       = self.p0_m
        u, v, w = self.p0_u, self.p0_v, self.p0_w
        Mach_3D = np.sqrt(self.weight_std(np.divide(u, cs), m)**2.0 + self.weight_std(np.divide(v, cs), m)**2.0 + \
                          self.weight_std(np.divide(w, cs), m)**2.0)
        return Mach_3D

    # Get gravitational potential energy (need pytreegrav module).
    def get_potential_energy(self):
        m, h, pos = self.p0_m, self.p0_hsml, self.p0_coord
        E_pot     = 0.5 * np.sum(m * pg.Potential(pos, m, h, G=self.G_code))
        return E_pot

    # Get kinetic energy [code units].
    def get_kinetic_energy(self):
        m, vel  = self.p0_m, self.p0_vel
        dv      = vel - np.average(vel, weights=m, axis=0)
        v_sqr   = np.sum(dv**2,axis=1)
        E_kin   = 0.5 * np.sum(m * v_sqr)
        return E_kin

    # Get magnetic energy [code units].
    def get_magnetic_energy(self):
        m, rho  = self.p0_m, self.p0_rho
        B_mag   = self.p0_B_mag
        vol     = (m / rho) * self.l_unit**3
        E_mag   = (1.0/(8.0 * np.pi)) * np.sum(B_mag * B_mag * vol * self.B_unit**2) / (self.E_unit * self.m_unit)
        return E_mag

    # Calculate gas mean molecular weight.
    def get_mean_molecular_weight(self):
        T_eff_atomic          = 1.23 * (5.0/3.0-1.0) * self.u_to_temp_units * self.p0_E_int
        nH_cgs                = self.p0_rho * self.nH_unit
        T_transition          = self._DMIN(8000., nH_cgs)
        f_mol                 = 1./(1. + T_eff_atomic**2/T_transition**2)
        return 4. / (1. + (3. + 4.*self.p0_Ne - 2.*f_mol) * self.HYDROGEN_MASSFRAC)

    # Compute weighted average.
    def weight_avg(self, data, weights):
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sum(data * weights) / weightsum
        else:
            return 0

    # Compute weighted standard deviation.
    def weight_std(self, data, weights):
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sqrt(np.sum(((data - self.weight_avg(data, weights))**2) * weights) / weightsum)
        else:
            return 0

    def _sigmoid_sqrt(self, x):
        return 0.5*(1 + x/np.sqrt(1+x*x))
    def _DMIN(self, a, b):
        return np.where(a < b, a, b)
    def _DMAX(self, a, b):
        return np.where(a > b, a, b)

    def _print_stats(self, data):
        print('Min:  {0:.3e}'.format(np.min(data)))
        print('Max:  {0:.3e}'.format(np.max(data)))
        print('Mean: {0:.3e}'.format(np.mean(data)))
