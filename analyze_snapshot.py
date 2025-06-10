#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy import stats

import pickle as pk

import yt
import unyt

class Cloud:
    """
    Class for calculating bulk cloud properties.
    Parameters:
        - M0: initial cloud mass [code_mass].
        - R0: initial cloud radius [code_length].
        - alpha0: initial cloud turbulent virial parameter.
        - G_code: gravitational constant in code units
        (default: 4300.17 in [pc * Msun^-1 * (m/s)^2]).
    """

    def __init__(self, M, R, alpha, G_code=4300.71):

        # Initial cloud mass, radius, and turbulent virial parameter.
        # G_code: gravitational constant in code units [default: 4300.71].
        self.M = M
        self.R = R
        self.L = (4.0 * np.pi * self.R**3 / 3.0)**(1.0/3.0)
        self.alpha  = alpha
        self.G_code = G_code

        self.rho     = self.get_initial_density()
        self.Sigma   = self.get_initial_surface_density()
        self.vrms    = self.get_initial_sigma_3D()
        self.t_cross = self.get_initial_t_cross()
        self.t_ff    = self.get_initial_t_ff()

    # ----------------------------- FUNCTIONS ---------------------------------

    # FROM INITIAL CLOUD PARAMETERS: surface density, R, vrms, Mach number.
    def get_initial_density(self, verbose=False):
        """
        Calculate the initial cloud density [code_mass/code_length**3].
        """
        rho = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        if verbose:
            print('Density: {0:.2f} Msun pc^-2'.format(rho))
        return rho


    def get_initial_surface_density(self, verbose=False):
        """
        Calculate the initial cloud surface density [code_mass/code_length**2].
        """
        Sigma = self.M / (np.pi * self.R**2)
        if verbose:
            print('Surface density: {0:.2f} Msun pc^-2'.format(Sigma))
        return Sigma

    # Initial 3D rms velocity.
    def get_initial_sigma_3D(self, verbose=False):
        """
        Calculate the initial 3D rms velocity [code_velocity].
        """
        sig_3D = np.sqrt((3.0 * self.alpha * self.G_code * self.M) / (5.0 * self.R))
        if verbose:
            print('sigma_3D = {0:.3f} m s^-1'.format(sig_3D))
        return sig_3D

    # Initial cloud trubulent crossing time.
    def get_initial_t_cross(self, verbose=False):
        """
        Calculate the initial turbulent crossing time as R/v_rms [code_time].
        """
        sig_3D  = self.get_initial_sigma_3D()
        t_cross = self.R / sig_3D
        if verbose:
            print('t_cross = {0:.3g} [code]'.format(t_cross))
        return t_cross

    # Initial cloud gravitational free-fall time.
    def get_initial_t_ff(self, verbose=False):
        """
        Calculate the initial freefall time [code_time].
        """
        rho  = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        t_ff = np.sqrt((3.0 * np.pi) / (32.0 * self.G_code * rho))
        if verbose:
            print('t_ff = {0:.3g} [code]'.format(t_ff))
        return t_ff

class Disk:
    """
    Class for analyzing disk properties within Snapshot.
    Initialize with HDF5 file created by Snapshot.get_disk().
    To-do: disk mass, radius, temperature, ionization fraction, etc.

      - USE_IDX: Use stored array indices, not particle IDs, for
        tracking disk particles if particle IDs are not unique.
    """

    def __init__(self, fname, cloud, USE_IDX=False, 
                 overwrite_snapdir=False, snapdir=None):
        
        self.fname   = fname
        self.cloud   = cloud
        self.USE_IDX = USE_IDX  # Use array indices instead of (non-unique) particle IDs.

        # Read from saved HDF5 file.
        with h5py.File(fname, 'r') as f:
            header = f['header']
            self.snapshot     = header.attrs['snapshot']       # Snapshot number.
            self.snapdir      = header.attrs['snapdir']        # Snapshot data directory.
            # Use if snapshots have been moved to a different directory than stored in Header.
            if overwrite_snapdir:
                #self.snapdir  = self.fname.split('snapshot_')[0]
                self.snapdir  = snapdir
            self.disk_type    = header.attrs['disk_type']
            self.disk_name    = header.attrs['disk_name']
            self.primary_sink = header.attrs['primary_sink']
            self.sink_ids     = header.attrs['sink_ids']
            self.num_sinks    = header.attrs['num_sinks']
            self.num_gas      = header.attrs['num_gas']
            self.n_H_min      = header.attrs['n_H_min']
            self.disk_dm      = header.attrs['disk_dm']
            self.disk_mass    = header.attrs['disk_mass']
            self.truncation_radius_AU = header.attrs['truncation_radius_AU']
            if header.attrs['truncated_by_neighbor'] == 'True':
                self.truncated_by_neighbor = True
            else:
                self.truncated_by_neighbor = False
            if USE_IDX:
                self.disk_ids = f.get('disk_idx')[:]
            else:
                self.disk_ids = f.get('disk_ids')[:]
              
        # Snapshot object.
        self.Snapshot = self.get_snapshot(self.cloud)  

        # Time.
        self.time = self.Snapshot.t

        # Unit conversions from parent snapshot.
        self.G_code          = self.Snapshot.G_code
        self.l_unit          = self.Snapshot.l_unit
        self.m_unit          = self.Snapshot.m_unit
        self.v_unit          = self.Snapshot.v_unit
        self.B_unit          = self.Snapshot.B_unit
        self.B_code          = self.Snapshot.B_code
        self.t_unit          = self.Snapshot.t_unit
        self.t_unit_myr      = self.Snapshot.t_unit_myr
        self.rho_unit        = self.Snapshot.rho_unit
        self.nH_unit         = self.Snapshot.nH_unit
        self.P_unit          = self.Snapshot.P_unit
        self.spec_L_unit     = self.Snapshot.spec_L_unit
        self.L_unit          = self.Snapshot.L_unit
        self.E_unit          = self.Snapshot.E_unit  # Internal energy per unit mass.
        self.pc_to_au        = 206265.0
        self.u_to_temp_units = self.Snapshot.u_to_temp_units
        #self.gizmo2gauss     = self.B_code / self.B_unit

        # Get array indices of disk particles from parent snapshot.
        if USE_IDX:
            self.idx_d = self.disk_ids
        else:
            self.idx_d = np.isin(self.Snapshot.p0_ids, self.disk_ids)

        # Disk particle attributes from parent snapshot.
        self.m     = self.Snapshot.p0_m[self.idx_d]      # Mass [code].
        self.x     = self.Snapshot.p0_x[self.idx_d]      # Coordinates [code].
        self.y     = self.Snapshot.p0_y[self.idx_d]
        self.z     = self.Snapshot.p0_z[self.idx_d]
        self.u     = self.Snapshot.p0_u[self.idx_d]      # Velocities [code].
        self.v     = self.Snapshot.p0_v[self.idx_d]
        self.w     = self.Snapshot.p0_w[self.idx_d]
        self.Bx    = self.Snapshot.p0_Bx[self.idx_d]     # Magnetic field [code].
        self.By    = self.Snapshot.p0_By[self.idx_d]
        self.Bz    = self.Snapshot.p0_Bz[self.idx_d]
        self.B_mag = self.Snapshot.p0_B_mag[self.idx_d]
        self.rho   = self.Snapshot.p0_rho[self.idx_d]    # Density [code].
        self.hsml  = self.Snapshot.p0_hsml[self.idx_d]   # Smoothing length.
        self.E_int = self.Snapshot.p0_E_int[self.idx_d]  # Internal energy per unit mass.
        self.P     = self.Snapshot.p0_P[self.idx_d]      # Pressure.
        self.cs    = self.Snapshot.p0_cs[self.idx_d]     # Sound speed.
        self.n_H   = self.Snapshot.p0_n_H[self.idx_d]    # H number density [cm^-3].
        self.Ne    = self.Snapshot.p0_Ne[self.idx_d]     # Electron abundance.

        # Mean molecular weight, temperature, dust temperature.
        self.mean_molecular_weight = self.Snapshot.p0_mean_molecular_weight[self.idx_d]
        self.temperature           = self.Snapshot.p0_temperature[self.idx_d]
        self.dust_temp             = self.Snapshot.p0_dust_temp[self.idx_d]

        # Non-ideal MHD coefficients.
        self.eta_O = self.Snapshot.p0_eta_O[self.idx_d]
        self.eta_H = self.Snapshot.p0_eta_H[self.idx_d]
        self.eta_A = self.Snapshot.p0_eta_A[self.idx_d]
        
        # OLD ATTRIBUTES:
        #self.n_He  = self.Snapshot.p0_n_He[self.idx_d]   # He number density [cm^-3].
        #self.H_mass_frac       = self.Snapshot.p0_H_mass_frac[self.idx_d]
        #self.He_mass_frac      = self.Snapshot.p0_He_mass_frac[self.idx_d]
        self.neutral_H_abundance = self.Snapshot.p0_neutral_H_abundance[self.idx_d]
        self.molecular_mass_frac = self.Snapshot.p0_molecular_mass_frac[self.idx_d]
        #self.total_metallicity = self.Snapshot.p0_total_metallicity[self.idx_d]

        # Disk + sink center-of-mass, angular momentum.
        m_cm, x_cm, v_cm  = self.Snapshot.system_center_of_mass(self.disk_ids, self.sink_ids, USE_IDX=USE_IDX)
        L_unit_vec, L_mag = self.Snapshot.get_net_ang_mom(self.disk_ids, self.sink_ids, USE_IDX=USE_IDX)
        x_hat, y_hat      = self.Snapshot._get_orthogonal_vectors(L_unit_vec)

        self.x_cm       = np.reshape(x_cm, (3, 1))
        self.v_cm       = np.reshape(v_cm, (3, 1))
        self.L_unit_vec = L_unit_vec
        self.L_mag      = L_mag

        # Rotation matrix to Cartesian coordinate frame with z || L_unit_vec.
        self.A = self.Snapshot._get_rotation_matrix(x_hat, y_hat, L_unit_vec)

        # Cartesian coordinates, velocities in system center-of-mass frame.
        self.X_cm = np.matmul(self.A, np.vstack((self.x, self.y, self.z)) - self.x_cm)
        self.V_cm = np.matmul(self.A, np.vstack((self.u, self.v, self.w)) - self.v_cm)

        # Rotate magnetic field to disk coordinate frame.
        self.B_cm = np.matmul(self.A, np.vstack((self.Bx, self.By, self.Bz)))

        # Convert to cylindrical coordinates.
        disk_r  = np.sqrt(self.X_cm[0, :]**2 + self.X_cm[1, :]**2)
        disk_t  = np.degrees(np.arctan(np.divide(self.X_cm[1, :], self.X_cm[0, :])))
        disk_vr = np.divide(np.multiply(self.V_cm[0, :], self.X_cm[0, :]) + \
                            np.multiply(self.V_cm[1, :], self.X_cm[1, :]), disk_r)
        disk_vt = np.divide(np.multiply(self.V_cm[1, :], self.X_cm[0, :]) - \
                            np.multiply(self.V_cm[0, :], self.X_cm[1, :]), disk_r)

        self.X_cyl = np.vstack((disk_r,  disk_t,  self.X_cm[2, :]))
        self.V_cyl = np.vstack((disk_vr, disk_vt, self.V_cm[2, :]))

        self.omega = np.divide((np.multiply(self.X_cm[0, :] * self.l_unit, self.V_cm[1, :] * self.v_unit) - \
                                np.multiply(self.X_cm[1, :] * self.l_unit, self.V_cm[0, :] * self.v_unit)),
                                (self.X_cm[0, :]**2 + self.X_cm[1, :]**2)* self.l_unit**2)

        # Convert magnetic field to cylindrical coordinates.
        disk_Br = np.divide(np.multiply(self.B_cm[0, :], self.X_cm[0, :]) + \
                            np.multiply(self.B_cm[1, :], self.X_cm[1, :]), disk_r)
        disk_Bt = np.divide(np.multiply(self.B_cm[1, :], self.X_cm[0, :]) - \
                            np.multiply(self.B_cm[0, :], self.X_cm[1, :]), disk_r)
        self.B_cyl = np.vstack((disk_Br, disk_Bt, self.B_cm[2, :]))

    def get_snapshot(self, cloud):
        fname_snap = os.path.join(self.snapdir, 'snapshot_{0:03d}.hdf5'.format(self.snapshot))
        return Snapshot(fname_snap, cloud)

    def get_radial_profile(self, y_vals, num_bins=100, USE_IDX=False):
        # Convert radial distance to AU.
        r_vals = self.X_cyl[0, :] * self.pc_to_au
        # Sort by radial distance.
        idx_sort       = np.argsort(r_vals)
        r_vals, y_vals = r_vals[idx_sort], y_vals[idx_sort]
        # Using equal-spaced bins.
        y_mean, y_bin_edges, _ = stats.binned_statistic(r_vals, y_vals, statistic='mean', bins=num_bins)
        # Bin centers.
        x_vals = (y_bin_edges[:-1] + y_bin_edges[1:])/2
        return x_vals, y_mean
    
    def write_to_file(self, fname, density_cut=False, n_H_min=1e9, verbose=True):
        
        mask_nH  = None
        idx_d    = np.isin(self.disk_ids, self.disk_ids)

        if density_cut:
            if verbose:
                print('Number of disk particles before density cut: {0:d}'.format(np.sum(idx_d)), flush=True)
            mask_nH  = (self.n_H >= n_H_min)
            #mask_all = np.union1d(idx_d, mask_nH)
            mask_all = np.logical_and(idx_d, mask_nH)
            idx_d    = mask_all
            if verbose:
                print('Number of disk particles after density cut: {0:d}'.format(np.sum(idx_d)), flush=True)
        
        f      = h5py.File(fname, 'w')
        header = f.create_dataset('header', (1,))
        
        header.attrs.create('primary_sink', self.primary_sink, dtype=int)
        header.attrs.create('sink_ids', self.sink_ids, dtype=int)
        header.attrs.create('n_H_min', self.n_H_min, dtype=float)
        header.attrs.create('truncation_radius_AU', self.truncation_radius_AU, dtype=float)
        
        header.attrs.create('time', self.time, dtype=float)
        header.attrs.create('m_unit', self.m_unit, dtype=float)
        header.attrs.create('l_unit', self.l_unit, dtype=float)
        header.attrs.create('v_unit', self.v_unit, dtype=float)
        header.attrs.create('t_unit', self.t_unit, dtype=float)
        header.attrs.create('x_cm', self.x_cm, dtype=float)
        header.attrs.create('v_cm', self.v_cm, dtype=float)
        header.attrs.create('L_vec', self.L_unit_vec * self.L_mag, dtype=float)
        header.attrs.create('rotation_matrix', self.A, dtype=float)
        
        # If USE_IDX, disk_ids are actually disk_idx.
        f.create_dataset('disk_ids', data=self.disk_ids[idx_d], dtype=int)
        f.create_dataset('mass', data=self.m[idx_d], dtype=float)
        f.create_dataset('X_orig', data=np.vstack((self.x, self.y, self.z))[:, idx_d], dtype=float)
        f.create_dataset('V_orig', data=np.vstack((self.u, self.v, self.w))[:, idx_d], dtype=float)
        f.create_dataset('B_orig', data=np.vstack((self.Bx, self.By, self.Bz))[:, idx_d], dtype=float)
        f.create_dataset('X_cm', data=self.X_cm[:, idx_d], dtype=float)
        f.create_dataset('V_cm', data=self.V_cm[:, idx_d], dtype=float)
        f.create_dataset('B_cm', data=self.B_cm[:, idx_d], dtype=float)
        f.create_dataset('X_cyl', data=self.X_cyl[:, idx_d], dtype=float)
        f.create_dataset('V_cyl', data=self.V_cyl[:, idx_d], dtype=float)
        f.create_dataset('B_cyl', data=self.B_cyl[:, idx_d], dtype=float)
        f.create_dataset('B_mag', data=self.B_mag[idx_d], dtype=float)
        f.create_dataset('rho', data=self.rho[idx_d], dtype=float)
        f.create_dataset('hsml', data=self.hsml[idx_d], dtype=float)
        f.create_dataset('E_int', data=self.E_int[idx_d], dtype=float)
        f.create_dataset('P', data=self.P[idx_d], dtype=float)
        f.create_dataset('cs', data=self.cs[idx_d], dtype=float)
        f.create_dataset('n_H', data=self.n_H[idx_d], dtype=float)
        f.create_dataset('Ne', data=self.Ne[idx_d], dtype=float)
        f.create_dataset('mean_molecular_weight', data=self.mean_molecular_weight[idx_d], dtype=float)
        f.create_dataset('temperature', data=self.temperature[idx_d], dtype=float)
        f.create_dataset('dust_temp', data=self.dust_temp[idx_d], dtype=float)
        f.create_dataset('eta_O', data=self.eta_O[idx_d], dtype=float)
        f.create_dataset('eta_H', data=self.eta_H[idx_d], dtype=float)
        f.create_dataset('eta_A', data=self.eta_A[idx_d], dtype=float)
        f.create_dataset('omega', data=self.omega[idx_d], dtype=float)
        
        f.close()
        
class Snapshot:
    """
    Class for reading gas/sink particle data from HDF5 snapshot files.
    Initialize with parent cloud.
      - USE_IDX: Use array indices, not particle IDs, for tracking disk
        particles if particle IDs are not unique.
    """

    def __init__(self, fname, cloud, B_unit=1e4, verbose=False, NMHD_version=5,
                 use_eos_substellar=False, USE_IDX=False):

        # Physical constants.
        self.PROTONMASS_CGS     = 1.6726e-24
        self.ELECTRONMASS_CGS   = 9.10953e-28
        self.BOLTZMANN_CGS      = 1.38066e-16
        self.HYDROGEN_MASSFRAC  = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS        = 2.9979e10
        self.HYDROGEN_MASSFRAC  = 0.76

        # Initial cloud parameters.
        self.fname   = fname
        self.snapdir = self.get_snapdir()
        self.Cloud   = cloud
        self.M0      = cloud.M      # Initial cloud mass, radius.
        self.R0      = cloud.R
        self.L0      = cloud.L      # Volume-equivalent length.
        self.alpha0  = cloud.alpha  # Initial virial parameter.
        
        # Version for calculating NMHD coefficients (5 = GIZMO 19f8fc81)
        self.NMHD_version = NMHD_version
        
        # If GIZMO was run with the EOS_SUBSTELLAR_ISM flag.
        self.use_eos_substellar = use_eos_substellar
        
        # Use array indices instead of particle IDs to uniquely track particles.
        self.USE_IDX = USE_IDX
        
        # Open HDF5 file.
        with h5py.File(fname, 'r') as f:
            header = f['Header']
            p0     = f['PartType0']
            
            # Do star particles exist in this snapshot?
            self.stars_exist = False
            if 'PartType5' in f:
                self.stars_exist = True
                p5 = f['PartType5']
                
            # Header attributes.
            self.box_size = header.attrs['BoxSize']
            self.num_p0   = header.attrs['NumPart_Total'][0]
            self.num_p5   = header.attrs['NumPart_Total'][5]
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

            # Other useful conversion factors.
            self.cm_to_AU = 6.6845871226706e-14
            self.cm_to_pc = 3.2407792896664e-19

            # Critical density for star formation
            self.sf_density_threshold = header.attrs['Density_Threshold_For_SF_CodeUnits']

            # PartType0 data.
            self.p0_ids   = p0['ParticleIDs'][()]         # Particle IDs.
            self.p0_cids  = p0['ParticleChildIDsNumber'][()]
            self.p0_m     = p0['Masses'][()]              # Masses.
            self.p0_rho   = p0['Density'][()]             # Density.
            self.p0_hsml  = p0['SmoothingLength'][()]     # Particle smoothing length.
            self.p0_E_int = p0['InternalEnergy'][()]      # Internal energy.
            self.p0_P     = p0['Pressure'][()]            # Pressure.
            self.p0_cs    = p0['SoundSpeed'][()]          # Sound speed.
            self.p0_x     = p0['Coordinates'][()][:, 0]   # Coordinates.
            self.p0_y     = p0['Coordinates'][()][:, 1]
            self.p0_z     = p0['Coordinates'][()][:, 2]
            self.p0_u     = p0['Velocities'][()][:, 0]    # Velocities.
            self.p0_v     = p0['Velocities'][()][:, 1]
            self.p0_w     = p0['Velocities'][()][:, 2]
            self.p0_Ne    = p0['ElectronAbundance'][()]   # Electron abundance.
            if 'MagneticField' in p0.keys():
                self.p0_Bx    = p0['MagneticField'][()][:, 0]
                self.p0_By    = p0['MagneticField'][()][:, 1]
                self.p0_Bz    = p0['MagneticField'][()][:, 2]
                self.p0_B_mag = np.sqrt(self.p0_Bx**2 + self.p0_By**2 + self.p0_Bz**2)
            else:
                self.p0_Bx    = np.zeros(len(self.p0_ids))
                self.p0_By    = np.zeros(len(self.p0_ids))
                self.p0_Bz    = np.zeros(len(self.p0_ids))
                self.p0_B_mag = np.zeros(len(self.p0_ids))
                
            self.p0_pot = p0['Potential'][()]             # Gravitational potential.
            
            # Hydrogen number density and total metallicity.
            self.p0_n_H  = (1.0 / self.PROTONMASS_CGS) * \
                            np.multiply(self.p0_rho * self.rho_unit, 1.0 - p0['Metallicity'][()][:, 0])
            self.p0_total_metallicity = p0['Metallicity'][()][:, 0]
            # Calculate mean molecular weight.
            self.p0_mean_molecular_weight = self.get_mean_molecular_weight(self.p0_ids)

            # Neutral hydrogen abundance, molecular mass fraction.
            self.p0_neutral_H_abundance = p0['NeutralHydrogenAbundance'][()]
            self.p0_molecular_mass_frac = p0['MolecularMassFraction'][()]
            
            # Calculate gas adiabatic index and temperature.
            '''
            fH, f, xe            = self.HYDROGEN_MASSFRAC, p0['MolecularMassFraction'][()], self.p0_Ne
            f_mono, f_di         = fH*(xe + 1.-f) + (1.-fH)/4., fH*f/2.
            gamma_mono, gamma_di = 5./3., 7./5.
            gamma                = 1. + (f_mono + f_di) / (f_mono/(gamma_mono-1.) + f_di/(gamma_di-1.))
            self.p0_temperature  = (gamma - 1.) * self.p0_mean_molecular_weight * \
                                    self.u_to_temp_units * self.p0_E_int
            '''
            
            # Gas temperature stored in snapshot.
            self.output_temperature = False
            if 'Temperature' in p0.keys():
                self.output_temperature   = True
                self.p0_temperature_GIZMO = p0['Temperature'][()]
            # Use approximation without sophisticated gamma_di treatment.
            else:
                self.p0_temperature_GIZMO = self.get_temperature(self.p0_ids, use_eos_substellar=False)
            self.p0_temperature = self.get_temperature(self.p0_ids, use_eos_substellar=self.use_eos_substellar)

            
            # Dust temperature.
            if 'Dust_Temperature' in p0.keys():
                self.p0_dust_temp = p0['Dust_Temperature'][()]
        
            # Get stored coefficients if HDF5 field exists.
            if 'MagneticField' in p0.keys():
                if 'NonidealDiffusivities' in p0.keys():
                    if verbose:
                        print('Reading NMHD coefficients from snapshot...', flush=True)
                    self.p0_eta_O = p0['NonidealDiffusivities'][()][:, 0]
                    self.p0_eta_H = p0['NonidealDiffusivities'][()][:, 1]
                    self.p0_eta_A = p0['NonidealDiffusivities'][()][:, 2]
                # Else, calculate non-ideal MHD coefficients.
                else:
                    if 'Dust_Temperature' in p0.keys():
                        if verbose:
                            print('Calculating NMHD coefficients from snapshot fields using version {0:d}...'.format(self.NMHD_version), flush=True)
                        eta_O, eta_H, eta_A = self.get_nonideal_MHD_coefficients(self.p0_ids, version=self.NMHD_version, use_eos_substellar=self.use_eos_substellar)
                    else:
                        eta_O, eta_H, eta_A = 0.0, 0.0, 0.0
                    self.p0_eta_O = eta_O
                    self.p0_eta_H = eta_H
                    self.p0_eta_A = eta_A
            else:
                self.p0_eta_O = np.zeros(len(self.p0_ids))
                self.p0_eta_H = np.zeros(len(self.p0_ids))
                self.p0_eta_A = np.zeros(len(self.p0_ids))
            
            if 'TimeStep' in p0.keys():
                self.p0_timestep = p0['TimeStep'][()]

            # For convenience, coordinates and velocities in a (n_gas, 3) array.
            self.p0_coord = np.vstack((self.p0_x, self.p0_y, self.p0_z)).T
            self.p0_vel   = np.vstack((self.p0_u, self.p0_v, self.p0_w)).T
            self.p0_mag   = np.vstack((self.p0_Bx, self.p0_By, self.p0_Bz)).T

            # PartType5 data.
            if self.stars_exist:
                # Particle IDs.
                self.p5_ids = p5['ParticleIDs'][()]               # Particle IDs.
                self.p5_m   = p5['Masses'][()]                    # Masses.
                self.p5_x   = p5['Coordinates'][()][:, 0]         # Coordinates.
                self.p5_y   = p5['Coordinates'][()][:, 1]
                self.p5_z   = p5['Coordinates'][()][:, 2]
                self.p5_u   = p5['Velocities'][()][:, 0]          # Velocities.
                self.p5_v   = p5['Velocities'][()][:, 1]
                self.p5_w   = p5['Velocities'][()][:, 2]
                self.p5_lx  = p5['BH_Specific_AngMom'][()][:, 0]  # Specific angular momentum.
                self.p5_ly  = p5['BH_Specific_AngMom'][()][:, 1]
                self.p5_lz  = p5['BH_Specific_AngMom'][()][:, 2]

                self.p5_coord = np.vstack((self.p5_x, self.p5_y, self.p5_z)).T
                self.p5_vel   = np.vstack((self.p5_u, self.p5_v, self.p5_w)).T
                
                self.p5_pot = p5['Potential'][()]                 # Gravitational potential.
                
                self.p5_bhmass            = p5['BH_Mass'][()]
                self.p5_bhmass_alpha_disk = p5['BH_Mass_AlphaDisk'][()]
                self.p5_bhmdot            = p5['BH_Mdot'][()]

                # Sink particle attributes.
                self.p5_sink_radius      = p5['SinkRadius'][()]
                self.p5_accretion_length = p5['BH_AccretionLength'][()]
                self.p5_sf_time          = p5['StellarFormationTime'][()]

        # Initial t_cross, t_ff, 3D rms velocity, surface density.
        self.t_cross0 = cloud.t_cross
        self.t_ff0    = cloud.t_ff
        self.vrms0    = cloud.vrms
        self.rho0     = cloud.rho
        self.Sigma0   = cloud.Sigma

    # ----------------------------- FUNCTIONS ---------------------------------
    
    # Try to get snapshot number from filename.
    def get_i(self):
        return int(self.fname.split('snapshot_')[1].split('.hdf5')[0])

    # Try to get snapshot datadir from filename.
    def get_snapdir(self):
        return self.fname.split('snapshot_')[0]

    def _get_center_of_mass(self, p_type, p_ids, USE_IDX=False):
        """
        Calculates center of mass for (subset of) gas/sink particles.
        Parameters:
            - p_type: particle type; gas = 'PartType0', sink = 'PartType5'.
            - p_ids: particle IDs of particles to include in center of mass
            calculation.
            - USE_IDX: p_ids are actually array indices of particles.
        Returns: M, cm_x, cm_v
            - M: total mass (scalar)
            - cm_x: center of mass position (array)
            - cm_v: center of mass velocity (array)
        """
        if p_type == 'PartType0':
            if USE_IDX:
                idx_p = p_ids
            else:
                idx_p = np.isin(self.p0_ids, p_ids)
            m, M    = self.p0_m[idx_p], np.sum(self.p0_m[idx_p])
            x, y, z = self.p0_x[idx_p], self.p0_y[idx_p], self.p0_z[idx_p]
            u, v, w = self.p0_u[idx_p], self.p0_v[idx_p], self.p0_w[idx_p]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
            if USE_IDX:
                idx_p = p_ids
            else:
                idx_p   = np.isin(self.p5_ids, p_ids)
            m, M    = self.p5_m[idx_p], np.sum(self.p5_m[idx_p])
            x, y, z = self.p5_x[idx_p], self.p5_y[idx_p], self.p5_z[idx_p]
            u, v, w = self.p5_u[idx_p], self.p5_v[idx_p], self.p5_w[idx_p]
        else:
            return None

        x_cm = np.sum(np.multiply(m, x))/M; u_cm = np.sum(np.multiply(m, u))/M
        y_cm = np.sum(np.multiply(m, y))/M; v_cm = np.sum(np.multiply(m, v))/M
        z_cm = np.sum(np.multiply(m, z))/M; w_cm = np.sum(np.multiply(m, w))/M

        cm_x = np.asarray([x_cm, y_cm, z_cm])
        cm_v = np.asarray([u_cm, v_cm, w_cm])

        return M, cm_x, cm_v

    def _two_body_center_of_mass(self, m1, x1, v1, m2, x2, v2):
        """
        Calculates center of mass for a two-body system.
        Parameters:
            - m1, m2: masses
            - x1, x2: position (array)
            - v1, v2" velocity (array)
        Returns: M, cm_x, cm_v
            - M: total mass (scalar)
            - cm_x: center of mass position (array)
            - cm_v: center of mass velocity (array)
        """
        m = m1 + m2
        cm_x = (m1 * x1[0] + m2 * x2[0]) / m
        cm_y = (m1 * x1[1] + m2 * x2[1]) / m
        cm_z = (m1 * x1[2] + m2 * x2[2]) / m

        cm_u = (m1 * v1[0] + m2 * v2[0]) / m
        cm_v = (m1 * v1[1] + m2 * v2[1]) / m
        cm_w = (m1 * v1[2] + m2 * v2[2]) / m

        return m, np.asarray([cm_x, cm_y, cm_z]), np.asarray([cm_u, cm_v, cm_w])

    # Get center of mass of (subset of) gas particles.
    def gas_center_of_mass(self, gas_ids, USE_IDX=False):
        """
        Calculates center of mass for (subset of) gas particles.
        Parameters:
            - gas_ids: particle IDs of gas particles to include in center of
            mass calculation.
        Returns: M, cm_x, cm_v
            - M: total gas mass (scalar)
            - cm_x: gas center of mass position (array)
            - cm_v: gas center of mass velocity (array)
        """
        M, cm_x, cm_v = self._get_center_of_mass('PartType0', gas_ids, USE_IDX=USE_IDX)
        return M, cm_x, cm_v

    # Get center of mass of (subset of) sink particles.
    def sink_center_of_mass(self, sink_ids):
        """
        Calculates center of mass for (subset specified by sink_ids of) sink
        particles.
        Parameters:
            - sink_ids: particle IDs of sink particles to include in center
            of mass calculation.
        Returns: M, cm_x, cm_v
            - M: total sink mass (scalar)
            - cm_x: sink center of mass position (array)
            - cm_v: sink center of mass velocity (array)
        """
        M, cm_x, cm_v = self._get_center_of_mass('PartType5', sink_ids)
        return M, cm_x, cm_v

    # Get center of mass of combined gas/sink particles system.
    def system_center_of_mass(self, gas_ids, sink_ids, USE_IDX=False):
        """
        Calculates center of mass for system consisting of both sink and gas
        particles.
        Parameters:
            - gas_ids: particle IDs of gas particles to include in center
            of mass calculation.
            - sink_ids: particle IDs of sink particles to include in center
            of mass calculation.
        Returns:
            - M: total sink + gas mass (scalar)
            - cm_x: sink + gas center of mass position (array)
            - cm_v: sink + gas center of mass velocity (array)
        """
        M1, x1, v1 = self.gas_center_of_mass(gas_ids, USE_IDX=USE_IDX)
        M2, x2, v2 = self.sink_center_of_mass(sink_ids)
        M, cm_x, cm_v = self._two_body_center_of_mass(M1, x1, v1, M2, x2, v2)
        return M, cm_x, cm_v

    # Get particle mass/position/velocity.
    def _get_particle_kinematics(self, p_type, p_ids, USE_IDX=False):
        """
        Returns masses, positions, and velocities of specified particles.
        Parameters:
            - p_type: particle type; gas = 'PartType0', sink = 'PartType5'.
            - p_ids: particle IDs of specified particles.
            - USE_IDX: p_ids are actually array indices of particles.
        Returns: m, x, v.
            - m: masses of specified particles (1D array)
            - x: positions of specified particles (3D array)
            - v: velocities of specified particles (3D array)
        """
        if p_type == 'PartType0':
            if USE_IDX:
                idx_p = p_ids
            else:
                if np.isscalar(p_ids):
                    idx_p = np.where(self.p0_ids == p_ids)[0][0]
                else:
                    idx_p = np.isin(self.p0_ids, p_ids)
            m       = self.p0_m[idx_p]
            x, y, z = self.p0_x[idx_p], self.p0_y[idx_p], self.p0_z[idx_p]
            u, v, w = self.p0_u[idx_p], self.p0_v[idx_p], self.p0_w[idx_p]
        elif p_type == 'PartType5':
            if USE_IDX:
                idx_p = p_ids
            else:
                if self.num_p5 == 0:
                    return None
                if np.isscalar(p_ids):
                    idx_p = np.where(self.p5_ids == p_ids)[0][0]
                else:
                    idx_p = np.isin(self.p5_ids, p_ids)
            m       = self.p5_m[idx_p]
            x, y, z = self.p5_x[idx_p], self.p5_y[idx_p], self.p5_z[idx_p]
            u, v, w = self.p5_u[idx_p], self.p5_v[idx_p], self.p5_w[idx_p]
        else:
            return None
        if np.isscalar(p_ids):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T

    # Get relative particle mass/position/velocity.
    def _get_particle_relative_kinematics(self, p_type, p_ids, point_x, point_v, USE_IDX=False):
        """
        Returns masses, positions, and velocities of specified particles relative
        to the point specified by point_x, point_v.
        Parameters:
            - p_type: particle type; gas = 'PartType0', sink = 'PartType5'.
            - p_ids: particle IDs of specified particles.
            - point_x: [array-like] (x, y, z) position specifying origin of
            new coordinate system.
            - point_v: [array-like] (u, v, w) velocity specifying origin of
            new coordinate system.
        Returns: m, x, v.
            - m: masses of specified particles (1D array)
            - x: relative positions of specified particles (3D array)
            - v: relative velocities of specified particles (3D array)
        """
        x0, y0, z0 = point_x[0], point_x[1], point_x[2]
        u0, v0, w0 = point_v[0], point_v[1], point_v[2]
        if p_type == 'PartType0':
            if USE_IDX:
                idx_p = p_ids
            else:
                if np.isscalar(p_ids):
                    idx_p = np.where(self.p0_ids == p_ids)[0][0]
                else:
                    idx_p = np.isin(self.p0_ids, p_ids)
            m       = self.p0_m[idx_p]
            x, y, z = self.p0_x[idx_p] - x0, self.p0_y[idx_p] - y0, self.p0_z[idx_p] - z0
            u, v, w = self.p0_u[idx_p] - u0, self.p0_v[idx_p] - v0, self.p0_w[idx_p] - w0
        elif p_type == 'PartType5':
            if USE_IDX:
                idx_p = p_ids
            else:
                if self.num_p5 == 0:
                    return None
                if np.isscalar(p_ids):
                    idx_p = np.where(self.p5_ids == p_ids)[0][0]
                else:
                    idx_p = np.isin(self.p5_ids, p_ids)
            m       = self.p5_m[idx_p]
            x, y, z = self.p5_x[idx_p] - x0, self.p5_y[idx_p] - y0, self.p5_z[idx_p] - z0
            u, v, w = self.p5_u[idx_p] - u0, self.p5_v[idx_p] - v0, self.p5_w[idx_p] - w0
        else:
            return None
        if np.isscalar(p_ids):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T

    # Get gas particle mass, position, velocity.
    def get_gas_kinematics(self, gas_ids, USE_IDX=False):
        """
        Returns masses, positions, and velocities of specified gas particles.
        Parameters:
            - gas_ids: particle IDs of specified gas particles.
        Returns: m, x, v.
            - m: masses of specified gas particles (1D array)
            - x: positions of specified gas particles (3D array)
            - v: velocities of specified gas particles (3D array)
        """
        m, x, v = self._get_particle_kinematics('PartType0', gas_ids, USE_IDX=USE_IDX)
        return m, x, v

    # Get gas particle mass, position, velocity relative to x0, v0.
    def get_gas_relative_kinematics(self, gas_ids, x0, v0, USE_IDX=False):
        """
        Returns masses, positions, and velocities of specified gas particles
        relative to the point specified by point_x, point_v.
        Parameters:
            - gas_ids: particle IDs of specified gas particles.
            - point_x: [array-like] (x, y, z) position specifying origin of
            new coordinate system.
            - point_v: [array-like] (u, v, w) velocity specifying origin of
            new coordinate system.
        Returns: m, x, v.
            - m: masses of specified gas particles (1D array)
            - x: relative positions of specified gas particles (3D array)
            - v: relative velocities of specified gas particles (3D array)
        """
        m, x, v = self._get_particle_relative_kinematics('PartType0', gas_ids, x0, v0, USE_IDX=USE_IDX)
        return m, x, v

    # Get sink particle mass, position, velocity.
    def get_sink_kinematics(self, sink_ids):
        """
        Returns masses, positions, and velocities of specified sink particles.
        Parameters:
            - sink_ids: particle IDs of specified sink particles.
        Returns: m, x, v.
            - m: masses of specified sink particles (1D array)
            - x: positions of specified sink particles (3D array)
            - v: velocities of specified sink particles (3D array)
        """
        m, x, v = self._get_particle_kinematics('PartType5', sink_ids)
        return m, x, v

    # Get sink particle mass, position, velocity relative to x0, v0.
    def get_sink_relative_kinematics(self, sink_ids, x0, v0):
        """
        Returns masses, positions, and velocities of specified sink particles
        relative to the point specified by point_x, point_v.
        Parameters:
            - sink_ids: particle IDs of specified sink particles.
            - point_x: [array-like] (x, y, z) position specifying origin of
            new coordinate system.
            - point_v: [array-like] (u, v, w) velocity specifying origin of
            new coordinate system.
        Returns: m, x, v.
            - m: masses of specified sink particles (1D array)
            - x: relative positions of specified sink particles (3D array)
            - v: relative velocities of specified sink particles (3D array)
        """
        m, x, v = self._get_particle_relative_kinematics('PartType5', sink_ids, x0, v0)
        return m, x, v

    # Return particle IDs of gas above density threshhold.
    def get_density_cut(self, rho_cut):
        """
        Returns particle IDs of gas particles with density above the density
        threshhold specified by rho_cut.
        Parameters:
            - rho_cut: density threshhold [g/cm^3]
        Returns:
            - particle IDs of gas particles above density threshhold (array).
        """
        cut = (self.p0_rho * self.rho_unit) > rho_cut
        return self.p0_ids[cut]

    # Get 3D rms velocity dispersion of gas_ids.
    def get_sigma_3D_gas(self, gas_ids):
        """
        Calculates the 3D rms velocity dispersion of the specified gas
        particles.
        Parameters:
            - gas_ids: particle IDs of gas particles to include in the
            calculation.
        Returns:
            - sigma_3D: 3D rms velocity dispersion [code units]
        """
        idx_g   = np.isin(self.p0_ids, gas_ids)
        m       = self.p0_m[idx_g]
        u, v, w = self.p0_u[idx_g], self.p0_v[idx_g], self.p0_w[idx_g]
        sigma_3D = np.sqrt((self.weight_std(u, m)**2.0 + self.weight_std(v, m)**2.0 + \
                            self.weight_std(w, m)**2.0))
        return sigma_3D

    # Get 3D rms Mach number (c_s in cm s^-1) of gas_ids.
    def get_Mach_gas(self, gas_ids, c_s=2.0e4):
        """
        Calculates the Mach number of the specified gas particles, assuming
        a constant sound speed c_s. (TO-DO: use actual sound speed.)
        Parameters:
            - gas_ids: particle IDs of gas particles to include in the
            calculation.
            - c_s: sound speed [cm/s].
        Returns:
            - Mach number.
        """
        sigma_3D = self.get_sigma_3D_gas(gas_ids)
        c_s      = c_s / self.v_unit
        Mach     = sigma_3D / c_s
        return Mach

    # Get turbulent virial parameter from 3D rms velocity.
    def get_alpha_gas(self, gas_ids):
        """
        Calculates the turbulent virial parameter of the specified gas
        particles.
        Parameters:
            - gas_ids: particle IDs of gas particles to include in the
            calculation.
        Returns:
            - alpha: turbulent virial parameter = (5*sigma_3D^2*R)/(3*GM)
        """
        sigma_3D = self.get_sigma_3D_gas(gas_ids)
        alpha    = (5.0 * sigma_3D**2 * self.R0) / (3.0 * self.G_code * self.M0)
        return alpha
    
    # Get RMS distance to center of mass.
    def get_rms_radius(self, gas_ids):
    
        idx_g               = np.isin(self.p0_ids, gas_ids)
        m_vals              = self.p0_m[idx_g]
        M_cm, x_cm, v_cm    = self.gas_center_of_mass(gas_ids)
        x0, y0, z0          = x_cm[0], x_cm[1], x_cm[2]
        x_rel, y_rel, z_rel = self.p0_x[idx_g] - x0, self.p0_y[idx_g] - y0, self.p0_z[idx_g] - z0        

        r_vals   = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        idx_sort = np.argsort(r_vals)
        m_vals, r_vals = m_vals[idx_sort], r_vals[idx_sort]
        r_rms = self.weight_std(r_vals, m_vals)
        return r_rms

    # Get half-mass radius of selected gas particles.
    def get_half_mass_radius(self, gas_ids, tol=0.5, verbose=False):
        
        # Initial cloud mass, half-mass radius,
        M, R = self.M0, self.R0
        m_half  = M / 2.0               # Or use M?
        r_half  = (0.5)**(1.0/3.0) * R  # If uniform density.
        r_guess = r_half

        # Center of mass, indices of selected gas particles.
        M_cm, x_cm, v_cm = self.gas_center_of_mass(gas_ids)
        idx_g            = np.isin(self.p0_ids, gas_ids)

        # Center of mass; coordinates of selected gas particles relative to center of mass.
        x0, y0, z0          = x_cm[0], x_cm[1], x_cm[2]
        x_rel, y_rel, z_rel = self.p0_x[idx_g] - x0, self.p0_y[idx_g] - y0, self.p0_z[idx_g] - z0
 
        # Masses and distances os selected particles.
        m_vals = self.p0_m[idx_g]
        r_vals = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2) 

        # Sort gas particles according to distancen from center of mass.
        idx_sort       = np.argsort(r_vals)
        m_vals, r_vals = m_vals[idx_sort], r_vals[idx_sort] 

        # Get current enclosed mass.
        cut = r_vals < r_guess
        m_enc = np.sum(m_vals[cut])

        while True:
            if verbose:
                print('Current:\tr_half = {0:.5f} pc\tm_enc = {1:.5f} Msun'.format(r_guess, m_enc))
            if m_enc <= m_half + tol and m_enc >= m_half - tol:
                if verbose:
                    print('Found half-mass radius (r = {0:.5f} pc)\t(m_enc = {1:.3f})'.format(r_guess, m_enc))
                return r_guess
            elif m_enc >= m_half + tol:
                r_guess *= 0.9
                cut = r_vals < r_guess
                m_enc = np.sum(m_vals[cut])
            else:
                r_guess *= 1.1
                cut = r_vals < r_guess
                m_enc = np.sum(m_vals[cut])

    # Sort particles by distance to specified coordinates.
    def _sort_particles_by_distance_to_point(self, p_type, p_ids, point, USE_IDX=False):
        """
        Sorts the specified particles according to distance from the specified
        point.
        Parameters:
            - p_type: particle type; gas = 'PartType0', sink = 'PartType5'.
            - p_ids: particle IDs of specified particles.
            - point: [array-like] (x, y, z) coordinates specifying origin of
            new coordinate system.
        Returns:
            - r_sort: [1D array] sorted distances to specified point, in
            increasing order.
            - ids_sort: [1D array] sorted particle IDs.
            - idx_sort: [1D array] indices corresponding to sort; e.g.
            p_ids[idx_sort] = ids_sort.
        """
        x0, y0, z0 = point[0], point[1], point[2]
        if p_type == 'PartType0':
            if USE_IDX:
                idx_p = p_ids
            else:
                idx_p = np.isin(self.p0_ids, p_ids)
            ids_p = self.p0_ids[idx_p]
            x1, y1, z1 = self.p0_x[idx_p], self.p0_y[idx_p], self.p0_z[idx_p]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
            if USE_IDX:
                idx_p = p_ids
            else:
                idx_p = np.isin(self.p5_ids, p_ids)
            ids_p = self.p5_ids[idx_p]
            x1, y1, z1 = self.p5_x[idx_p], self.p5_y[idx_p], self.p5_z[idx_p]
        else:
            return None
        x, y, z  = x1 - x0, y1 - y0, z1 - z0
        r        = np.sqrt(x**2 + y**2 + z**2)
        idx_sort = np.argsort(r)
        if USE_IDX:
            return r[idx_sort], idx_p[idx_sort], idx_sort
        else:
            return r[idx_sort], ids_p[idx_sort], idx_sort

    # Sort gas particles by distance to point (x, y, z).
    def sort_gas_by_distance_to_point(self, gas_ids, point, USE_IDX=False):
        """
        Sorts the specified gas particles according to distance from the
        specified point.
        Parameters:
            - gas_ids: particle IDs of the specified gas particles.
            - point: [array-like] (x, y, z) coordinates specifying origin of
            new coordinate system.
        Returns:
            - r_sort: [1D array] sorted distances to specified point, in
            increasing order.
            - ids_sort: [1D array] sorted gas particle IDs.
            - idx_sort: [1D array] indices corresponding to sort; e.g.
            gas_ids[idx_sort] = ids_sort.
        """
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType0', gas_ids, point, USE_IDX=USE_IDX)
        return r, ids, idx

    # Sort sink particles by distance to point (x, y, z).
    def sort_sinks_by_distance_to_point(self, sink_ids, point):
        """
        Sorts the specified sink particles according to distance from the
        specified point.
        Parameters:
            - sink_ids: particle IDs of the specified sink particles.
            - point: [array-like] (x, y, z) coordinates specifying origin of
            new coordinate system.
        Returns:
            - r_sort: [1D array] sorted distances to specified point, in
            increasing order.
            - ids_sort: [1D array] sorted sink particle IDs.
            - idx_sort: [1D array] indices corresponding to sort; e.g.
            sink_ids[idx_sort] = ids_sort.
        """
        if self.num_p5 == 0:
            return
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType5', sink_ids, point)
        return r, ids, idx

    # Sort gas particles by distance to sink.
    def sort_gas_by_distance_to_sink(self, gas_ids, sink_id, USE_IDX=False):
        """
        Sorts the specified gas particles according to distance from the
        specified sink particle.
        Parameters:
            - gas_ids: particle IDs of the specified gas particles.
            - sink_id: particle ID of the specified sink particle.
        Returns:
            - r_sort: [1D array] sorted distances to the specified sink
            particle, in increasing order.
            - ids_sort: [1D array] sorted gas particle IDs.
            - idx_sort: [1D array] indices corresponding to sort; e.g.
            gas_ids[idx_sort] = ids_sort.
        """
        if self.num_p5 == 0:
            return None
        ms, xs, vs  = self.get_sink_kinematics(sink_id)
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType0', gas_ids, xs, USE_IDX=USE_IDX)
        return r, ids, idx

    # Sort specified sink particles by distance to primary sink particle.
    def sort_sinks_by_distance_to_sink(self, sink_ids, primary_sink_id):
        """
        Sorts sink particles specified by sink_ids according to distance from
        the sink particle specified by primary_sink_id.
        Parameters:
            - sink_ids: particle IDs of sinks to sort by distance to
            primary sink particle.
            - primary_sink_id: sort by distance to this sink particle.
        Returns:
            - r_sort: [1D array] sorted distances to the primary sink
            particle, in increasing order.
            - ids_sort: [1D array] sorted sink particle IDs.
            - idx_sort: [1D array] indices corresponding to sort.
        """
        # Get kinematics of primary sink particle.
        ms, xs, vs  = self.get_sink_kinematics(primary_sink_id)
        # Determine particle IDs of other sink particles.
        mask_r   = np.isin(self.p5_ids, sink_ids)
        ids_rest = self.p5_ids[mask_r]
        # Exclude primary_sink if in sink_ids.
        if np.sum(np.isin(ids_rest, primary_sink_id)) == 1:
            mask_s   = np.isin(ids_rest, primary_sink_id, invert=True)
            ids_rest = ids_rest[mask_s]
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType5', ids_rest, xs)
        return r, ids, idx

    # Get specific angular momentum of gas particles about center sink particle.
    def get_net_ang_mom(self, gas_ids, sink_ids=None, USE_IDX=False):
        # Just use gas particles to get center of mass, angular momentum.
        if sink_ids is None:
            m_cm, x_cm, v_cm = self.gas_center_of_mass(gas_ids, USE_IDX=USE_IDX)
        else:
            m_cm, x_cm, v_cm = self.system_center_of_mass(gas_ids, sink_ids, USE_IDX=USE_IDX)
        m_g, x_g, v_g    = self.get_gas_relative_kinematics(gas_ids, x_cm, v_cm, USE_IDX=USE_IDX)
        ang_mom_vec      = np.sum(np.cross(x_g, v_g), axis=0)
        ang_mom_mag      = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag

    # Get orthogonal (x, y) unit vectors given vector z.
    def _get_orthogonal_vectors(self, z):
        # Use original x basis vector for consistency.
        x = np.asarray([1.0, 0.0, 0.0])
        x -= x.dot(z) * z / np.linalg.norm(z)**2
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return x, y

    # Get rotation matrix corresponding to new coordinate system.
    def _get_rotation_matrix(self, x, y, z):
        A = np.stack((x, y, z), axis=0)
        return A

    # Identify gas particles in gas_ids belonging to disk around sink_ids.
    def get_disk(self, sink_ids, gas_ids, r_max_AU=500.0, n_H_min=1e9, verbose=False,
                 disk_name='', save_disk=False, diskdir=None, get_L_vec=False, USE_IDX=False,
                 check_nearest_neighbor=True, set_rmax_half=False, return_masks=False):
        """
        Identifies subset of specified gas particles belonging to disk
        around specified sink particles based upon the following checks:
          - Gas is rotationally supported (v_phi > 2 v_r).
          - Gas is close to hydrostatic equilibrium (v_phi > 2 v_z).
          - Rotational support exceeds thermal support (0.5 * rho * v_phi^2 > 2 P).
          - Gas exceeds minimum number density threshold (n_H_min).
        Parameters:
            - sink_ids: particle IDs of the specified sink particles.
            - gas_ids: particle IDs of the specified gas particles.
            - r_max_AU: max disk radius in AU about sink center of mass.
            - n_H_min: minimum number density threshold.
        Returns:
            - disk_ids [1D array]: particle IDs of gas particles belonging to
            disk.
        """
        is_single_disk    = True
        num_sinks_in_disk = 1
        # Getting disk around single sink particle?
        if np.isscalar(sink_ids):
            primary_sink   = sink_ids
            if disk_name == '':
                disk_name      = 'disk_single_{0:d}'.format(sink_ids)
            if verbose:
                print('Getting single disk (sink ID {0:d})...'.format(sink_ids), flush=True)
        # Getting disk around multiple sink particles? Label by most massive sink.
        else:
            is_single_disk    = False
            sink_ids_mask     = np.isin(self.p5_ids, sink_ids)
            sink_masses       = self.p5_m[sink_ids_mask]
            sink_ids_in_disk  = self.p5_ids[sink_ids_mask]
            num_sinks_in_disk = len(sink_masses)
            primary_sink      = sink_ids_in_disk[np.argmax(sink_masses)]
            if disk_name == '':
                disk_name         = 'disk_multi_{0:d}_{1:d}'.format(num_sinks_in_disk, primary_sink)
            if verbose:
                print('Getting multiple disk (primary sink ID {0:d}; {1:d} sinks total)...'.format(primary_sink, num_sinks_in_disk), flush=True)
        # Sink system center of mass, position, velocity.
        m, sink_x, sink_v = self.sink_center_of_mass(sink_ids)

        # Initial Boolean mask specified by either gas_ids or array indices.
        if USE_IDX:
            mask_init = np.isin(np.arange(len(self.p0_ids)), gas_ids)
        else:
            mask_init = np.isin(self.p0_ids, gas_ids)
        if verbose:
            print('...size(mask_init) = {0:d}'.format(np.sum(mask_init)), flush=True)
        # Convert r_max to cgs and code units.
        r_max_cgs  = r_max_AU / self.cm_to_AU
        r_max_code = r_max_cgs / self.l_unit
        if verbose:
            print('Initial r_max = {0:.1f} AU.'.format(r_max_AU), flush=True)
        # Check that r_max is less than distance from center of mass to nearest non-disk sink particle.
        r_max_truncated_by_neighbor = False
        r_max = r_max_code
        if (self.num_p5 > 1) and (check_nearest_neighbor):
            if verbose:
                print('Checking for nearby sinks within {0:.1f} AU...'.format(r_max_AU), flush=True)
            if is_single_disk:
                r, ids, idx = self.sort_sinks_by_distance_to_sink(self.p5_ids, primary_sink)
            else:
                other_sink_ids = self.p5_ids[np.isin(self.p5_ids, sink_ids, invert=True)]
                if len(other_sink_ids) == 0:
                    r = np.asarray([10.0*r_max_code])
                else:
                    # Sort other sinks by distance to sink system center of mass.
                    r, ids, idx = self.sort_sinks_by_distance_to_point(other_sink_ids, sink_x)
                    # Sort other sinks by distance to primary (most massive) sink.
                    #r, ids, idx = self.sort_sinks_by_distance_to_sink(other_sink_ids, primary_sink)
            # Minimum distance to non-disk sink particles [code units].
            r_near_code = r[0]
            r_near_cgs  = r_near_code * self.l_unit
            r_near_AU   = r_near_cgs * self.cm_to_AU
            if verbose:
                print('Found nearest sink at r = {0:.1f} AU.'.format(r_near_AU), flush=True)
            if set_rmax_half:
                if verbose:
                    print('Using half distance to nearest sink particle...', flush=True)
                r_near_code /= 2.0
            if r_max_code < r_near_code:
                r_max = r_max_code
                if verbose:
                    print('Keeping r_max = {0:.1f} AU.'.format(r_max_AU), flush=True)
            else:
                r_max = r_near_code
                if verbose:
                    r_max_truncated_by_neighbor = True
                    print('Updating r_max = {0:.1f} AU.'.format(r_near_AU), flush=True)
        # Sort gas particles by distance to sink system center of mass.
        r_sort, ids_sort, idx_sort = self.sort_gas_by_distance_to_point(gas_ids, sink_x, USE_IDX=USE_IDX)
        r_max_idx                  = np.argmax(r_sort > r_max)
        ids_in_sphere              = ids_sort[:r_max_idx]
        if USE_IDX:
            mask_r_max      = np.isin(np.arange(len(self.p0_ids)), ids_in_sphere)
            mask_r_max      = np.logical_and(mask_init, mask_r_max)
            g_ids_in_sphere = np.arange(len(self.p0_ids))[mask_r_max]
        else:
            mask_r_max      = np.isin(self.p0_ids, ids_in_sphere)
            mask_r_max      = np.logical_and(mask_init, mask_r_max)
            g_ids_in_sphere = self.p0_ids[mask_r_max]
        if verbose:
            print('...size(mask_r_max) = {0:d}'.format(np.sum(mask_r_max)), flush=True)
            print(g_ids_in_sphere)
        # Transform to sink-centered coordinate system; z axis parallel to net ang. momentum.
        if verbose:
            print('Transforming to sink-centered disk coordinate system...', flush=True)
        # Get angular momentum vector from all gas particles within r_max AU of sink particles.
        L_unit_vec, L_mag = self.get_net_ang_mom(g_ids_in_sphere, sink_ids=sink_ids, USE_IDX=USE_IDX)
        # Define rotation matrix for coordinate system transformation.
        x_vec, y_vec = self._get_orthogonal_vectors(L_unit_vec)
        A            = self._get_rotation_matrix(x_vec, y_vec, L_unit_vec)
        # Original coordinates (within r_max AU sphere); array shape = (N, 3).
        x_orig = self.p0_coord[mask_r_max]
        v_orig = self.p0_vel[mask_r_max]
        # Coordinates relative to sink particle coordiantes; array shape = (N, 3).
        x_centered = x_orig - sink_x.T
        v_centered = v_orig - sink_v.T
        # Rotated to angular momentum frame; array shape = (3, N).
        x_rot = np.matmul(A, x_centered.T)
        v_rot = np.matmul(A, v_centered.T)
        # Cartesian coordinate system.
        x, y, z = x_rot[0, :], x_rot[1, :], x_rot[2, :]
        u, v, w = v_rot[0, :], v_rot[1, :], v_rot[2, :]
        # Cylindrical coordinate system.
        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan(np.divide(y, x))
        # Radial/azimuthal/z-velocity for disk membership checks.
        v_r   = np.divide(np.multiply(x, u) + np.multiply(y, v), r)
        v_phi = np.divide((np.multiply(x, v) - np.multiply(y, u)), r)
        v_z   = w
        if verbose:
            print('Checking rotational support (v_phi > 2 * v_r)...', flush=True)
        # Gas is rotationally supported?
        is_rotating = np.greater(np.abs(v_phi), 2.0 * np.abs(v_r))
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_rotating)), flush=True)
            print('Checking hydrostatic equilibrium (v_phi > 2 * v_z)...', flush=True)
        # Gas is in hydrostatic equilibrium?
        is_hydrostatic = np.greater(np.abs(v_phi), 2.0 * np.abs(v_z))
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_hydrostatic)), flush=True)
            print('Checking if rotational support exceeds thermal pressure support...', flush=True)
        # Rotational support is greater than thermal pressure support?
        is_rotationally_supported = np.greater(0.5 * np.multiply(self.p0_rho[mask_r_max] * self.rho_unit,
                                                                (v_phi * self.v_unit)**2),
                                               2.0 * self.p0_P[mask_r_max] * self.P_unit)
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_rotationally_supported)), flush=True)
            print('Checking density threshold (n_H > {0:.1g} cm^-3)...'.format(n_H_min), flush=True)
        # Satisfies density threshold?
        is_dense = np.greater(self.p0_n_H[mask_r_max], n_H_min)  # Can experiment with density threshold.
        # Combined boolean mask.
        mask_all = np.logical_and(np.logical_and(np.logical_and(is_dense, is_rotating), is_hydrostatic),
                                  is_rotationally_supported)
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_dense)), flush=True)
            print('Number of particles satisfying all checks: {0:d}'.format(np.sum(mask_all)), flush=True)
        disk_ids = g_ids_in_sphere[mask_all]
        
        mask_dict = {'mask_init':mask_init, 'mask_r_max':mask_r_max, 'g_ids_in_sphere':g_ids_in_sphere,
                     'is_rotating':is_rotating, 'is_hydrostatic':is_hydrostatic, 
                     'is_rotationally_supported':is_rotationally_supported, 'is_dense':is_dense, 'mask_all':mask_all}

        # Save disk IDs and relevant disk identification parameters to HDF5 file.
        if save_disk:
            # Default to saving in snapshot directory.
            if diskdir is None:
                diskdir = self.snapdir
            fname_disk = os.path.join(diskdir, 'snapshot_{0:03d}_{1:s}.hdf5'.format(self.get_i(), disk_name))
            if verbose:
                print('Saving to {0:s}...'.format(fname_disk), flush=True)
            f_disk = h5py.File(fname_disk, 'w')
            # Header.
            header_disk = f_disk.create_dataset('header', (1,))
            header_disk.attrs.create('snapshot', self.get_i())
            header_disk.attrs.create('snapdir', self.snapdir)
            if is_single_disk:
                header_disk.attrs.create('disk_type', 'single')  # Single disk.
            else:
                header_disk.attrs.create('disk_type', 'multiple')  # Multiple disk.
            header_disk.attrs.create('disk_name', disk_name)
            header_disk.attrs.create('primary_sink', primary_sink)
            header_disk.attrs.create('sink_ids', sink_ids)
            header_disk.attrs.create('num_sinks', num_sinks_in_disk)
            header_disk.attrs.create('num_gas', len(disk_ids))
            header_disk.attrs.create('sink_ids', np.asarray(sink_ids))
            header_disk.attrs.create('truncation_radius_AU', r_max * self.l_unit * self.cm_to_AU)
            if r_max_truncated_by_neighbor:
                header_disk.attrs.create('truncated_by_neighbor', 'True')
            else:
                header_disk.attrs.create('truncated_by_neighbor', 'False')
            header_disk.attrs.create('n_H_min', n_H_min)
            header_disk.attrs.create('disk_mass', self.p0_m[0] * len(disk_ids))
            header_disk.attrs.create('disk_dm', self.p0_m[0])
            # Dataset of disk IDs.
            if USE_IDX:
                f_disk.create_dataset('disk_idx', data=np.asarray(disk_ids))
            else:
                f_disk.create_dataset('disk_ids', data=np.asarray(disk_ids))
            f_disk.close()

        if return_masks:
            return disk_ids, mask_dict
        else:
            return disk_ids

    # Calculate dust-to-metals ratio, normalized to solar value of 1/2.
    def _return_dust_to_metals_ratio_vs_solar(self, gas_ids, defined='RT_INFRARED', USE_IDX=False):
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        if defined == 'RT_INFRARED':
            return self._sigmoid_sqrt(-0.006*(self.p0_dust_temp[idx_g] - 1500.))
        if defined == 'COOL_LOW_TEMPERATURES':
            return np.where(self.p0_dust_temp[idx_g] >= 2000.0, 1e-4, np.exp(-np.power(self.p0_dust_temp[idx_g]/1000.,3)))

    # Calculate gas mean molecular weight.
    def get_mean_molecular_weight(self, gas_ids, USE_IDX=False):
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        T_eff_atomic          = 1.23 * (5.0/3.0-1.0) * self.u_to_temp_units * self.p0_E_int[idx_g]
        nH_cgs                = self.p0_rho[idx_g] * self.nH_unit
        T_transition          = self._DMIN(8000., nH_cgs)
        f_mol                 = 1./(1. + T_eff_atomic**2/T_transition**2)
        return 4. / (1. + (3. + 4.*self.p0_Ne[idx_g] - 2.*f_mol) * self.HYDROGEN_MASSFRAC)
    
    # Calculate gas adiabatic index.
    def get_gamma_eos(self, gas_ids, USE_IDX=False, use_eos_substellar=False):
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        fH = self.HYDROGEN_MASSFRAC 
        f  = self.p0_molecular_mass_frac[idx_g] 
        xe = self.p0_Ne[idx_g]
        f_mono, f_di         = fH*(xe + 1.-f) + (1.-fH)/4., fH*f/2.
        gamma_mono, gamma_di = 5./3., 7./5.
        if use_eos_substellar:
            gamma_di = self.get_hydrogen_molecule_gamma(gas_ids, USE_IDX=USE_IDX);
        gamma = 1. + (f_mono + f_di) / (f_mono/(gamma_mono-1.) + f_di/(gamma_di-1.))
        return gamma

    # Calculate gas temperature based on gas adiabatic index.
    def get_temperature(self, gas_ids, USE_IDX=False, use_eos_substellar=False):
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        mol_weight  = self.p0_mean_molecular_weight[idx_g]
        E_int       = self.p0_E_int[idx_g]
        gamma       = self.get_gamma_eos(gas_ids, USE_IDX=USE_IDX, use_eos_substellar=use_eos_substellar)
        temperature = (gamma - 1.) * mol_weight * self.u_to_temp_units * E_int
        return temperature

    # Calculate first adiabatic index of hydrogen molecule assuming
    # a 3:1 ortho:para mixture. Use stored snapshot gas temperature (from ThermalProperties()).
    def get_hydrogen_molecule_gamma(self, gas_ids, USE_IDX=False, verbose=False):
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        T_stored = self.p0_temperature_GIZMO[idx_g]
        gamma_H  = np.ones(len(gas_ids))
        idx_l    = (T_stored < 12.5)  # Lower limit: only translation modes.
        idx_u    = (T_stored > 1e5)   # Upper limit: all degrees of freedom are excited.
        idx_m    = np.logical_and((T_stored >= 12.5), (T_stored <= 1e5))
        gamma_H[idx_l] = (5.0/3.0)
        gamma_H[idx_u] = (9.0/7.0)
        gas_ids_m      = gas_ids[idx_m]
        if verbose:
            print('num(upper limit) = {0:d} (T > 1e5)'.format(np.sum(idx_u)))
            print('num(lower limit) = {0:d} (T < 12.5)'.format(np.sum(idx_l)))
            print('num(middle)      = {0:d}'.format(np.sum(idx_m)))
        z_rot = self.get_hydrogen_molecule_zrot_mixture(gas_ids_m, USE_IDX=USE_IDX)
        z_vib = self.get_hydrogen_molecule_zvib(gas_ids_m, USE_IDX=USE_IDX)
        cv    = 1.5 * np.ones(np.sum(idx_m))        # Translation
        cv   += (z_rot[:, 2] / self.BOLTZMANN_CGS)  # Rotation
        cv   += (z_vib[:, 2] / self.BOLTZMANN_CGS)  # Vibration
        gamma_H[idx_m] = np.divide((cv + 1.0), cv)
        return gamma_H
        
    # Rotational partition function of hydrogen molecule and derived quantities,
    # considering a 3:1 mixture of ortho- and parahydrogen that cannot efficiently
    # come into equilibrium.
    # Returns an (n_gas, 3) array containing the partition function value, the average rotational energy per
    # molecule, and the heat capacity per molecule at constant volume.
    def get_hydrogen_molecule_zrot_mixture(self, gas_ids, USE_IDX=False, verbose=False):
        n_gas = len(gas_ids)
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        T_stored   = self.p0_temperature_GIZMO[idx_g]
        EPSILON    = 2.220446049250313e-16
        THETA_ROT  = 85.4                   # in K
        ortho_frac = 0.75                   # 3:1 mixture
        para_frac  = 1.0 - ortho_frac
        x          = np.divide(THETA_ROT, T_stored)
        expmx      = np.exp(-x)
        expmx4     = np.power(expmx, 4.0)
        expterm    = np.multiply(expmx4, np.multiply(expmx, expmx))
        z          = np.zeros((n_gas, 2))  # index 0 for para, 1 for ortho
        zterm      = np.zeros((n_gas, 2))
        dz_dtemp   = np.zeros((n_gas, 2))
        d2z_dtemp2 = np.zeros((n_gas, 2))
        
        z[:, 0], zterm[:, 0] = 1.0, 1.0
        z[:, 1], zterm[:, 1] = 9.0, 9.0
        
        # Sum over rotation levels. Need to do for each array index?
        if verbose:
            print('Summing over rotation levels for {0:d} particles'.format(n_gas), flush=True)
        num_iters = np.zeros(n_gas)
        for k in range(n_gas):
            error           = 1e100
            j, iteration    = 2, 0
            dzterm, d2zterm = 0.0, 0.0
            while(error > EPSILON):
                iteration += 1
                p            = j % 2
                zterm[k, p] *= (2 * j + 1) * expterm[k] / (2 * j - 3)
                jjplusone    = j * (j + 1)
                if (p == 1): # ortho
                    dzterm  = (jjplusone - 2) * x[k] * zterm[k, 1]
                    d2zterm = ((jjplusone - 2) * x[k] - 2) * dzterm
                else:        # para
                    dzterm  = jjplusone * x[k] * zterm[k, 0]
                    d2zterm = (jjplusone * x[k] - 2) * dzterm
                z[k, p]          += zterm[k, p]
                dz_dtemp[k, p]   += dzterm
                d2z_dtemp2[k, p] += d2zterm
                err0 = zterm[k, 0] / z[k, 0]
                err1 = zterm[k, 1] / z[k, 1]
                if (err1 > err0):
                    error = err1
                else:
                    error = err0
                expterm[k] *= expmx4[k]
                j          += 1
            num_iters[k] = iteration
        if verbose:
            print('Done summing over rotation levels.', flush=True)
            print('Num. iterations: ', flush=True)
            print('min = {0:.1f}, max = {1:.1f}'.format(np.min(num_iters), np.max(num_iters)), flush=True)
            print('med = {0:.1f}, avg = {1:.1f}'.format(np.median(num_iters), np.mean(num_iters)), flush=True)
        # Partition function, mean energy per molecule, heat capacity.
        result       = np.zeros((n_gas, 3))
        result[:, 0] = np.exp(para_frac * np.log(z[:, 0]) + ortho_frac * np.log(z[:, 1]))
        result[:, 1] = self.BOLTZMANN_CGS * np.multiply(T_stored, (para_frac * np.divide(dz_dtemp[:, 0], z[:, 0]) + \
                                                                   ortho_frac * np.divide(dz_dtemp[:, 1], z[:, 1])))
        result[:, 2] = self.BOLTZMANN_CGS * (ortho_frac * np.divide((2 * dz_dtemp[:, 1] + d2z_dtemp2[:, 1] - \
                       np.divide(np.multiply(dz_dtemp[:, 1], dz_dtemp[:, 1]), z[:, 1])), z[:, 1]) + \
                       para_frac *  np.divide((2 * dz_dtemp[:, 0] + d2z_dtemp2[:, 0] - \
                       np.divide(np.multiply(dz_dtemp[:, 0], dz_dtemp[:, 0]), z[:, 0])), z[:, 0]))
        return result

    # Vibrational partition function of hydrogen molecule and derived quantities.
    # Returns an (n_gas, 3) array containing the partition function value, the 
    # average rotational energy per molecule, and the heat capacity per molecule at 
    # constant volume.
    def get_hydrogen_molecule_zvib(self, gas_ids, USE_IDX=False): 
        n_gas = len(gas_ids)
        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        T_stored     = self.p0_temperature_GIZMO[idx_g]
        THETA_VIB    = 6140.0
        x            = THETA_VIB/T_stored
        result       = np.zeros((n_gas, 3))
        # expm1 function returns e^x - 1; more accurate when x is close to zero.
        result[:, 0] = -1.0 / np.expm1(-x)
        result[:, 1] = self.BOLTZMANN_CGS * THETA_VIB / np.expm1(x)
        result[:, 2] = THETA_VIB * np.divide(np.multiply(result[:, 0], result[:, 1]), 
                                             np.multiply(T_stored, T_stored))
        return result

    # Calculate non-ideal MHD coefficients.
    def get_nonideal_MHD_coefficients(self, gas_ids, USE_IDX=False, version=5, a=0.1, cr=1.0e-17, etamax=1e24, 
                                      use_eos_substellar=False, verbose=False):

        '''
        version 0: wrong sign on Z_grain.
        version 1: correct sign on Z_grain.
        version 2: new nu_i prefactor.
        version 3: new nu_i prefactor; WRONG positive_definite eta_A formulation.
        version 4: new nu_i prefactor; ALSO WRONG posdef sigma_A2.
        version 5: new nu_i prefactor; CORRECT posdef sigma_A2.
        version 6: include x_neutral factor.
        version 7: apply etamax=1e24 cm^2 s^-1 cap.
        '''
        
        if verbose:
            print('Calculating NMHD resistivities using version {0:d}...'.format(version), flush=True)

        if USE_IDX:
            idx_g = gas_ids
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
            
        '''
        if use_stored_temperature and self.output_temperature:
            p0_temperature = self.p0_temperature_GIZMO[idx_g]
            if verbose:
                print('Using temperature as stored by OUTPUT_TEMPERATURE flag...', flush=True)
        else:
            p0_temperature = self.p0_temperature[idx_g]
            if verbose:
                print('Using temperature as calculated from estimating the gas adiabatic index...', flush=True)
        '''
        
        p0_temperature = self.get_temperature(gas_ids, USE_IDX=USE_IDX, use_eos_substellar=use_eos_substellar)

        zeta_cr        = cr
        a_grain_micron = a
        ag01           = a_grain_micron/0.1
        m_ion          = 24.3
        dust_to_metals = self._return_dust_to_metals_ratio_vs_solar(gas_ids, defined='RT_INFRARED', USE_IDX=USE_IDX)
        f_dustgas      = 0.5 * self.p0_total_metallicity[idx_g] * dust_to_metals

        m_neutral  = self.p0_mean_molecular_weight[idx_g]
        m_grain    = 7.51e9 * ag01*ag01*ag01
        n_eff      = self.p0_rho[idx_g] * self.nH_unit
        #k0         = 1.95e-4 * ag01*ag01 * np.sqrt(self.p0_temperature[idx_g])
        k0         = 1.95e-4 * ag01*ag01 * np.sqrt(p0_temperature)
        ngr_ngas   = (m_neutral/m_grain) * f_dustgas
        #psi_prefac = 167.1 / (ag01 * self.p0_temperature[idx_g])
        psi_prefac = 167.1 / (ag01 * p0_temperature)
        alpha      = zeta_cr * psi_prefac / (ngr_ngas*ngr_ngas * k0 * (n_eff/m_neutral))
        y          = np.sqrt(m_ion*self.PROTONMASS_CGS/self.ELECTRONMASS_CGS)

        mu_eff  = 2.38
        x_elec  = self._DMAX(1e-16, self.p0_Ne[idx_g]*self.HYDROGEN_MASSFRAC*mu_eff)
        R       = x_elec * psi_prefac/ngr_ngas
        psi_0   = -3.787124454911839
        psi     = psi_0
        psi     = np.where(R < 100., psi_0/(1.+np.power(R/0.18967,-0.5646)), psi)
        psi     = np.where(R < 0.002, R*(1.-y)/(1.+2.*y*R), psi)

        n_elec  = x_elec * n_eff/mu_eff
        n_ion   = n_elec * y * np.exp(psi)/(1.-psi)
        Z_grain = psi / psi_prefac

        xe = n_elec / n_eff
        xi = n_ion / n_eff
        xg = ngr_ngas

        #nu_g  = 7.90e-6 * ag01*ag01 * np.sqrt(self.p0_temperature[idx_g]/m_neutral) / (m_neutral+m_grain)
        #nu_ei = 51. * xe * np.power(self.p0_temperature[idx_g], -1.5)
        #nu_e  = nu_ei + 6.21e-9 * np.power(self.p0_temperature[idx_g]/100., 0.65) / m_neutral
        nu_g  = 7.90e-6 * ag01*ag01 * np.sqrt(p0_temperature/m_neutral) / (m_neutral+m_grain)
        nu_ei = 51. * xe * np.power(p0_temperature, -1.5)
        nu_e  = nu_ei + 6.21e-9 * np.power(p0_temperature/100., 0.65) / m_neutral
        nu_ie = ((self.ELECTRONMASS_CGS * xe) / (m_ion * self.PROTONMASS_CGS * xi)) * nu_ei
        if (version == 0) or (version == 1):
            nu_i = (xe/xi) * nu_ei + 1.57e-9 / (m_neutral+m_ion)
        else:
            nu_i = nu_ie + 1.57e-9/(m_neutral+m_ion)

        beta_prefac = self.ELECTRONCHARGE_CGS * (self.p0_B_mag[idx_g] * self.B_unit) / (self.PROTONMASS_CGS * self.C_LIGHT_CGS * n_eff)

        beta_i = beta_prefac / (m_ion * nu_i)
        beta_e = beta_prefac / (self.ELECTRONMASS_CGS/self.PROTONMASS_CGS * nu_e)
        beta_g = beta_prefac / (m_grain * nu_g) * np.abs(Z_grain)

        be_inv = 1./(1. + beta_e**2)
        bi_inv = 1./(1. + beta_i**2)
        bg_inv = 1./(1. + beta_g**2)

        sigma_O = xe*beta_e + xi*beta_i + xg * np.abs(Z_grain) * beta_g

        if version == 0:
            sigma_H = -xe*be_inv + xi*bi_inv - xg*Z_grain*bg_inv  # Old GIZMO typo.
        else:
            sigma_H = -xe*be_inv + xi*bi_inv + xg*Z_grain*bg_inv

        sigma_P     = xe*beta_e*be_inv + xi*beta_i*bi_inv + xg*np.abs(Z_grain)*beta_g*bg_inv
        sigma_perp2 = sigma_H*sigma_H + sigma_P*sigma_P

        # Alternative formulation for eta_A which is automatically positive-definite (version=2).
        #sign_Zgrain = Z_grain/np.abs(Z_grain)
        sign_Zgrain = np.where(Z_grain != 0.0, Z_grain/np.abs(Z_grain), 0.0)

        if version == 3:
            sigma_A2 = (xe*beta_e*be_inv)*(xi*beta_i*bi_inv)*np.power(-beta_e+beta_i,2) + \
                       (xe*beta_e*be_inv)*(xg*np.abs(Z_grain)*beta_g*bg_inv)*np.power(-beta_e+sign_Zgrain*beta_g,2) + \
                       (xi*beta_i*bi_inv)*(xg*np.abs(Z_grain)*beta_g*bg_inv)*np.power(-beta_e+sign_Zgrain*beta_g,2)
        elif version == 4:
            sigma_A2 = (xe*beta_e*be_inv)*(xi*beta_i*bi_inv)*np.power(-beta_e+beta_i,2) + \
                       (xe*beta_e*be_inv)*(xg*np.abs(Z_grain)*beta_g*bg_inv)*np.power(-beta_e+sign_Zgrain*beta_g,2) + \
                       (xi*beta_i*bi_inv)*(xg*np.abs(Z_grain)*beta_g*bg_inv)*np.power(-beta_i+sign_Zgrain*beta_g,2)
                   
        else:
            sigma_A2 = (xe*beta_e*be_inv)*(xi*beta_i*bi_inv)*np.power(beta_i+beta_e,2) + \
                       (xe*beta_e*be_inv)*(xg*np.abs(Z_grain)*beta_g*bg_inv)*np.power(sign_Zgrain*beta_g+beta_e,2) + \
                       (xi*beta_i*bi_inv)*(xg*np.abs(Z_grain)*beta_g*bg_inv)*np.power(sign_Zgrain*beta_g-beta_i,2)

        eta_prefac = (self.p0_B_mag[idx_g] * self.B_unit) * self.C_LIGHT_CGS / (4.0 * np.pi * self.ELECTRONCHARGE_CGS * n_eff)

        eta_O = eta_prefac / sigma_O
        eta_H = eta_prefac * sigma_H / sigma_perp2
        if (version >= 3):
            if (version >= 6):
                x_neutral = self._DMAX(0., 1-m_ion*xi)
                eta_A = x_neutral * eta_prefac * (sigma_A2)/(sigma_O*sigma_perp2)
            else:
                eta_A = eta_prefac * (sigma_A2)/(sigma_O*sigma_perp2)
        else:
            eta_A = eta_prefac * (sigma_P/sigma_perp2 - 1/sigma_O)

        # check against unphysical negative diffusivities
        #eta_O = self._DMAX(0, eta_O)
        #eta_H = self._DMAX(0, eta_H)
        #eta_A = self._DMAX(0, eta_A)
        
        # eta_O, eta_H, and eta_A in cgs units; eta_ohmic, eta_hall, and eta_ad in code units.
        units_cgs_to_code = self.t_unit / self.l_unit**2 
        eta_ohmic         = eta_O * units_cgs_to_code
        eta_hall          = eta_H * units_cgs_to_code
        eta_ad            = eta_A * units_cgs_to_code
        # Apply cap on eta.
        if (version >= 7):
            etamag            = np.abs(eta_ohmic) + np.abs(eta_hall) + np.abs(eta_ad)
            etamax_code       = etamax * units_cgs_to_code  # cap the coefficients
            etacor            = np.ones(len(eta_ohmic))
            idx_gtr           = (etamag > etamax_code)
            etacor[idx_gtr]   = np.divide(etamax_code, etamag[idx_gtr])   # suppression factor
            eta_ohmic         = np.multiply(eta_ohmic, etacor)
            eta_hall          = np.multiply(eta_hall, etacor)
            eta_ad            = np.multiply(eta_ad, etacor)
            eta_O             = eta_ohmic / units_cgs_to_code
            eta_H             = eta_hall / units_cgs_to_code
            eta_A             = eta_ad / units_cgs_to_code

        #return eta_O, eta_H, eta_A         # return in cgs units
        return eta_ohmic, eta_hall, eta_ad  # return in code units.
    
    # Use YT to get SlicePlot data and save to pickle file, for making nice
    # paper figures.
    def get_yt_slc_data(self, zoom=200, ax='x', field_list=[('gas', 'density')], 
                        pickle_data=True, pickdir=None,
                        center_sink_ids=None, verbose=True):
    
        # Pickle name.
        snap_name = self.fname.rsplit('/')[-1].split('.')[0]
        pick_name = '{0:s}_zoom_{1:d}_ax_{2:s}_slc.pkl'.format(snap_name, zoom, ax)
    
        load_data_from_pkl = False
        get_new_data       = True
        new_field_list     = field_list
    
        # Save plot data as dictionary in pickle file.
        plot_data = {}
    
        # First check if pickle file with plot data already exists.
        if pickdir is None:
            pickdir = os.path.join(self.snapdir, 'pickle/')
        if not os.path.exists(pickdir):
            if verbose:
                print('Pickle directory doesn\'t exist; creating pickle directory...')
            os.mkdir(pickdir)
            if verbose:
                print(pickdir)
        # Check if pickle file already exists.
        fname_pkl = os.path.join(pickdir, pick_name)
        if os.path.isfile(fname_pkl):
            if verbose:
                print('Pickle file exists:')
                print(fname_pkl)
            load_data_from_pkl = True
        else:
            if verbose:
                print('No pickle file found. Getting all field data from snapshot...')
        # Check if desired fields are in pickle file.
        if load_data_from_pkl:
            with open(fname_pkl, "rb") as f_pkl:
                if verbose:
                    print('Loading existing pickle file...')
                plot_data_pkl = pk.load(f_pkl)
            plot_data = plot_data_pkl
            pkl_keys  = list(plot_data_pkl.keys())
            if verbose:
                print('Current keys in pickle dict:')
                print(pkl_keys)
            des_keys = []
            for field_tuple in field_list:
                des_keys.append(field_tuple[-1])
            fields_to_add   = []
            new_field_count = 0
            for des_key in des_keys:
                if des_key not in pkl_keys:
                    new_field_count += 1
                    fields_to_add.append(('gas', des_key))
            if (new_field_count == 0):
                get_new_data = False 
                if verbose:
                    print('All fields found in pickle file; returning stored plot dict...')
            else:
                new_field_list = fields_to_add
                if verbose:
                    print('Getting new field data: ')
                    print(fields_to_add)
            
        if get_new_data:
            if verbose:
                print('Using YT to get new plot data...')
            yt.set_log_level(50)
            unit_base = {'UnitMagneticField_in_gauss': self.B_unit,
                        'UnitLength_in_cm': self.l_unit,
                        'UnitMass_in_g': self.m_unit,
                        'UnitVelocity_in_cm_per_s': self.v_unit}
            ds = yt.load(self.fname, unit_base=unit_base); ad = ds.all_data()
            # Using snapshots rotated to disk coordinate frame.
            if center_sink_ids is None:
                c  = ds.arr([0, 0, 0], 'code_length')
            # Using unprocessed GIZMO HDF5 snapshot files.
            else:
                # Get sink center of mass.
                m, cm_x, cm_v = self.sink_center_of_mass(self.p5_ids)
                c             = ds.arr(cm_x, 'code_length')
    
            # Get data region.
            half_width = ds.quan(self.box_size/(2.0 * zoom), 'code_length')
            left_edge  = c - half_width
            right_edge = c + half_width
            box        = ds.region(c, left_edge, right_edge, fields=field_list, ds=ds)
    
            # Get slice plot data.
            slc = yt.SlicePlot(ds, ax, new_field_list, center=c, data_source=box)
            slc.set_axes_unit('AU')
            slc.zoom(zoom)
    
            # Need to plot/save figures to save data.
            tempname = os.path.join(pickdir, 'temp_slc.png')
            slc.save(tempname)

            for i, field_tuple in enumerate(list(slc.plots)):
                field_name = field_tuple[1]
                plot = slc.plots[list(slc.plots)[i]]
                ax   = plot.axes
                img  = ax.images[0]
                data = np.asarray(img.get_array())
                plot_data[field_name] = data
        
            plot_data['xlim']  = slc.xlim
            plot_data['ylim']  = slc.ylim
            plot_data['width'] = slc.width
    
            # Pickle new plot data:
            if pickle_data:
                with open(fname_pkl, 'wb') as f_out:
                    pk.dump(plot_data, f_out)
        return plot_data

    # Utility functions.
    def weight_avg(self, data, weights):
        "Weighted average"
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sum(data * weights) / weightsum
        else:
            return 0
    def weight_std(self, data, weights):
        "Weighted standard deviation."
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



