#!/usr/bin/env python3

import os
import h5py
import numpy as np

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
    """

    def __init__(self, fname):

        # Read from saved HDF5 file.
        f      = h5py.File(fname, 'r')
        header = f['header']

        self.fname        = fname
        self.snapshot     = header.attrs['snapshot']
        self.snapdir      = header.attrs['snapdir']
        self.disk_type    = header.attrs['disk_type']
        self.disk_name    = header.attrs['disk_name']
        self.primary_sink = header.attrs['primary_sink']
        self.sink_ids     = header.attrs['sink_ids']
        self.num_sinks    = header.attrs['num_sinks']
        self.num_gas      = header.attrs['num_gas']
        self.n_H_min      = header.attrs['n_H_min']
        self.disk_dm      = header.attrs['disk_dm']
        self.disk_mass    = header.attrs['disk_mass']
        self.truncation_radius_AU  = header.attrs['truncation_radius_AU']
        if header.attrs['truncated_by_neighbor'] == 'True':
            self.truncated_by_neighbor = True
        else:
            self.truncated_by_neighbor = False
        self.disk_ids = f.get('disk_ids')[:]
        f.close()

    def get_snapshot(self, cloud):
        fname_snap = os.path.join(self.snapdir, 'snapshot_{0:03d}.hdf5'.format(self.snapshot))
        return Snapshot(fname_snap, cloud)

class Disk_USE_IDX:
    """
    Use array indices, not particle IDs, if IDs are not unique.
    """

    def __init__(self, fname):

        # Read from saved HDF5 file.
        f      = h5py.File(fname, 'r')
        header = f['header']

        self.fname        = fname
        self.snapshot     = header.attrs['snapshot']
        self.snapdir      = header.attrs['snapdir']
        self.disk_type    = header.attrs['disk_type']
        self.disk_name    = header.attrs['disk_name']
        self.primary_sink = header.attrs['primary_sink']
        self.sink_ids     = header.attrs['sink_ids']
        self.num_sinks    = header.attrs['num_sinks']
        self.num_gas      = header.attrs['num_gas']
        self.n_H_min      = header.attrs['n_H_min']
        self.disk_dm      = header.attrs['disk_dm']
        self.disk_mass    = header.attrs['disk_mass']
        self.truncation_radius_AU  = header.attrs['truncation_radius_AU']
        if header.attrs['truncated_by_neighbor'] == 'True':
            self.truncated_by_neighbor = True
        else:
            self.truncated_by_neighbor = False
        self.disk_ids = f.get('disk_ids')[:]
        self.disk_idx = f.get('disk_idx')[:]
        f.close()

    def get_snapshot(self, cloud):
        fname_snap = os.path.join(self.snapdir, 'snapshot_{0:03d}.hdf5'.format(self.snapshot))
        return Snapshot(fname_snap, cloud)

class Snapshot:
    """
    Class for reading gas/sink particle data from HDF5 snapshot files.
    Initialize with parent cloud.
    """

    def __init__(self, fname, cloud, m_p=1.661e-24, B_unit=1e4):

        # Initial cloud parameters.
        self.fname   = fname
        self.snapdir = self.get_snapdir()
        self.Cloud   = cloud
        self.M0      = cloud.M      # Initial cloud mass, radius.
        self.R0      = cloud.R
        self.L0      = cloud.L      # Volume-equivalent length.
        self.alpha0  = cloud.alpha  # Initial virial parameter.
        self.m_p     = m_p

        # Open h5py file.
        self.f = h5py.File(fname, 'r')
        self.header = self.f['Header']

        # Do star particles exist in this snapshot?
        self.stars_exist = False
        if 'PartType5' in self.f:
            self.stars_exist = True

        # Header attributes.
        self.box_size = self.header.attrs['BoxSize']
        self.num_p0   = self.header.attrs['NumPart_Total'][0]
        self.num_p5   = self.header.attrs['NumPart_Total'][5]
        self.t        = self.header.attrs['Time']

        # Unit conversions to cgs; note typo in header for G_code.
        self.G_code = self.header.attrs['Gravitational_Constant_In_Code_Inits']
        self.B_code = self.header.attrs['Internal_UnitB_In_Gauss']
        self.l_unit = self.header.attrs['UnitLength_In_CGS']
        self.m_unit = self.header.attrs['UnitMass_In_CGS']
        self.v_unit = self.header.attrs['UnitVelocity_In_CGS']
        self.B_unit = B_unit
        self.t_unit      = self.l_unit / self.v_unit
        self.t_unit_myr  = self.t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
        self.rho_unit    = self.m_unit / self.l_unit**3
        self.P_unit      = self.m_unit / self.l_unit / self.t_unit**2
        self.spec_L_unit = self.l_unit * self.v_unit       # Specific angular momentum (get_net_ang_mom).
        self.L_unit      = self.spec_L_unit * self.m_unit  # Angular momentum.

        # Other useful conversion factors.
        self.cm_to_AU = 6.6845871226706e-14
        self.cm_to_pc = 3.2407792896664e-19

        # Critical density for star formation
        self.sf_density_threshold = self.header.attrs['Density_Threshold_For_SF_CodeUnits']

        # PartType0 data.
        self.p0     = self.f['PartType0']
        self.p0_ids = self.p0['ParticleIDs'][:]

        # Masses.
        self.p0_m = self.p0['Masses'][:]

        # Coordinates.
        self.p0_x = self.p0['Coordinates'][:, 0]
        self.p0_y = self.p0['Coordinates'][:, 1]
        self.p0_z = self.p0['Coordinates'][:, 2]

        # Velocities.
        self.p0_u = self.p0['Velocities'][:, 0]
        self.p0_v = self.p0['Velocities'][:, 1]
        self.p0_w = self.p0['Velocities'][:, 2]

        # Magnetic field.
        self.p0_Bx = self.p0['MagneticField'][:, 0]
        self.p0_By = self.p0['MagneticField'][:, 1]
        self.p0_Bz = self.p0['MagneticField'][:, 2]

        # Mass density [code], H/He number density [cm^-3] (m_p = 1.661e-24 g).
        self.p0_rho          = self.p0['Density'][:]
        self.p0_H_mass_frac  = 1.0 - self.p0['Metallicity'][:, 0]
        self.p0_He_mass_frac = self.p0['Metallicity'][:, 1]
        self.p0_n_H  = (1.0 / self.m_p) * np.multiply(self.p0_rho * self.rho_unit, \
                        self.p0_H_mass_frac)
        self.p0_n_He = (1.0 / (4.0 * self.m_p)) * np.multiply(self.p0_rho * self.rho_unit, \
                        self.p0_He_mass_frac)

        # Electron abundance, neutral hydrogen abundance, moldecular mass fraction.
        self.p0_electron_abundance  = self.p0['ElectronAbundance'][:]
        self.p0_neutral_H_abundance = self.p0['NeutralHydrogenAbundance'][:]
        self.p0_H2_mass_frac        = self.p0['MolecularMassFraction'][:]

        # Internal energy, pressure (P = (gamma - 1) * rho * E_int), sound speed.
        self.p0_E_int = self.p0['InternalEnergy'][:]
        self.p0_P     = self.p0['Pressure'][:]
        self.p0_cs    = self.p0['SoundSpeed'][:]

        # PartType5 data.
        self.p5 = None
        if self.stars_exist:
            self.p5 = self.f['PartType5']

            # Particle IDs.
            self.p5_ids = self.p5['ParticleIDs'][:]

            # Masses, standard gravitational parameter.
            self.p5_m  = self.p5['Masses'][:]
            self.p5_mu = self.G_code * self.p5_m

            # Coordinates.
            self.p5_x = self.p5['Coordinates'][:, 0]
            self.p5_y = self.p5['Coordinates'][:, 1]
            self.p5_z = self.p5['Coordinates'][:, 2]

            # Velocities.
            self.p5_u = self.p5['Velocities'][:, 0]
            self.p5_v = self.p5['Velocities'][:, 1]
            self.p5_w = self.p5['Velocities'][:, 2]

            # Specific angular momentum.
            self.p5_lx = self.p5['BH_Specific_AngMom'][:, 0]
            self.p5_ly = self.p5['BH_Specific_AngMom'][:, 1]
            self.p5_lz = self.p5['BH_Specific_AngMom'][:, 2]

            # Sink particle attributes.
            self.p5_sink_radius      = self.p5['SinkRadius'][:]
            self.p5_accretion_length = self.p5['BH_AccretionLength'][:]
            self.p5_sf_time          = self.p5['StellarFormationTime'][:]

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

    def _get_center_of_mass(self, p_type, p_ids):
        """
        Calculates center of mass for (subset of) gas/sink particles.
        Parameters:
            - p_type: particle type; gas = 'PartType0', sink = 'PartType5'.
            - p_ids: particle IDs of particles to include in center of mass
            calculation.
        Returns: M, cm_x, cm_v
            - M: total mass (scalar)
            - cm_x: center of mass position (array)
            - cm_v: center of mass velocity (array)
        """
        if p_type == 'PartType0':
            idx_p   = np.isin(self.p0_ids, p_ids)
            m, M    = self.p0_m[idx_p], np.sum(self.p0_m[idx_p])
            x, y, z = self.p0_x[idx_p], self.p0_y[idx_p], self.p0_z[idx_p]
            u, v, w = self.p0_u[idx_p], self.p0_v[idx_p], self.p0_w[idx_p]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
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
    def _get_center_of_mass_USE_IDX(self, p_type, p_idx):
        if p_type == 'PartType0':
            m, M    = self.p0_m[p_idx], np.sum(self.p0_m[p_idx])
            x, y, z = self.p0_x[p_idx], self.p0_y[p_idx], self.p0_z[p_idx]
            u, v, w = self.p0_u[p_idx], self.p0_v[p_idx], self.p0_w[p_idx]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
            m, M    = self.p5_m[p_idx], np.sum(self.p5_m[p_idx])
            x, y, z = self.p5_x[p_idx], self.p5_y[p_idx], self.p5_z[p_idx]
            u, v, w = self.p5_u[p_idx], self.p5_v[p_idx], self.p5_w[p_idx]
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
    def gas_center_of_mass(self, gas_ids):
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
        M, cm_x, cm_v = self._get_center_of_mass('PartType0', gas_ids)
        return M, cm_x, cm_v
    def gas_center_of_mass_USE_IDX(self, gas_idx):
        M, cm_x, cm_v = self._get_center_of_mass_USE_IDX('PartType0', gas_idx)
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
    def system_center_of_mass(self, gas_ids, sink_ids):
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
        M1, x1, v1 = self.gas_center_of_mass(gas_ids)
        M2, x2, v2 = self.sink_center_of_mass(sink_ids)
        M, cm_x, cm_v = self._two_body_center_of_mass(M1, x1, v1, M2, x2, v2)
        return M, cm_x, cm_v
    def system_center_of_mass_USE_IDX(self, gas_idx, sink_ids):
        M1, x1, v1 = self.gas_center_of_mass_USE_IDX(gas_idx)
        M2, x2, v2 = self.sink_center_of_mass(sink_ids)
        M, cm_x, cm_v = self._two_body_center_of_mass(M1, x1, v1, M2, x2, v2)
        return M, cm_x, cm_v

    # Get particle mass/position/velocity.
    def _get_particle_kinematics(self, p_type, p_ids):
        """
        Returns masses, positions, and velocities of specified particles.
        Parameters:
            - p_type: particle type; gas = 'PartType0', sink = 'PartType5'.
            - p_ids: particle IDs of specified particles.
        Returns: m, x, v.
            - m: masses of specified particles (1D array)
            - x: positions of specified particles (3D array)
            - v: velocities of specified particles (3D array)
        """
        if p_type == 'PartType0':
            if np.isscalar(p_ids):
                idx_p = np.where(self.p0_ids == p_ids)[0][0]
            else:
                idx_p = np.isin(self.p0_ids, p_ids)
            m       = self.p0_m[idx_p]
            x, y, z = self.p0_x[idx_p], self.p0_y[idx_p], self.p0_z[idx_p]
            u, v, w = self.p0_u[idx_p], self.p0_v[idx_p], self.p0_w[idx_p]
        elif p_type == 'PartType5':
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
    def _get_particle_kinematics_USE_IDX(self, p_type, p_idx):
        if p_type == 'PartType0':
            m       = self.p0_m[p_idx]
            x, y, z = self.p0_x[p_idx], self.p0_y[p_idx], self.p0_z[p_idx]
            u, v, w = self.p0_u[p_idx], self.p0_v[p_idx], self.p0_w[p_idx]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
            m       = self.p5_m[p_idx]
            x, y, z = self.p5_x[p_idx], self.p5_y[p_idx], self.p5_z[p_idx]
            u, v, w = self.p5_u[p_idx], self.p5_v[p_idx], self.p5_w[p_idx]
        else:
            return None
        if np.isscalar(p_idx):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T

    # Get relative particle mass/position/velocity.
    def _get_particle_relative_kinematics(self, p_type, p_ids, point_x, point_v):
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
            if np.isscalar(p_ids):
                idx_p = np.where(self.p0_ids == p_ids)[0][0]
            else:
                idx_p = np.isin(self.p0_ids, p_ids)
            m       = self.p0_m[idx_p]
            x, y, z = self.p0_x[idx_p] - x0, self.p0_y[idx_p] - y0, self.p0_z[idx_p] - z0
            u, v, w = self.p0_u[idx_p] - u0, self.p0_v[idx_p] - v0, self.p0_w[idx_p] - w0
        elif p_type == 'PartType5':
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
    def _get_particle_relative_kinematics_USE_IDX(self, p_type, p_idx, point_x, point_v):
        x0, y0, z0 = point_x[0], point_x[1], point_x[2]
        u0, v0, w0 = point_v[0], point_v[1], point_v[2]
        if p_type == 'PartType0':
            m       = self.p0_m[p_idx]
            x, y, z = self.p0_x[p_idx] - x0, self.p0_y[p_idx] - y0, self.p0_z[p_idx] - z0
            u, v, w = self.p0_u[p_idx] - u0, self.p0_v[p_idx] - v0, self.p0_w[p_idx] - w0
        elif p_type == 'PartType5':
            m       = self.p5_m[p_idx]
            x, y, z = self.p5_x[p_idx] - x0, self.p5_y[p_idx] - y0, self.p5_z[p_idx] - z0
            u, v, w = self.p5_u[p_idx] - u0, self.p5_v[p_idx] - v0, self.p5_w[p_idx] - w0
        else:
            return None
        if np.isscalar(p_idx):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T

    # Get gas particle mass, position, velocity.
    def get_gas_kinematics(self, gas_ids):
        """
        Returns masses, positions, and velocities of specified gas particles.
        Parameters:
            - gas_ids: particle IDs of specified gas particles.
        Returns: m, x, v.
            - m: masses of specified gas particles (1D array)
            - x: positions of specified gas particles (3D array)
            - v: velocities of specified gas particles (3D array)
        """
        m, x, v = self._get_particle_kinematics('PartType0', gas_ids)
        return m, x, v
        # Get gas particle mass, position, velocity.
    def get_gas_kinematics_USE_IDX(self, gas_idx):
        m, x, v = self._get_particle_kinematics_USE_IDX('PartType0', gas_idx)
        return m, x, v

    # Get gas particle mass, position, velocity relative to x0, v0.
    def get_gas_relative_kinematics(self, gas_ids, x0, v0):
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
        m, x, v = self._get_particle_relative_kinematics('PartType0', gas_ids, x0, v0)
        return m, x, v
    def get_gas_relative_kinematics_USE_IDX(self, gas_idx, x0, v0):
        m, x, v = self._get_particle_relative_kinematics_USE_IDX('PartType0', gas_idx, x0, v0)
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
    def _sort_particles_by_distance_to_point(self, p_type, p_ids, point):
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
            idx_p = np.isin(self.p0_ids, p_ids)
            ids_p = self.p0_ids[idx_p]
            x1, y1, z1 = self.p0_x[idx_p], self.p0_y[idx_p], self.p0_z[idx_p]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
            idx_p = np.isin(self.p5_ids, p_ids)
            ids_p = self.p5_ids[idx_p]
            x1, y1, z1 = self.p5_x[idx_p], self.p5_y[idx_p], self.p5_z[idx_p]
        else:
            return None
        x, y, z  = x1 - x0, y1 - y0, z1 - z0
        r        = np.sqrt(x**2 + y**2 + z**2)
        idx_sort = np.argsort(r)
        r_sort   = r[idx_sort]
        ids_sort = ids_p[idx_sort]
        return r_sort, ids_sort, idx_sort
    def _sort_particles_by_distance_to_point_USE_IDX(self, p_type, p_idx, point):
        x0, y0, z0 = point[0], point[1], point[2]
        if p_type == 'PartType0':
            x1, y1, z1 = self.p0_x[p_idx], self.p0_y[p_idx], self.p0_z[p_idx]
        elif p_type == 'PartType5':
            if self.num_p5 == 0:
                return None
            x1, y1, z1 = self.p5_x[p_idx], self.p5_y[p_idx], self.p5_z[p_idx]
        else:
            return None
        x, y, z    = x1 - x0, y1 - y0, z1 - z0
        r          = np.sqrt(x**2 + y**2 + z**2)
        idx_sort   = np.argsort(r)
        r_sort     = r[idx_sort]
        p_idx_sort = p_idx[idx_sort]
        return r_sort, p_idx_sort

    # Sort gas particles by distance to point (x, y, z).
    def sort_gas_by_distance_to_point(self, gas_ids, point):
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
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType0', gas_ids, point)
        return r, ids, idx
    def sort_gas_by_distance_to_point_USE_IDX(self, gas_idx, point):
        r_sort, gas_idx_sort = self._sort_particles_by_distance_to_point_USE_IDX('PartType0', gas_idx, point)
        return r_sort, gas_idx_sort

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
    def sort_gas_by_distance_to_sink(self, gas_ids, sink_id):
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
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType0', gas_ids, xs)
        return r, ids, idx
    def sort_gas_by_distance_to_sink_USE_IDX(self, gas_idx, sink_id):
        if self.num_p5 == 0:
            return None
        ms, xs, vs  = self.get_sink_kinematics(sink_id)
        r_sort, gas_idx_sort = self._sort_particles_by_distance_to_point_USE_IDX('PartType0', gas_idx, xs)
        return r_sort, gas_idx_sort

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
    def get_net_ang_mom(self, gas_ids, sink_ids=None):
        # Just use gas particles to get center of mass, angular momentum.
        if sink_ids is None:
            m_cm, x_cm, v_cm = self.gas_center_of_mass(gas_ids)
        else:
            m_cm, x_cm, v_cm = self.system_center_of_mass(gas_ids, sink_ids)
        m_g, x_g, v_g    = self.get_gas_relative_kinematics(gas_ids, x_cm, v_cm)
        ang_mom_vec      = np.sum(np.cross(x_g, v_g), axis=0)
        ang_mom_mag      = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag
    def get_net_ang_mom_USE_IDX(self, gas_idx, sink_ids=None):
        if sink_ids is None:
            m_cm, x_cm, v_cm = self.gas_center_of_mass_USE_IDX(gas_idx)
        else:
            m_cm, x_cm, v_cm = self.system_center_of_mass_USE_IDX(gas_idx, sink_ids)
        m_g, x_g, v_g    = self.get_gas_relative_kinematics_USE_IDX(gas_idx, x_cm, v_cm)
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
                 disk_name='', save_disk=False, diskdir=None, get_L_vec=False):
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
                print('Getting single disk (sink ID {0:d})...'.format(sink_ids))
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
                print('Getting multiple disk (primary sink ID {0:d}; {1:d} sinks total)...'.format(primary_sink, num_sinks_in_disk))
        # Sink system center of mass, position, velocity.
        m, sink_x, sink_v = self.sink_center_of_mass(sink_ids)
        # Initial Boolean mask specified by gas_ids.
        mask_init = np.isin(self.p0_ids, gas_ids)
        if verbose:
            print('...size(mask_init) = {0:d}'.format(np.sum(mask_init)))
        # Convert r_max to cgs and code units.
        r_max_cgs  = r_max_AU / self.cm_to_AU
        r_max_code = r_max_cgs / self.l_unit
        if verbose:
            print('Initial r_max = {0:.1f} AU.'.format(r_max_AU))
        # Check that r_max is less than distance from center of mass to nearest non-disk sink particle.
        r_max_truncated_by_neighbor = False
        r_max = r_max_code
        if self.num_p5 > 1:
            if verbose:
                print('Checking for nearby sinks within {0:.1f} AU...'.format(r_max_AU))
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
                print('Found nearest sink at r = {0:.1f} AU.'.format(r_near_AU))
            if r_max_code < r_near_code:
                r_max = r_max_code
                if verbose:
                    print('Keeping r_max = {0:.1f} AU.'.format(r_max_AU))
            else:
                r_max = r_near_code
                if verbose:
                    r_max_truncated_by_neighbor = True
                    print('Updating r_max = {0:.1f} AU.'.format(r_near_AU))
        # Sort gas particles by distance to sink system center of mass.
        r_sort, ids_sort, idx_sort = self.sort_gas_by_distance_to_point(gas_ids, sink_x)
        # Update initial mask to exclude gas particles beyond r_max from consideration.
        r_max_idx     = np.argmax(r_sort > r_max)
        ids_in_sphere = ids_sort[:r_max_idx]
        mask_r_max    = np.isin(self.p0_ids, ids_in_sphere)
        mask_r_max    = np.logical_and(mask_init, mask_r_max)
        if verbose:
            print('...size(mask_r_max) = {0:d}'.format(np.sum(mask_r_max)))
        # IDs of gas particles within sphere of radius r_max.
        g_ids_in_sphere = self.p0_ids[mask_r_max]
        # Transform to sink-centered coordinate system; z axis parallel to net ang. momentum.
        if verbose:
            print('Transforming to sink-centered disk coordinate system...')
        # Get angular momentum vector from all gas particles within r_max AU of sink particles.
        L_unit_vec, L_mag = self.get_net_ang_mom(g_ids_in_sphere, sink_ids=sink_ids)
        # Define rotation matrix for coordinate system transformation.
        x_vec, y_vec = self._get_orthogonal_vectors(L_unit_vec)
        A            = self._get_rotation_matrix(x_vec, y_vec, L_unit_vec)
        # Original coordinates (within r_max AU sphere); array shape = (N, 3).
        x_orig = self.p0['Coordinates'][:][mask_r_max]
        v_orig = self.p0['Velocities'][:][mask_r_max]
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
            print('Checking rotational support (v_phi > 2 * v_r)...')
        # Gas is rotationally supported?
        is_rotating = np.greater(np.abs(v_phi), 2.0 * np.abs(v_r))
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_rotating)))
            print('Checking hydrostatic equilibrium (v_phi > 2 * v_z)...')
        # Gas is in hydrostatic equilibrium?
        is_hydrostatic = np.greater(np.abs(v_phi), 2.0 * np.abs(v_z))
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_hydrostatic)))
            print('Checking if rotational support exceeds thermal pressure support...')
        # Rotational support is greater than thermal pressure support?
        is_rotationally_supported = np.greater(0.5 * np.multiply(self.p0_rho[mask_r_max] * self.rho_unit,
                                                                (v_phi * self.v_unit)**2),
                                               2.0 * self.p0_P[mask_r_max] * self.P_unit)
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_rotationally_supported)))
            print('Checking density threshold (n_H > {0:.1g} cm^-3)...'.format(n_H_min))
        # Satisfies density threshold?
        is_dense = np.greater(self.p0_n_H[mask_r_max], n_H_min)  # Can experiment with density threshold.
        # Combined boolean mask.
        mask_all = np.logical_and(np.logical_and(np.logical_and(is_dense, is_rotating), is_hydrostatic),
                                  is_rotationally_supported)
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_dense)))
            print('Number of particles satisfying all checks: {0:d}'.format(np.sum(mask_all)))
        disk_ids = g_ids_in_sphere[mask_all]

        # Save disk IDs and relevant disk identification parameters to HDF5 file.
        if save_disk:
            # Default to saving in snapshot directory.
            if diskdir is None:
                diskdir = self.snapdir
            fname_disk = os.path.join(diskdir, 'snapshot_{0:03d}_{1:s}.hdf5'.format(self.get_i(), disk_name))
            if verbose:
                print('Saving to {0:s}...'.format(fname_disk))
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
            f_disk.create_dataset('disk_ids', data=np.asarray(disk_ids))
            f_disk.close()

        return disk_ids

    # USE_IDX if particle IDs are not unique..
    def get_disk_USE_IDX(self, sink_ids, gas_idx, r_max_AU=500.0, n_H_min=1e9, verbose=False,
                         disk_name='', save_disk=False, diskdir=None):
        is_single_disk    = True
        num_sinks_in_disk = 1
        # Getting disk around single sink particle?
        if np.isscalar(sink_ids):
            primary_sink = sink_ids
            if disk_name == '':
                disk_name = 'disk_single_{0:d}'.format(sink_ids)
            if verbose:
                print('Getting single disk (IDX version; sink ID {0:d})...'.format(sink_ids))
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
                print('Getting multiple disk (IDX version; primary sink ID {0:d}; {1:d} sinks total)...'.format(primary_sink, num_sinks_in_disk))
        # Sink system center of mass, position, velocity.
        m, sink_x, sink_v = self.sink_center_of_mass(sink_ids)
        # Convert r_max to cgs and code units.
        r_max_cgs  = r_max_AU / self.cm_to_AU
        r_max_code = r_max_cgs / self.l_unit
        if verbose:
            print('Initial r_max = {0:.1f} AU.'.format(r_max_AU))
        # Check that r_max is less than distance from center of mass to nearest non-disk sink particle.
        r_max_truncated_by_neighbor = False
        r_max = r_max_code
        if self.num_p5 > 1:
            if verbose:
                print('Checking for nearby sinks within {0:.1f} AU...'.format(r_max_AU))
            if is_single_disk:
                r, ids, idx = self.sort_sinks_by_distance_to_sink(self.p5_ids, primary_sink)
            else:
                other_sink_ids = self.p5_ids[np.isin(self.p5_ids, sink_ids, invert=True)]
                if len(other_sink_ids) == 0:
                    r = np.asarray([10.0*r_max_code])
                else:
                    # Sort other sinks by distance to sink system center of mass.
                    r, ids, idx = self.sort_sinks_by_distance_to_point(other_sink_ids, sink_x)
            # Minimum distance to non-disk sink particles [code units].
            r_near_code = r[0]
            r_near_cgs  = r_near_code * self.l_unit
            r_near_AU   = r_near_cgs * self.cm_to_AU
            if verbose:
                print('Found nearest sink at r = {0:.1f} AU.'.format(r_near_AU))
            if r_max_code < r_near_code:
                r_max = r_max_code
                if verbose:
                    print('Keeping r_max = {0:.1f} AU.'.format(r_max_AU))
            else:
                r_max = r_near_code
                if verbose:
                    r_max_truncated_by_neighbor = True
                    print('Updating r_max = {0:.1f} AU.'.format(r_near_AU))
        r_sort, gas_idx_sort = self.sort_gas_by_distance_to_point_USE_IDX(gas_idx, sink_x)
        # Mask over g_idx to exclude gas particles beyond r_max from consideration.
        r_max_idx       = np.argmax(r_sort > r_max)
        idx_in_sphere   = gas_idx_sort[:r_max_idx]
        mask_r_max      = np.isin(gas_idx, idx_in_sphere)
        g_idx_in_sphere = gas_idx[mask_r_max]
        if verbose:
            print('...size(mask_r_max) = {0:d}'.format(np.sum(mask_r_max)))
        # Transform to sink-centered coordinate system; z axis parallel to net ang. momentum.
        if verbose:
            print('Transforming to sink-centered disk coordinate system...')
        # Get angular momentum vector from all gas particles within r_max AU of sink particles.
        L_unit_vec, L_mag = self.get_net_ang_mom_USE_IDX(g_idx_in_sphere, sink_ids=sink_ids)
        # Define rotation matrix for coordinate system transformation.
        x_vec, y_vec = self._get_orthogonal_vectors(L_unit_vec)
        A            = self._get_rotation_matrix(x_vec, y_vec, L_unit_vec)
        # Original coordinates (within r_max AU sphere); array shape = (N, 3).
        x_orig = self.p0['Coordinates'][:][g_idx_in_sphere]
        v_orig = self.p0['Velocities'][:][g_idx_in_sphere]
        # Number density, density, pressure of particles in r_max sphere.
        n_H = self.p0_n_H[g_idx_in_sphere]
        rho = self.p0_rho[g_idx_in_sphere]
        P   = self.p0_P[g_idx_in_sphere]
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
            print('Checking rotational support (v_phi > 2 * v_r)...')
        # Gas is rotationally supported?
        is_rotating       = np.greater(np.abs(v_phi), 2.0 * np.abs(v_r))
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_rotating)))
            print('Checking hydrostatic equilibrium (v_phi > 2 * v_z)...')
        # Gas is in hydrostatic equilibrium?
        is_hydrostatic       = np.greater(np.abs(v_phi), 2.0 * np.abs(v_z))
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_hydrostatic)))
            print('Checking if rotational support exceeds thermal pressure support...')
        # Rotational support is greater than thermal pressure support?
        is_rotationally_supported = np.greater(0.5 * np.multiply(rho * self.rho_unit,
                                                                (v_phi * self.v_unit)**2),
                                               2.0 * P * self.P_unit)
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_rotationally_supported)))
            print('Checking density threshold (n_H > {0:.1g} cm^-3)...'.format(n_H_min))
        # Satisfies density threshold?
        is_dense = np.greater(n_H, n_H_min)  # Can experiment with density threshold.
        # Combined boolean mask.
        mask_all = np.logical_and(np.logical_and(np.logical_and(is_dense, is_rotating), is_hydrostatic),
                                  is_rotationally_supported)
        g_idx_all = g_idx_in_sphere[mask_all]
        if verbose:
            print('...found {0:d} particles.'.format(np.sum(is_dense)))
            print('Number of particles satisfying all checks: {0:d}'.format(np.sum(mask_all)))
        disk_ids = self.p0_ids[g_idx_all]
        disk_idx = g_idx_all
        # Save disk IDs and relevant disk identification parameters to HDF5 file.
        if save_disk:
            # Default to saving in snapshot directory.
            if diskdir is None:
                diskdir = self.snapdir
            fname_disk = os.path.join(diskdir, 'snapshot_{0:03d}_{1:s}.hdf5'.format(self.get_i(), disk_name))
            if verbose:
                print('Saving to {0:s}...'.format(fname_disk))
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
            f_disk.create_dataset('disk_ids', data=np.asarray(disk_ids))
            # Dataset of disk idxs.
            f_disk.create_dataset('disk_idx', data=np.asarray(disk_idx))
            f_disk.close()

        return disk_idx

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



