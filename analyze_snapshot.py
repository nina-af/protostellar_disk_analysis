#!/usr/bin/env python3

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

class Snapshot:
    """
    Class for reading gas/sink particle data from HDF5 snapshot files.
    Initialize with parent cloud.
    """

    def __init__(self, fname, cloud, m_p=1.661e-24, B_unit=1e4):

        # Initial cloud parameters.
        self.fname  = fname
        self.Cloud  = cloud
        self.M0     = cloud.M      # Initial cloud mass, radius.
        self.R0     = cloud.R
        self.L0     = cloud.L      # Volume-equivalent length.
        self.alpha0 = cloud.alpha  # Initial virial parameter.
        self.m_p    = m_p

        # Open h5py file.
        self.f = h5py.File(fname, 'r')
        self.header = self.f['Header']

        # Do star particles exist in this snapshot?
        self.stars_exist = False
        if 'PartType5' in self.f:
            self.stars_exist = True

        # Header attributes.
        self.box_size = self.header.attrs['BoxSize']
        self.num_p0 = self.header.attrs['NumPart_Total'][0]
        self.num_p5 = self.header.attrs['NumPart_Total'][5]
        self.t = self.header.attrs['Time']

        # Unit conversions; note typo in G_code.
        self.G_code = self.header.attrs['Gravitational_Constant_In_Code_Inits']
        self.B_code = self.header.attrs['Internal_UnitB_In_Gauss']
        self.l_unit = self.header.attrs['UnitLength_In_CGS']
        self.m_unit = self.header.attrs['UnitMass_In_CGS']
        self.v_unit = self.header.attrs['UnitVelocity_In_CGS']
        self.B_unit = B_unit
        self.rho_unit   = self.m_unit / self.l_unit**3
        self.t_unit     = self.l_unit / self.v_unit
        self.t_unit_myr = self.t_unit / (3600.0 * 24.0 * 365.0 * 1e6)

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

    def _two_body_center_of_mass(m1, x1, v1, m2, x2, v2):
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
        M, cm_x, cm_v = self._two_body_center_of_mass(M1, M2, x1, v1, x2, v2)
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
        m, x, v = self._get_particle_relative_kinematics('PartType5', sink_ids)
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
        sigma_3D = np.sqrt((weight_std(u, m)**2.0 + weight_std(v, m)**2.0 + \
                            weight_std(w, m)**2.0))
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
        
        r_vals   = np.sqrt(x_rel*2 + y_rel**2 + z_rel**2)
        idx_sort = np.argsort(r_vals)
        m_vals, r_vals = m_vals[idx_sort], r_vals[idx_sort]
        r_rms = weight_std(r_vals, m_vals)
        return r_rms

    # Get half-mass radius of selected gas particles.
    def get_half_mass_radius_gas(self, gas_ids, tol=0.5, verbose=False):
        
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
        cm_enc = np.sum(m_vals[cut])

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

    # Sort sink particles by distance to sink.
    def sort_sinks_by_distance_to_sink(self, sink_id):
        """
        Sorts all other sink particles according to distance from the specified
        sink particle.
        Parameters:
            - sink_id: particle ID of the specified sink particle (all other
            sink particles sorted by distance to this sink).
        Returns:
            - r_sort: [1D array] sorted distances to the specified sink
            particle, in increasing order.
            - ids_sort: [1D array] sorted sink particle IDs.
            - idx_sort: [1D array] indices corresponding to sort.
        """
        # Need at least two sink particles.
        if self.num_p5 < 2:
            return None
        ms, xs, vs  = self.get_sink_kinematics(sink_id)
        # Index corresponding to chosen sink id.
        idx_s = np.where(self.p5_ids == sink_id)[0][0]
        # Particle IDs of other sink particles.
        ids_rest = np.delete(self.p5_ids, idx_s, 0)
        r, ids, idx = self._sort_particles_by_distance_to_point('PartType5', ids_rest, xs)
        return r, ids, idx


    # For disk identification: need coordinate transformation to disk frame.
    # (incomplete)
    # Get net angular momentum unit vector of sink + gas particles.
    def get_net_ang_mom(self, gas_ids, sink_id):
        m_cm, x_cm, v_cm = self.system_center_of_mass(gas_ids, sink_id)
        m_g, x_g, v_g    = self.get_gas_relative_kinematics(gas_ids, x_cm, v_cm)
        ang_mom_vec      = np.sum(np.cross(x_g, v_g), axis=0)
        ang_mom_mag      = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec

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


    # Utility functions.
    def weight_avg(data, weights):
        "Weighted average"
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sum(data * weights) / weightsum
        else:
            return 0

    def weight_std(data, weights):
        "Weighted standard deviation."
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sqrt(np.sum(((data - weight_avg(data, weights))**2) * weights) / weightsum)
        else:
            return 0



