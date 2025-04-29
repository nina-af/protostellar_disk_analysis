#!/usr/bin/env python3

import numpy as np
import sys
import os
import h5py

imin, imax = 0, 500  # First and last snapshots containing sink particle data.
snapdir    = 'path_to_snapshots/'
stardir    = os.path.join(snapdir, 'stars_only/')

def get_fname(i, snapdir):
    return os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))

def get_data(imin, imax, snapdir, stardir):
    for i in range(imin, imax+1, 1):
        fname = get_fname(i, snapdir)
        with h5py.File(fname, 'r') as f:
            # Header attributes.
            header   = f['Header']
            box_size = header.attrs['BoxSize']
            t        = header.attrs['Time']
            G_code   = header.attrs['Gravitational_Constant_In_Code_Inits']
            if 'Internal_UnitB_In_Gauss' in header.attrs:
                B_code = header.attrs['Internal_UnitB_In_Gauss']
            else:
                B_code = 2.916731267922059e-09
            l_unit = header.attrs['UnitLength_In_CGS']
            m_unit = header.attrs['UnitMass_In_CGS']
            v_unit = header.attrs['UnitVelocity_In_CGS']
            t_unit = l_unit / v_unit
            B_unit = 1e4
            # PartType5 data.
            p5     = f['PartType5']
            p5_ids = p5['ParticleIDs'][()]               # Particle IDs.
            p5_m   = p5['Masses'][()]                    # Masses.
            p5_x   = p5['Coordinates'][()][:, 0]         # Coordinates.
            p5_y   = p5['Coordinates'][()][:, 1]
            p5_z   = p5['Coordinates'][()][:, 2]
            p5_u   = p5['Velocities'][()][:, 0]          # Velocities.
            p5_v   = p5['Velocities'][()][:, 1]
            p5_w   = p5['Velocities'][()][:, 2]
            p5_lx  = p5['BH_Specific_AngMom'][()][:, 0]  # Specific angular momentum.
            p5_ly  = p5['BH_Specific_AngMom'][()][:, 1]
            p5_lz  = p5['BH_Specific_AngMom'][()][:, 2]
        # For convenience, coordinates and velocities in a (n_gas, 3) array.
        p5_coord   = np.vstack((p5_x, p5_y, p5_z)).T
        p5_vel     = np.vstack((p5_u, p5_v, p5_w)).T
        p5_ang_mom = np.vstack((p5_lx, p5_ly, p5_lz)).T
        # Save each dict as an HDF5 dataset.
        fname_stars = os.path.join(stardir, 'snapshot_{0:03d}_stars.hdf5'.format(i))
        f      = h5py.File(fname_stars, 'w')
        header = f.create_dataset('Header', (1,))
        header.attrs.create('BoxSize', box_size, dtype=float)
        header.attrs.create('Time', t, dtype=float)
        header.attrs.create('m_unit', m_unit, dtype=float)
        header.attrs.create('l_unit', l_unit, dtype=float)
        header.attrs.create('v_unit', v_unit, dtype=float)
        header.attrs.create('t_unit', t_unit, dtype=float)
        header.attrs.create('B_unit', B_unit, dtype=float)
        header.attrs.create('G_code', G_code, dtype=float)
        f.create_dataset('ParticleIDs', data=p5_ids, dtype=int)
        f.create_dataset('Masses', data=p5_m, dtype=float)
        f.create_dataset('Coordinates', data=p5_coord, dtype=float)
        f.create_dataset('Velocities', data=p5_vel, dtype=float)
        f.create_dataset('SpecificAngMom', data=p5_ang_mom, dtype=float)
        f.close()
    
get_data(imin, imax, snapdir, stardir)
    
    
