#!/usr/bin/env python3

import sys
import os
import h5py

import pickle as pk
import numpy as np

import yt
import unyt

import matplotlib as mpl
import matplotlib.pyplot as plt

class YTSlicePlotData:
    """
    Class for getting slice plot data from YT plots.
    Plot data can be saved and read from pickle files.
    """
    
    def __init__(self, fname_snap=None, zoom=200, ax='x', 
                 field_list=[('gas', 'density')], 
                 pickle_data=True, pickdir=None,
                 center_coords=None, verbose=True, B_unit=1e4,
                 fname_pkl_stored=None, init_from_pkl=False,
                 ids_ordered=None, center_sink_ids=None):
        
        # Physical constants.
        self.PROTONMASS_CGS     = 1.6726e-24
        self.ELECTRONMASS_CGS   = 9.10953e-28
        self.BOLTZMANN_CGS      = 1.38066e-16
        self.HYDROGEN_MASSFRAC  = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS        = 2.9979e10
        self.HYDROGEN_MASSFRAC  = 0.76
        
        if init_from_pkl:
            with open(fname_pkl_stored, "rb") as f_pkl:
                if verbose:
                    print('No snapshot file; loading existing pickle file...', flush=True)
                self.plot_data = pk.load(f_pkl)
        else:
        
            # Get snapdir, snapshot name from snapshot filename.
            snap_name = fname_snap.rsplit('/')[-1].split('.')[0]
            snapdir   = fname_snap.rsplit(snap_name)[0]
        
            # Open HDF5 file and get snapshot time, units.
            with h5py.File(fname_snap, 'r') as f:
                header = f['Header']
                
                # Header attributes.
                self.box_size = header.attrs['BoxSize']
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

                # Sink particle data.
                self.stars_exist = False  # Do star particles exist in this snapshot?
                if 'PartType5' in f:
                    self.stars_exist = True
                    p5 = f['PartType5']
                    self.p5_ids = p5['ParticleIDs'][()]               # Particle IDs.
                    self.p5_m   = p5['Masses'][()]                    # Masses.
                    self.p5_x   = p5['Coordinates'][()][:, 0]         # Coordinates.
                    self.p5_y   = p5['Coordinates'][()][:, 1]
                    self.p5_z   = p5['Coordinates'][()][:, 2]
        
            # Check for pickle directory in snapdir.
            if pickdir is None:
                pickdir = os.path.join(snapdir, 'pickle/')
            if not os.path.exists(pickdir):
                if verbose:
                    print('Pickle directory doesn\'t exist; creating pickle directory...', flush=True)
                os.mkdir(pickdir)
                if verbose:
                    print(pickdir, flush=True)
            # Pickle filename.
            pick_name = '{0:s}_zoom_{1:d}_ax_{2:s}_slc.pkl'.format(snap_name, zoom, ax)
            fname_pkl = os.path.join(pickdir, pick_name)
            # Check if pickle file already exists.
            if os.path.isfile(fname_pkl):
                if verbose:
                    print('Pickle file already exists:', flush=True)
                    print(fname_pkl, flush=True)
            else:
                if verbose:
                    print('No pickle file found...', flush=True)
        
            self.fname_snap      = fname_snap
            self.fname_pkl       = fname_pkl
            self.pickdir         = pickdir
            self.snapdir         = snapdir
            self.zoom            = zoom
            self.ax              = ax
            self.field_list      = field_list
            self.center_coords   = center_coords
            self.pickle_data     = pickle_data
            self.ids_ordered     = ids_ordered
            self.center_sink_ids = center_sink_ids

            # If center_sink_ids is not None, set center_coords as center of mass.
            if self.center_sink_ids is not None:
                if verbose:
                    print('Updating center_coords...', flush=True)
                self.center_coords = self.get_center_of_mass()
                if verbose:
                    print('New center_coords: ' + str(self.center_coords), flush=True)
        
            # Get YT plot data.
            self.plot_data = self.get_plot_data(self.field_list, verbose=verbose)

    # Get sink particle center of mass.
    def get_center_of_mass(self):
        if not self.stars_exist:
            return None
        idx     = np.isin(self.p5_ids, self.center_sink_ids)
        m, M    = self.p5_m[idx], np.sum(self.p5_m[idx])
        x, y, z = self.p5_x[idx], self.p5_y[idx], self.p5_z[idx]
        x_cm    = np.sum(np.multiply(m, x))/M
        y_cm    = np.sum(np.multiply(m, y))/M
        z_cm    = np.sum(np.multiply(m, z))/M
        cm_x    = np.asarray([x_cm, y_cm, z_cm])
        return cm_x

    # Sort sink particle IDs according to custom ordering.
    def sort_ids(self):
        n_sinks  = len(self.p5_ids)
        idx_sort = []
        for i in range(n_sinks):
            sink_id = self.ids_ordered[i]
            idx_s   = np.argwhere(self.p5_ids == sink_id)[0][0]
            idx_sort.append(idx_s)
        idx_sort = np.asarray(idx_sort)
        return idx_sort

    # Get sink particle mass, coordinate data, for plotting sink particles.
    def get_sink_particle_data(self):
        if not self.stars_exist:
            return None
        # If ids_ordered is None, just use default order.
        if self.ids_ordered is None:
            self.ids_ordered = self.p5_ids
        # Sort sink particle ids according to ids_ordered.
        idx = self.sort_ids()
        m   = self.p5_m[idx]
        x   = self.p5_x[idx]
        y   = self.p5_y[idx]
        z   = self.p5_z[idx]
        sink_base = 'sink_data_'
        sink_data_labels = []
        for i in range(len(self.p5_ids)):
            sink_label = sink_base + str(i+1)
            sink_data_labels.append(sink_label)
        sink_data_dict = {'center_coords':self.center_coords, 'ids':self.ids_ordered}
        for i, sink_label in enumerate(sink_data_labels):
            data = {'m':m[i], 'coords':[x[i], y[i], z[i]]}
            sink_data_dict[sink_label] = data
        return sink_data_dict
        
        
    def get_plot_data(self, field_list=[('gas', 'density')], verbose=True):
        
        load_data_from_pkl = False
        get_new_data       = True
        new_field_list     = field_list
    
        # Save plot data as dictionary in pickle file.
        plot_data = {}
        
        # First check if pickle file with plot data already exists.
        if os.path.isfile(self.fname_pkl):
            load_data_from_pkl = True
        # Check if desired fields are in pickle file.
        if load_data_from_pkl:
            with open(self.fname_pkl, "rb") as f_pkl:
                if verbose:
                    print('Loading existing pickle file...', flush=True)
                plot_data_pkl = pk.load(f_pkl)
            plot_data = plot_data_pkl
            pkl_keys  = list(plot_data_pkl.keys())
            if verbose:
                print('Current keys in pickle dict:', flush=True)
                print(pkl_keys, flush=True)
            if not 'sink_coord_data' in pkl_keys:
                if verbose:
                    print('Need to get sink_coord_data...', flush=True)
                plot_data['sink_coord_data'] = self.get_sink_particle_data()
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
                    print('All fields found in pickle file; returning stored plot dict...', flush=True)
            else:
                new_field_list = fields_to_add
                if verbose:
                    print('Getting new field data: ', flush=True)
                    print(fields_to_add, flush=True)
        else:
            if verbose:
                print('No pickle file found...', flush=True)
                  
        # Get new field data using YT.
        if get_new_data:
            if verbose:
                print('Using YT to get new plot data...', flush=True)
            yt.set_log_level(50)
            unit_base = {'UnitMagneticField_in_gauss': self.B_unit,
                        'UnitLength_in_cm': self.l_unit,
                        'UnitMass_in_g': self.m_unit,
                        'UnitVelocity_in_cm_per_s': self.v_unit}
            ds = yt.load(self.fname_snap, unit_base=unit_base); ad = ds.all_data()
            # Using snapshots rotated to disk coordinate frame.
            if self.center_coords is None:
                #c = ds.arr([0, 0, 0], 'code_length')
                c = ds.domain_center
            else:
                c = ds.arr(self.center_coords, 'code_length')
    
            # Get data region.
            half_width = ds.quan(self.box_size/(2.0 * self.zoom), 'code_length')
            left_edge  = c - half_width
            right_edge = c + half_width
            box        = ds.region(c, left_edge, right_edge, fields=new_field_list, ds=ds)
    
            # Get slice plot data.
            slc = yt.SlicePlot(ds, self.ax, new_field_list, center=c, data_source=box)
            slc.set_axes_unit('AU')
            slc.zoom(self.zoom)
    
            # Need to plot/save figures to save data.
            tempname = os.path.join(self.pickdir, 'temp_slc.png')
            slc.save(tempname)
            
            plot_data_shape = (0, 0)
            for i, field_tuple in enumerate(list(slc.plots)):
                field_name = field_tuple[1]
                plot = slc.plots[list(slc.plots)[i]]
                ax   = plot.axes
                img  = ax.images[0]
                data = np.asarray(img.get_array())
                plot_data[field_name] = data
                if i == 0:
                    plot_data_shape = np.shape(data)
                    
            if 'empty_data' not in plot_data.keys():
                plot_data['empty_data'] = np.zeros(plot_data_shape)
        
            plot_data['xlim']  = slc.xlim
            plot_data['ylim']  = slc.ylim
            plot_data['width'] = slc.width

            if verbose:
                print('Need to get sink_coord_data...', flush=True)
            plot_data['sink_coord_data'] = self.get_sink_particle_data()
    
            # Pickle new plot data:
            if self.pickle_data:
                with open(self.fname_pkl, 'wb') as f_out:
                    pk.dump(plot_data, f_out)
        return plot_data
    
    def print_stats(self, field_name, data_unit=1.0):
        data = self.plot_data[field_name]*data_unit
        print('Min:  {0:.2e}'.format(np.min(data)), flush=True)
        print('Max:  {0:.2e}'.format(np.max(data)), flush=True)
        print('Mean: {0:.2e}'.format(np.mean(data)), flush=True)
        
        
    def draw_streamlines(self, ax, proj_ax='z', density=3, color='white', 
                         arrowsize=1, linewidth=1, field='magnetic'):
        
        data = self.plot_data
        
        xmin = (data['xlim'][0].in_units('AU').v).item()
        xmax = (data['xlim'][1].in_units('AU').v).item()
        ymin = (data['ylim'][0].in_units('AU').v).item()
        ymax = (data['ylim'][1].in_units('AU').v).item()
        
        #X, Y = np.linspace(1, 800, num=800), np.linspace(1, 800, num=800)
        X, Y = np.linspace(xmin, xmax, num=800), np.linspace(ymin, ymax, num=800)
        
        if field == 'magnetic':
            if proj_ax == 'x':
                field_x, field_y = 'magnetic_field_y', 'magnetic_field_z'
            elif proj_ax == 'y':
                field_x, field_y = 'magnetic_field_z', 'magnetic_field_x'
            else:
                field_x, field_y = 'magnetic_field_x', 'magnetic_field_y'
        elif field == 'velocity':
            if proj_ax == 'x':
                field_x, field_y = 'velocity_y', 'velocity_z'
            elif proj_ax == 'y':
                field_x, field_y = 'velocity_z', 'velocity_x'
            else:
                field_x, field_y = 'velocity_x', 'velocity_y'
        
        U, V = self.plot_data[field_x], self.plot_data[field_y]
        im = ax.streamplot(X, Y, U, V, density=density, color=color, 
                           arrowsize=arrowsize, linewidth=linewidth)
        return im
        
    def draw_imshow(self, ax, field_name, cmap, vmin, vmax, norm_type, 
                    linthresh=None, data_unit=1.0, label1=None, label2=None,
                    xticks=None, xtick_labels=None, yticks=None, ytick_labels=None,
                    fs_tick_labels=6, fs_text_labels=6, tick_color='black', text_color='white',
                    label1_x=0.15, label1_y=0.9, label2_x=0.77, label2_y=0.9,
                    use_bbox=False, tl_maj=2, tl_min=1, boxstyle='square'):
        
        data = self.plot_data
        if field_name != 'plasma_beta_custom':
            if verbose:
                self.print_stats(field_name, data_unit=data_unit)
        
        prefactor = 1.0
    
        #fs_tick_labels = 6
        #fs_text_labels = 8
    
        f = field_name
        if (f == 'plasma_beta_custom'):
            data_cs = data['sound_speed']
            data_va = data['alfven_speed']
            data_f  = 2.0 * np.divide(data_cs**2, data_va**2)
        else:
            data_f = data[f]
        
        if vmin is None:
            if f == 'nonideal_eta_H':
                prefactor = -1.0
                vmin = prefactor*data_f.max()*data_unit
            else:
                vmin = data_f.min()*data_unit
        else:
            vmin = vmin
        if vmax is None:
            if f == 'nonideal_eta_H':
                prefactor = -1.0
                vmax = prefactor*data_f.min()*data_unit
            else:
                vmax = data_f.max()*data_unit
        else:
            vmax = vmax
        
        if norm_type == 'log':
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        elif norm_type == 'symlog':
            norm = mpl.colors.SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
        xmin = (data['xlim'][0].in_units('AU').v).item()
        xmax = (data['xlim'][1].in_units('AU').v).item()
        ymin = (data['ylim'][0].in_units('AU').v).item()
        ymax = (data['ylim'][1].in_units('AU').v).item()
    
        w, h = xmax - xmin, ymax - ymin
    
        extent = (-w/2, w/2, -h/2, h/2)

        imshow_args = dict(interpolation='nearest', norm=norm,
                           origin='lower', cmap=cmap,
                           #vmin=vmin, vmax=vmax,
                           extent=extent)

        im = ax.imshow(prefactor*data_f*data_unit, **imshow_args)
        
        tick_color = 'black'
        if cmap in ['plasma', 'magma', 'viridis']:
            tick_color = 'white'
        elif cmap in ['Reds', 'Greens']:
            tick_color = 'black'
    
        if label1 is not None:
            if use_bbox:
                ax.text(label1_x, label1_y, label1, fontsize=fs_text_labels, c='black', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle=boxstyle))
            else:
                ax.text(label1_x, label1_y, label1, fontsize=fs_text_labels, c=tick_color, 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if label2 is not None:
            if use_bbox:
                ax.text(label2_x, label2_y, label2, fontsize=fs_text_labels, c='black', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle=boxstyle))
            else:
                ax.text(label2_x, label2_y, label2, fontsize=fs_text_labels, c=tick_color, 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
        ax.tick_params(which='major', length=tl_maj, left=True, right=True, top=True, bottom=True)
        ax.tick_params(which='minor', length=tl_min, left=True, right=True, top=True, bottom=True)
        ax.tick_params(which='both', direction='in')
        ax.tick_params(which='both', color=tick_color)
            
        if xticks is not None:
            ax.set_xticks(xticks, labels=xtick_labels)
            if xtick_labels is None:
                ax.tick_params(labelbottom=False, labeltop=False)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
        if yticks is not None:
            ax.set_yticks(yticks, labels=ytick_labels)
            if ytick_labels is None:
                ax.tick_params(labelleft=False, labelright=False)
        else:
            ax.tick_params(labelleft=False, labelright=False, left=False, right=False)
            
          
        '''
        if xticks is not None:
            ax.set_xticks(xticks, labels=xtick_labels)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
        if yticks is not None:
            ax.set_yticks(yticks, labels=ytick_labels)
        else:
            ax.tick_params(labelleft=False, labelright=False, left=False, right=False)
        '''
            
        ax.tick_params(labelsize=fs_tick_labels)
        
        return im
    
    def draw_colorbar(self, mappable, cax, cbar_location, cticks, clabel,
                      fs_cb_labels=6, fs_tick_labels=6, cbar_ticks=None, cbar_ticklabels=None,
                      **colorbar_args):
        
        #fs_cb_labels   = 8
        #fs_tick_labels = 6
        
        if cbar_location == 'top':
            orientation = 'horizontal'
        elif cbar_location == 'right':
            orientation = 'vertical'
        else:
            orientation = 'vertical'
        if orientation == 'horizontal':
            va = 'bottom'
            position = 'top'
        elif orientation == 'vertical':
            va = 'top'
            position = 'bottom'

        cb = plt.colorbar(mappable, cax=cax, orientation=orientation, ticks=cticks)
        cb.ax.xaxis.set_ticks_position(position)
        cb.ax.xaxis.set_label_position(position)
        cb.set_label(clabel, fontsize=fs_cb_labels, va=va)
        if cbar_ticklabels is not None:
            cb.set_ticks(ticks=cbar_ticks, labels=cbar_ticklabels)
        cb.ax.tick_params(labelsize=fs_tick_labels)
        cb.ax.tick_params(which='major', length=4)
        cb.ax.tick_params(which='minor', length=2)
        
        return cb
    
    
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

class YTProjectionPlotData:
    """
    Class for getting projection plot data from YT plots.
    Plot data can be saved and read from pickle files.
    """
    
    def __init__(self, fname_snap=None, zoom=200, ax='x', 
                 field_list=[('gas', 'density')], 
                 pickle_data=True, pickdir=None,
                 center_coords=None, verbose=True, B_unit=1e4,
                 fname_pkl_stored=None, init_from_pkl=False,
                 ids_ordered=None, center_sink_ids=None):
        
        # Physical constants.
        self.PROTONMASS_CGS     = 1.6726e-24
        self.ELECTRONMASS_CGS   = 9.10953e-28
        self.BOLTZMANN_CGS      = 1.38066e-16
        self.HYDROGEN_MASSFRAC  = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS        = 2.9979e10
        self.HYDROGEN_MASSFRAC  = 0.76
        
        if init_from_pkl:
            with open(fname_pkl_stored, "rb") as f_pkl:
                if verbose:
                    print('No snapshot file; loading existing pickle file...', flush=True)
                self.plot_data = pk.load(f_pkl)
        else:
        
            # Get snapdir, snapshot name from snapshot filename.
            snap_name = fname_snap.rsplit('/')[-1].split('.')[0]
            snapdir   = fname_snap.rsplit(snap_name)[0]
        
            # Open HDF5 file and get snapshot time, units.
            with h5py.File(fname_snap, 'r') as f:
                header = f['Header']
                
                # Header attributes.
                self.box_size = header.attrs['BoxSize']
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

                # Sink particle data.
                self.stars_exist = False  # Do star particles exist in this snapshot?
                if 'PartType5' in f:
                    self.stars_exist = True
                    p5 = f['PartType5']
                    self.p5_ids = p5['ParticleIDs'][()]               # Particle IDs.
                    self.p5_m   = p5['Masses'][()]                    # Masses.
                    self.p5_x   = p5['Coordinates'][()][:, 0]         # Coordinates.
                    self.p5_y   = p5['Coordinates'][()][:, 1]
                    self.p5_z   = p5['Coordinates'][()][:, 2]
        
            # Check for pickle directory in snapdir.
            if pickdir is None:
                pickdir = os.path.join(snapdir, 'pickle/')
            if not os.path.exists(pickdir):
                if verbose:
                    print('Pickle directory doesn\'t exist; creating pickle directory...', flush=True)
                os.mkdir(pickdir)
                if verbose:
                    print(pickdir, flush=True)
            # Pickle filename.
            pick_name = '{0:s}_zoom_{1:d}_ax_{2:s}_prj.pkl'.format(snap_name, zoom, ax)
            fname_pkl = os.path.join(pickdir, pick_name)
            # Check if pickle file already exists.
            if os.path.isfile(fname_pkl):
                if verbose:
                    print('Pickle file already exists:', flush=True)
                    print(fname_pkl, flush=True)
            else:
                if verbose:
                    print('No pickle file found...', flush=True)
        
            self.fname_snap      = fname_snap
            self.fname_pkl       = fname_pkl
            self.pickdir         = pickdir
            self.snapdir         = snapdir
            self.zoom            = zoom
            self.ax              = ax
            self.field_list      = field_list
            self.center_coords   = center_coords
            self.pickle_data     = pickle_data
            self.ids_ordered     = ids_ordered
            self.center_sink_ids = center_sink_ids

            # If center_sink_ids is not None, set center_coords as center of mass.
            if self.center_sink_ids is not None:
                if verbose:
                    print('Updating center_coords...', flush=True)
                self.center_coords = self.get_center_of_mass()
                if verbose:
                    print('New center_coords: ' + str(self.center_coords), flush=True)
        
            self.plot_data = self.get_plot_data(self.field_list, verbose=verbose)

    # Get sink particle center of mass.
    def get_center_of_mass(self):
        if not self.stars_exist:
            return None
        idx_s   = np.isin(self.p5_ids, self.center_sink_ids)
        m, M    = self.p5_m[idx_s], np.sum(self.p5_m[idx_s])
        x, y, z = self.p5_x[idx_s], self.p5_y[idx_s], self.p5_z[idx_s]
        x_cm    = np.sum(np.multiply(m, x))/M
        y_cm    = np.sum(np.multiply(m, y))/M
        z_cm    = np.sum(np.multiply(m, z))/M
        cm_x    = np.asarray([x_cm, y_cm, z_cm])
        return cm_x

    # Sort sink particle IDs according to custom ordering.
    def sort_ids(self):
        n_sinks  = len(self.p5_ids)
        idx_sort = []
        for i in range(n_sinks):
            sink_id = self.ids_ordered[i]
            idx_s   = np.argwhere(self.p5_ids == sink_id)[0][0]
            idx_sort.append(idx_s)
        idx_sort = np.asarray(idx_sort)
        return idx_sort

    # Get sink particle mass, coordinate data, for plotting sink particles.
    def get_sink_particle_data(self):
        if not self.stars_exist:
            return None
        # If ids_ordered is None, just use default order.
        if self.ids_ordered is None:
            self.ids_ordered = self.p5_ids
        # Sort sink particle ids according to ids_ordered.
        idx = self.sort_ids()
        m   = self.p5_m[idx]
        x   = self.p5_x[idx]
        y   = self.p5_y[idx]
        z   = self.p5_z[idx]
        sink_base = 'sink_data_'
        sink_data_labels = []
        for i in range(len(self.p5_ids)):
            sink_label = sink_base + str(i+1)
            sink_data_labels.append(sink_label)
        sink_data_dict = {'center_coords':self.center_coords, 'ids':self.ids_ordered}
        for i, sink_label in enumerate(sink_data_labels):
            data = {'m':m[i], 'coords':[x[i], y[i], z[i]]}
            sink_data_dict[sink_label] = data
        return sink_data_dict
        
    # Get YT ProjectionPlot data.
    def get_plot_data(self, field_list=[('gas', 'density')], verbose=True):
        
        load_data_from_pkl = False
        get_new_data       = True
        new_field_list     = field_list
    
        # Save plot data as dictionary in pickle file.
        plot_data = {}
        
        # First check if pickle file with plot data already exists.
        if os.path.isfile(self.fname_pkl):
            load_data_from_pkl = True
        # Check if desired fields are in pickle file.
        if load_data_from_pkl:
            with open(self.fname_pkl, "rb") as f_pkl:
                if verbose:
                    print('Loading existing pickle file...', flush=True)
                plot_data_pkl = pk.load(f_pkl)
            plot_data = plot_data_pkl
            pkl_keys  = list(plot_data_pkl.keys())
            if verbose:
                print('Current keys in pickle dict:', flush=True)
                print(pkl_keys, flush=True)
            if not 'sink_coord_data' in pkl_keys:
                if verbose:
                    print('Need to get sink_coord_data...', flush=True)
                if self.stars_exist:
                    plot_data['sink_coord_data'] = self.get_sink_particle_data()
                else:
                    plot_data['sink_coord_data'] = None
            des_keys = []
            for field_tuple in field_list:
                if (field_tuple[-1] in ['magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z',
                                        'velocity_x', 'velocity_y', 'velocity_z']):
                    des_keys.append('{0:s}_weighted'.format(field_tuple[-1]))
                elif (field_tuple[-1] in ['velocity_dispersion']):
                    if (self.ax == 'x'):
                        des_keys.append('velocity_x_dispersion')
                    elif (self.ax == 'y'):
                        des_keys.append('velocity_y_dispersion')
                    elif (self.ax == 'z'):
                        des_keys.append('velocity_z_dispersion')
                else:
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
                    print('All fields found in pickle file; returning stored plot dict...', flush=True)
            else:
                new_field_list = fields_to_add
                if verbose:
                    print('Getting new field data: ', flush=True)
                    print(fields_to_add, flush=True)
        else:
            if verbose:
                print('No pickle file found...', flush=True)
            get_new_data  = True
            fields_to_add = []
            des_keys      = []
            for field_tuple in field_list:
                if (field_tuple[-1] in ['magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z',
                                        'velocity_x', 'velocity_y', 'velocity_z']):
                    des_keys.append('{0:s}_weighted'.format(field_tuple[-1]))
                elif (field_tuple[-1] in ['velocity_dispersion']):
                    if (self.ax == 'x'):
                        des_keys.append('velocity_x_dispersion')
                    elif (self.ax == 'y'):
                        des_keys.append('velocity_y_dispersion')
                    elif (self.ax == 'z'):
                        des_keys.append('velocity_z_dispersion')
                else:
                    des_keys.append(field_tuple[-1])
            for des_key in des_keys:
                fields_to_add.append(('gas', des_key))
            new_field_list = fields_to_add
            if verbose:
                print('Getting new field data: ', flush=True)
                print(fields_to_add, flush=True)

                    
        # Weigh magnetic field info by density.
        weighted_fields_to_add   = []
        unweighted_fields_to_add = []

        # Moment fields = velocity dispersion.
        moment_fields_to_add     = []
        
        if get_new_data:
            for field_tuple in new_field_list:
                if (field_tuple in [('gas', 'magnetic_field_x_weighted'), ('gas', 'magnetic_field_y_weighted'),
                                    ('gas', 'magnetic_field_z_weighted'), ('gas', 'velocity_x_weighted'),
                                    ('gas', 'velocity_y_weighted'), ('gas', 'velocity_z_weighted')]):
                    # Strip '_weighted' from end of field name.
                    field_name_old  = field_tuple[1]
                    field_name_list = field_name_old.split('_')
                    field_name_new  = field_name_list[0]
                    for i in range(1, len(field_name_list)-1):
                        field_name_new += ('_' + field_name_list[i])
                    field_tuple_new = ('gas', field_name_new)
                    weighted_fields_to_add.append(field_tuple_new)
                elif (field_tuple in [('gas', 'velocity_x_dispersion'),
                                      ('gas', 'velocity_y_dispersion'),
                                      ('gas', 'velocity_z_dispersion')]):
                    moment_fields_to_add.append(field_tuple)
                else:
                    unweighted_fields_to_add.append(field_tuple)
            print('Density-weighted fields to add:', flush=True)
            print(weighted_fields_to_add, flush=True)
            print('Unweighted fields to add:', flush=True)
            print(unweighted_fields_to_add, flush=True)
            print('Moment fields to add:', flush=True)
            print(moment_fields_to_add, flush=True)
            
            plot_data = self.get_plot_data_from_YT(plot_data, field_list=unweighted_fields_to_add, 
                                                   weighted=False, verbose=verbose)
            plot_data = self.get_plot_data_from_YT(plot_data, field_list=weighted_fields_to_add, 
                                                   weighted=True, verbose=verbose)
            plot_data = self.get_plot_data_from_YT(plot_data, field_list=moment_fields_to_add,
                                                   moment_field=True, verbose=verbose)

            if verbose:
                print('Need to get sink_coord_data...', flush=True)
            if self.stars_exist:
                plot_data['sink_coord_data'] = self.get_sink_particle_data()
            else:
                plot_data['sink_coord_data'] = None

        # Pickle new plot data:
        if self.pickle_data:
            with open(self.fname_pkl, 'wb') as f_out:
                pk.dump(plot_data, f_out)

        return plot_data
            
    def get_plot_data_from_YT(self, plot_data, field_list=[('gas', 'density')], weighted=False,
                              moment_field=False, verbose=True):
        # Get new field data using YT.
        if verbose:
            print('Using YT to get new plot data...', flush=True)
        yt.set_log_level(50)
        unit_base = {'UnitMagneticField_in_gauss': self.B_unit,
                     'UnitLength_in_cm': self.l_unit,
                     'UnitMass_in_g': self.m_unit,
                     'UnitVelocity_in_cm_per_s': self.v_unit}
        ds = yt.load(self.fname_snap, unit_base=unit_base); ad = ds.all_data()
        # Using snapshots rotated to disk coordinate frame.
        if self.center_coords is None:
            #c = ds.arr([0, 0, 0], 'code_length')
            c = ds.domain_center
        else:
            c = ds.arr(self.center_coords, 'code_length')
    
        # Get data region.
        half_width = ds.quan(self.box_size/(2.0 * self.zoom), 'code_length')
        left_edge  = c - half_width
        right_edge = c + half_width
        box        = ds.region(c, left_edge, right_edge, fields=field_list, ds=ds)
    
        # Get projection plot data.
        if weighted:
            prj = yt.ProjectionPlot(ds, self.ax, field_list, center=c, data_source=box,
                                    weight_field=('gas', 'density'))
        elif moment_field:
            if verbose:
                print('Getting moment fields for axis {0:s}:'.format(self.ax), flush=True)
                print(field_list, flush=True)
            new_field_list = []
            if (('gas', 'velocity_x_dispersion') in field_list):
                new_field_list.append(('gas', 'velocity_x'))
            elif (('gas', 'velocity_y_dispersion') in field_list):
                new_field_list.append(('gas', 'velocity_y'))
            elif (('gas', 'velocity_z_dispersion') in field_list):
                new_field_list.append(('gas', 'velocity_z'))
            if verbose:
                print(new_field_list, flush=True)
            prj = yt.ProjectionPlot(ds, self.ax, new_field_list, center=c, data_source=box,
                                    weight_field=('gas', 'density'), moment=2)
        else:
            prj = yt.ProjectionPlot(ds, self.ax, field_list, center=c, data_source=box)
        prj.set_axes_unit('AU')
        prj.zoom(self.zoom)
    
        # Need to plot/save figures to save data.
        tempname = os.path.join(self.pickdir, 'temp_prj.png')
        prj.save(tempname)
            
        plot_data_shape = (0, 0)
        for i, field_tuple in enumerate(list(prj.plots)):
            field_name = field_tuple[1]
            if weighted:
                field_name = '{0:s}_weighted'.format(field_name)
            if moment_field:
                field_name = '{0:s}_dispersion'.format(field_name)
            plot = prj.plots[list(prj.plots)[i]]
            ax   = plot.axes
            img  = ax.images[0]
            data = np.asarray(img.get_array())
            plot_data[field_name] = data
            if i == 0:
                plot_data_shape = np.shape(data)
                    
        if 'empty_data' not in plot_data.keys():
            plot_data['empty_data'] = np.zeros(plot_data_shape)
        
        plot_data['xlim']  = prj.xlim
        plot_data['ylim']  = prj.ylim
        plot_data['width'] = prj.width
    
        # Pickle new plot data:
        if self.pickle_data:
            with open(self.fname_pkl, 'wb') as f_out:
                pk.dump(plot_data, f_out)
                
        return plot_data
    
    def print_stats(self, field_name, data_unit=1.0):
        data = self.plot_data[field_name]*data_unit
        print('Min:  {0:.2e}'.format(np.min(data)), flush=True)
        print('Max:  {0:.2e}'.format(np.max(data)), flush=True)
        print('Mean: {0:.2e}'.format(np.mean(data)), flush=True)
        
        
    def draw_streamlines(self, ax, proj_ax='z', density=3, color='white', 
                         arrowsize=1, linewidth=1, field='magnetic'):
        
        data = self.plot_data
        
        xmin   = (data['xlim'][0].in_units('AU').v).item()
        xmax   = (data['xlim'][1].in_units('AU').v).item()
        ymin   = (data['ylim'][0].in_units('AU').v).item()
        ymax   = (data['ylim'][1].in_units('AU').v).item()
        w, h   = xmax - xmin, ymax - ymin
        extent = (-w/2, w/2, -h/2, h/2)
        
        #X, Y = np.linspace(1, 800, num=800), np.linspace(1, 800, num=800)
        #X, Y = np.linspace(xmin, xmax, num=800), np.linspace(ymin, ymax, num=800)
        X, Y = np.linspace(extent[0], extent[1], num=800), np.linspace(extent[2], extent[3], num=800)
        
        if field == 'magnetic':
            if proj_ax == 'x':
                field_x, field_y = 'magnetic_field_y_weighted', 'magnetic_field_z_weighted'
            elif proj_ax == 'y':
                field_x, field_y = 'magnetic_field_z_weighted', 'magnetic_field_x_weighted'
            else:
                field_x, field_y = 'magnetic_field_x_weighted', 'magnetic_field_y_weighted'
        elif field == 'velocity':
            if proj_ax == 'x':
                field_x, field_y = 'velocity_y_weighted', 'velocity_z_weighted'
            elif proj_ax == 'y':
                field_x, field_y = 'velocity_z_weighted', 'velocity_x_weighted'
            else:
                field_x, field_y = 'velocity_x_weighted', 'velocity_y_weighted'
        
        U, V = self.plot_data[field_x], self.plot_data[field_y]
        im = ax.streamplot(X, Y, U, V, density=density, color=color, 
                           arrowsize=arrowsize, linewidth=linewidth)
        return im, U, V, X, Y
        
    def draw_imshow(self, ax, field_name, cmap, vmin, vmax, norm_type, 
                    linthresh=None, data_unit=1.0, label1=None, label2=None,
                    xticks=None, xtick_labels=None, yticks=None, ytick_labels=None,
                    fs_tick_labels=6, fs_text_labels=6, tick_color='black', text_color='white',
                    label1_x=0.15, label1_y=0.9, label2_x=0.77, label2_y=0.9,
                    use_bbox=False, verbose=False,
                    tl_maj=2, tl_min=1, boxstyle='square'):
        
        data = self.plot_data
        if field_name != 'plasma_beta_custom':
            if verbose:
                self.print_stats(field_name, data_unit=data_unit)
        
        prefactor = 1.0
    
        #fs_tick_labels = 6
        #fs_text_labels = 8
    
        f = field_name
        if (f == 'plasma_beta_custom'):
            data_cs = data['sound_speed']
            data_va = data['alfven_speed']
            data_f  = 2.0 * np.divide(data_cs**2, data_va**2)
        else:
            data_f = data[f]
        
        if vmin is None:
            if f == 'nonideal_eta_H':
                prefactor = -1.0
                vmin = prefactor*data_f.max()*data_unit
            else:
                vmin = data_f.min()*data_unit
        else:
            vmin = vmin
        if vmax is None:
            if f == 'nonideal_eta_H':
                prefactor = -1.0
                vmax = prefactor*data_f.min()*data_unit
            else:
                vmax = data_f.max()*data_unit
        else:
            vmax = vmax
        
        if norm_type == 'log':
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        elif norm_type == 'symlog':
            norm = mpl.colors.SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
        xmin   = (data['xlim'][0].in_units('AU').v).item()
        xmax   = (data['xlim'][1].in_units('AU').v).item()
        ymin   = (data['ylim'][0].in_units('AU').v).item()
        ymax   = (data['ylim'][1].in_units('AU').v).item()
        w, h   = xmax - xmin, ymax - ymin
        extent = (-w/2, w/2, -h/2, h/2)
        
        if verbose:
            print('IMSHOW: extent = ' + str(extent))

        imshow_args = dict(interpolation='nearest', norm=norm,
                           origin='lower', cmap=cmap,
                           #vmin=vmin, vmax=vmax,
                           extent=extent)

        im = ax.imshow(prefactor*data_f*data_unit, **imshow_args)
        
        tick_color = 'black'
        if cmap in ['plasma', 'magma', 'viridis']:
            tick_color = 'white'
        elif cmap in ['Reds', 'Greens']:
            tick_color = 'black'
    
        if label1 is not None:
            if use_bbox:
                ax.text(label1_x, label1_y, label1, fontsize=fs_text_labels, c='black', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle=boxstyle))
            else:
                ax.text(label1_x, label1_y, label1, fontsize=fs_text_labels, c=tick_color, 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if label2 is not None:
            if use_bbox:
                ax.text(label2_x, label2_y, label2, fontsize=fs_text_labels, c='black', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle=boxstyle))
            else:
                ax.text(label2_x, label2_y, label2, fontsize=fs_text_labels, c=tick_color, 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
        ax.tick_params(which='major', length=tl_maj, left=True, right=True, top=True, bottom=True)
        ax.tick_params(which='minor', length=tl_min, left=True, right=True, top=True, bottom=True)
        ax.tick_params(which='both', direction='in')
        ax.tick_params(which='both', color=tick_color)
            
        if xticks is not None:
            ax.set_xticks(xticks, labels=xtick_labels)
            if xtick_labels is None:
                ax.tick_params(labelbottom=False, labeltop=False)
            else:
                ax.tick_params(labelsize=fs_tick_labels)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
        if yticks is not None:
            ax.set_yticks(yticks, labels=ytick_labels)
            if ytick_labels is None:
                ax.tick_params(labelleft=False, labelright=False)
            else:
                ax.tick_params(labelsize=fs_tick_labels)
        else:
            ax.tick_params(labelleft=False, labelright=False, left=False, right=False)
            
          
        '''
        if xticks is not None:
            ax.set_xticks(xticks, labels=xtick_labels)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
        if yticks is not None:
            ax.set_yticks(yticks, labels=ytick_labels)
        else:
            ax.tick_params(labelleft=False, labelright=False, left=False, right=False)
        '''
            
        ax.tick_params(labelsize=fs_tick_labels)
        
        return im
    
    def draw_colorbar(self, mappable, cax, cbar_location, cticks, clabel,
                      fs_cb_labels=6, fs_tick_labels=6, cbar_ticks=None, cbar_ticklabels=None,
                      **colorbar_args):
        
        #fs_cb_labels   = 8
        #fs_tick_labels = 6
        
        if cbar_location == 'top':
            orientation = 'horizontal'
        elif cbar_location == 'right':
            orientation = 'vertical'
        else:
            orientation = 'vertical'
        if orientation == 'horizontal':
            va = 'bottom'
            position = 'top'
        elif orientation == 'vertical':
            va = 'top'
            position = 'bottom'

        cb = plt.colorbar(mappable, cax=cax, orientation=orientation, ticks=cticks)
        cb.ax.xaxis.set_ticks_position(position)
        cb.ax.xaxis.set_label_position(position)
        cb.set_label(clabel, fontsize=fs_cb_labels, va=va)
        if cbar_ticklabels is not None:
            cb.set_ticks(ticks=cbar_ticks, labels=cbar_ticklabels)
        cb.ax.tick_params(labelsize=fs_tick_labels)
        cb.ax.tick_params(which='major', length=4)
        cb.ax.tick_params(which='minor', length=2)
        
        return cb
        
