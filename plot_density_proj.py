#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import yt
import unyt

from analyze_snapshot import Cloud, Snapshot

# Some default zoom limits.
plot_dict_5   = {'zoom':5, 'zmin':8e-3, 'zmax':4e-1,
                 'set_center':'box_center', 'center_sink_ids':None,
                 'center_coordinates':None,
                 'use_box':True, 'min_size':None, 'max_size':None, 'mass_step':None,
                 'label':'', 'save_fig':False, 'projdir':None}

plot_dict_50  = {'zoom':50, 'zmin':2e-2, 'zmax':1e1,
                 'set_center':'sink_center', 'center_sink_ids':None,
                 'center_coordinates':None,
                 'use_box':True, 'min_size':None, 'max_size':None, 'mass_step':None,
                 'label':'', 'save_fig':False, 'projdir':None}

plot_dict_100 = {'zoom':100, 'zmin':5e-2, 'zmax':1e1,
                 'set_center':'sink_center', 'center_sink_ids':None,
                 'center_coordinates':None,
                 'use_box':True, 'min_size':None, 'max_size':None, 'mass_step':None,
                 'label':'', 'save_fig':False, 'projdir':None}

plot_dict_400 = {'zoom':400, 'zmin':2e-1, 'zmax':2e1,
                 'set_center':'sink_center', 'center_sink_ids':None,
                 'center_coordinates':None,
                 'use_box':True, 'min_size':None, 'max_size':None, 'mass_step':None,
                 'label':'', 'save_fig':False, 'projdir':None}

def find_nearest_size(mass_value, mass_array_min, mass_array_max, mass_array_step):
    
    mass_vals = np.arange(mass_array_min, mass_array_max, mass_array_step)
    size_vals = np.full(len(mass_vals), 35, dtype=int)
    
    for j in range(len(size_vals)):
        size_vals[j] += 10*j
    
    mass_array = np.asarray(mass_vals)
    size_array = np.asarray(size_vals)
    
    idx   = (np.abs(mass_array - mass_value)).argmin()
    return size_array[idx]

# Sort sink particle IDs according to custom ordering p5_ids_sort.
def sort_ids(p5_ids, sink_order_list):
    n_sinks  = len(p5_ids)
    idx_sort = []
    for i in range(n_sinks):
        sink_id = sink_order_list[i]
        idx_s   = np.argwhere(p5_ids == sink_id)[0][0]
        idx_sort.append(idx_s)
    idx_sort = np.asarray(idx_sort)
    return idx_sort
    
def plot_density_proj(s, verbose=False, **kwargs):
    
    # Specify default plotting values: zoom, zmin, zmax, min_size, max_size, mass_step.
    if 'zoom' in kwargs:
        if kwargs['zoom'] is None:
            zoom = 200
        else:
            zoom = kwargs['zoom']
    else: 
        zoom = 200
    if 'zmin' in kwargs:
        if kwargs['zmin'] is None:
            zmin = 2e-1
        else:
            zmin = kwargs['zmin']
    else:
        zmin = 2e-1
    if 'zmax' in kwargs:
        if kwargs['zmax'] is None:
            zmax = 2e1
        else:
            zmax = kwargs['zmax']
    if 'min_size' in kwargs:
        if kwargs['min_size'] is None:
            min_size = 0.1
        else:
            min_size = kwargs['min_size']
    else:
        min_size = 0.1
    if 'max_size' in kwargs:
        if kwargs['max_size'] is None:
            max_size = 5.0
        else:
            max_size = kwargs['max_size']
    else:
        max_size = 5.0
    if 'mass_step' in kwargs:
        if kwargs['mass_step'] is None:
            mass_step = 0.5
        else:
            mass_step = kwargs['mass_step']
    else:
        mass_step = 0.5
    if 'cmap' in kwargs:
        if kwargs['cmap'] is None:
            cmap = 'plasma'
        else:
            cmap = kwargs['cmap']

    # Plot line specified by 'unit_vec' array.
    plot_unit_vec = False
    if 'unit_vec' in kwargs:
        if kwargs['unit_vec'] is not None:
            plot_unit_vec = True
            unit_vec = kwargs['unit_vec']
            
    # Plot sink particles in color.
    plot_sink_colors = False
    sink_color_list  = []
    sink_order_list  = []
    if 'plot_sink_colors' in kwargs:
        if kwargs['plot_sink_colors'] is not None:
            plot_sink_colors = kwargs['plot_sink_colors']
        if ('sink_color_list' in kwargs) and (kwargs['sink_color_list'] is not None):
            sink_color_list = kwargs['sink_color_list']
        if ('sink_order_list' in kwargs) and (kwargs['sink_order_list'] is not None):
            sink_order_list = kwargs['sink_order_list']

    yt.set_log_level(50)
    unit_base = {'UnitMagneticField_in_gauss': s.B_unit,
                 'UnitLength_in_cm': s.l_unit,
                 'UnitMass_in_g': s.m_unit,
                 'UnitVelocity_in_cm_per_s': s.v_unit}

    ds     = yt.load(s.fname, unit_base=unit_base); ad = ds.all_data()
    t_myrs = np.asarray(ds.current_time) * s.t_unit_myr

    # Check whether there are sink particles to plot.
    plot_particles = False
    if ('PartType5', 'Coordinates') in ds.field_list:
        plot_particles = True
        coords         = ad['PartType5', 'Coordinates']
        masses         = ad['PartType5', 'Masses']
        all_sink_ids   = ad['PartType5', 'ParticleIDs']
        # Sort sink particles if plotting in specific colors.
        if plot_sink_colors:
            idx_sort = sort_ids(all_sink_ids, sink_order_list)
            all_sink_ids = all_sink_ids[idx_sort]
            coords       = coords[idx_sort, :]
            masses       = masses[idx_sort]
    
    # Set plot coordinate origin; default to box center.
    if kwargs['set_center'] == 'custom_center':
        if kwargs['center_coordinates'] is None:
            c = ds.domain_center
        else:
            c = ds.arr(kwargs['center_coordinates'], 'code_length')
    elif kwargs['set_center'] == 'sink_center':
        if kwargs['center_sink_ids'] is None:
            m, cm_x, cm_v = s.sink_center_of_mass(s.p5_ids)
        else:
            m, cm_x, cm_v = s.sink_center_of_mass(kwargs['center_sink_ids'])
        c = ds.arr(cm_x, 'code_length')
    else:
        c = ds.domain_center
     
    if verbose:
        print('Center: ' + str(c))
    
    # Create a box for the projection plot (set by box_size and current zoom).
    if kwargs['use_box']:
        half_width = ds.quan(s.box_size/(2.0 * zoom), 'code_length')
        if verbose:
            print('Zoom = {0:d}:\tbox_half_width = {1:.5f} [code]'.format(zoom, s.box_size/(2.0 * zoom)))
        left_edge  = c - half_width
        right_edge = c + half_width
        box        = ds.region(c, left_edge, right_edge, fields=[('gas', 'density')], ds=ds)
      
    # To-do: add option of plotting other fields.
    field  = [('gas', 'density')]
    dirs   = ['x', 'y', 'z']
    title  = ['', 'L [AU]', '']
    zlabel = 'Projected Density (g cm$^{-2}$)'
    
    figsize       = (20.0, 60.0)
    nrows         = 1
    ncols         = 3
    rect          = '111'
    direction     = 'row'
    axes_pad      = 0.2
    share_all     = False
    label_mode    = 'L'
    cbar_mode     = 'edge'
    cbar_location = 'right'
    cbar_size     = '5%'
    cbar_pad      = '5%'
    cbar_set_cax  = True
    
    fig = plt.figure()
    grid = ImageGrid(fig, rect=(0.075, 0.075, 1.5, 1.5), 
                     nrows_ncols=(nrows, ncols), direction=direction, axes_pad=axes_pad,
                     share_all=share_all, aspect=True, label_mode=label_mode, cbar_mode=cbar_mode,
                     cbar_location=cbar_location, cbar_pad=cbar_pad, cbar_size=cbar_size,
                     cbar_set_cax=cbar_set_cax)
    
    for i, d in enumerate(dirs):
        print('Plotting column...')
        
        if kwargs['use_box']:
            prj = yt.ProjectionPlot(ds, d, field, center=c, data_source=box)
        else:
            prj = yt.ProjectionPlot(ds, d, field, center=c)
        
        prj.set_cmap(field=field, cmap=cmap)
        prj.set_colorbar_label(field, zlabel)
        prj.set_zlim(field=field, zmin=zmin, zmax=zmax)
        prj.zoom(zoom); prj.set_axes_unit('AU'); prj.set_font_size(15.0)   
        
        prj.set_xlabel(title[i]); prj.set_ylabel(title[1])
        
        if plot_particles:
            #for j, particle in enumerate(ad[('PartType5', 'Coordinates')]):
            for j, particle in enumerate(coords):

                # Check to make sure particles are within plotting window.
                dx     = np.abs(c.v - coords[j, :].v)
                max_dx = half_width.v
                if (dx[0] > max_dx) or (dx[1] > max_dx) or (dx[2] > max_dx):
                    continue
                
                m_sink = masses[j]
                scale  = find_nearest_size(m_sink.v, min_size, max_size, mass_step)
                
                if plot_sink_colors:
                    color = sink_color_list[j]
                    #print(int(all_sink_ids[j].v))
                    #print(color)
                    #print('sink {0:d}: color = {1:s}'.format(int(all_sink_ids[j].v), color))
                else:
                    color = 'white'
                
                prj.annotate_marker(particle, marker='o', plot_args={'s':scale, 'color':'black'})
                prj.annotate_marker(particle, marker='o', plot_args={'s':scale, 'color':color, 'linewidth':1.5,
                                                                     'edgecolor':'black'})

        if plot_unit_vec:
            p1 = c - 0.5 * ds.arr(unit_vec, 'code_length')  # Line endpoints.
            p2 = c + 0.5 * ds.arr(unit_vec, 'code_length')
            prj.annotate_line(p1, p2, coord_system="data")
                
        plot = prj.plots[field]; plot.figure = fig; plot.axes = grid[i].axes; plot.cax = grid.cbar_axes[i]
                
        prj._setup_plots(); prj.run_callbacks()
        
    f_str = '{0:.3f} Myr'.format(t_myrs)
    t_str = '{0:.1f} t_cross'.format(s.t / s.t_cross0)
    if 'label' in kwargs:
        if kwargs['label'] is None:
            l_str = ''
        else:
            l_str = '{0:s}'.format(kwargs['label'])
    else:
        l_str = ''
    fig.text(0.83, 1.1, f_str, color='white', size=20)
    fig.text(0.83, 1.05, t_str, color='white', size=20)
    fig.text(0.11, 1.1, l_str, color='white', size=20)
    
    Fsize = fig.get_size_inches()
    fig.set_size_inches(11.5, 8)
    
    if kwargs['save_fig']:
        current_i = s.get_i()
        fname_out = os.path.join(kwargs['projdir'], 'zoom_{0:d}_proj_{1:03d}.png'.format(zoom, current_i))
        plt.savefig(fname_out, bbox_inches='tight', facecolor='white', edgecolor='white')
        plt.close(fig)
    else:
        plt.show()

# Make projection plots of two models side-by-side.
# USE_IDX=True if particle IDs are not unique (e.g., with STARFORGE jets).
def plot_density_proj_disk(s, disk_name, disk_ids, verbose=False, USE_IDX=False, **kwargs):

    # Specify default plotting values: zoom, zmin, zmax, min_size, max_size, mass_step.
    if 'zoom' in kwargs:
        if kwargs['zoom'] is None:
            zoom = 200
        else:
            zoom = kwargs['zoom']
    else: 
        zoom = 200
    if 'zmin' in kwargs:
        if kwargs['zmin'] is None:
            zmin = 2e-1
        else:
            zmin = kwargs['zmin']
    else:
        zmin = 2e-1
    if 'zmax' in kwargs:
        if kwargs['zmax'] is None:
            zmax = 2e1
        else:
            zmax = kwargs['zmax']
    if 'min_size' in kwargs:
        if kwargs['min_size'] is None:
            min_size = 0.1
        else:
            min_size = kwargs['min_size']
    else:
        min_size = 0.1
    if 'max_size' in kwargs:
        if kwargs['max_size'] is None:
            max_size = 5.0
        else:
            max_size = kwargs['max_size']
    else:
        max_size = 5.0
    if 'mass_step' in kwargs:
        if kwargs['mass_step'] is None:
            mass_step = 0.5
        else:
            mass_step = kwargs['mass_step']
    else:
        mass_step = 0.5
    if 'cmap' in kwargs:
        if kwargs['cmap'] is None:
            cmap = 'plasma'
        else:
            cmap = kwargs['cmap']
    if 'cmap2' in kwargs:
        if kwargs['cmap2'] is None:
            cmap2 = 'Blues'
        else:
            cmap2 = kwargs['cmap2']
            
    # Plot line specified by 'unit_vec' array.
    plot_unit_vec = False
    if 'unit_vec' in kwargs:
        if kwargs['unit_vec'] is not None:
            plot_unit_vec = True
            unit_vec = kwargs['unit_vec']

    yt.set_log_level(50)
    unit_base = {'UnitMagneticField_in_gauss': s.B_unit,
                 'UnitLength_in_cm': s.l_unit,
                 'UnitMass_in_g': s.m_unit,
                 'UnitVelocity_in_cm_per_s': s.v_unit}

    if USE_IDX:
        disk_x = s.p0_x[disk_ids]
        def disk(pfilter, data):
            filter = np.isin(data[('PartType0', 'Coordinates')][:, 0].v, disk_x)
            return filter
        yt.add_particle_filter(
            'disk_{0:s}'.format(disk_name), function=disk, filtered_type='PartType0', requires=['Coordinates']
        )
    else:
        def disk(pfilter, data):
            filter = np.isin(data[('PartType0', 'ParticleIDs')], disk_ids)
            return filter
        yt.add_particle_filter(
            'disk_{0:s}'.format(disk_name), function=disk, filtered_type='PartType0', requires=['ParticleIDs']
        )

    ds     = yt.load(s.fname, unit_base=unit_base); ds.add_particle_filter('disk_{0:s}'.format(disk_name))
    ad = ds.all_data(); plot_particles = False
    t_myrs = np.asarray(ds.current_time) * s.t_unit_myr

    # Check whether there are sink particles to plot.
    plot_particles = False
    if ('PartType5', 'Coordinates') in ds.field_list:
        plot_particles = True
        coords         = ad['PartType5', 'Coordinates']
        masses         = ad['PartType5', 'Masses']

    # Set plot coordinate origin; default to box center.
    if kwargs['set_center'] == 'custom_center':
        if kwargs['center_coordinates'] is None:
            c = ds.domain_center
        else:
            c = ds.arr(kwargs['center_coordinates'], 'code_length')
    elif kwargs['set_center'] == 'sink_center':
        if kwargs['center_sink_ids'] is None:
            # NEED TO UPDATE - no sink_id.
            m, cm_x, cm_v = s.sink_center_of_mass(sink_id)
        else:
            m, cm_x, cm_v = s.sink_center_of_mass(kwargs['center_sink_ids'])
        c = ds.arr(cm_x, 'code_length')
    else:
        c = ds.domain_center

    if verbose:
        print('Center: ' + str(c))

    # Create a box for the projection plot (set by box_size and current zoom).
    if kwargs['use_box']:
        half_width = ds.quan(s.box_size/(2.0 * zoom), 'code_length')
        if verbose:
            print('Zoom = {0:d}:\tbox_half_width = {1:.5f} [code]'.format(zoom, s.box_size/(2.0 * zoom)))
        left_edge  = c - half_width
        right_edge = c + half_width
        box        = ds.region(c, left_edge, right_edge, fields=[('gas', 'density')], ds=ds)

    field_1 = [('gas', 'density')]
    field_2 = [('disk_{0:s}'.format(disk_name), 'Density')]
    dirs    = ['x', 'y', 'z']
    title   = ['', 'L [AU]', '']
    zlabel  = "Projected Density (g cm$^{-2}$)"

    figsize       = (40.0, 60.0)
    nrows         = 2
    ncols         = 3
    rect          = '111'
    direction     = 'row'
    axes_pad      = 0.2
    share_all     = False
    label_mode    = 'L'
    cbar_mode     = 'edge'
    cbar_location = 'right'
    cbar_size     = '5%'
    cbar_pad      = '5%'
    cbar_set_cax  = True

    fig = plt.figure()
    grid = ImageGrid(fig, rect=(0.075, 0.075, 1.5, 1.5),
                     nrows_ncols=(nrows, ncols), direction=direction, axes_pad=axes_pad,
                     share_all=share_all, aspect=True, label_mode=label_mode, cbar_mode=cbar_mode,
                     cbar_location=cbar_location, cbar_pad=cbar_pad, cbar_size=cbar_size,
                     cbar_set_cax=cbar_set_cax)

    for i, d in enumerate(dirs):
        print('Plotting column...')

        if kwargs['use_box']:
            prj1 = yt.ProjectionPlot(ds, d, field_1, center=c, data_source=box)
            prj2 = yt.ProjectionPlot(ds, d, field_2, center=c, data_source=box)
        else:
            prj1 = yt.ProjectionPlot(ds, d, field, center=c)
            prj2 = yt.ProjectionPlot(ds, d, field, center=c)

        prj1.set_cmap(field=field_1, cmap=cmap); prj1.set_colorbar_label(field_1, zlabel)
        prj1.set_zlim(field=field_1, zmin=zmin, zmax=zmax)
        prj1.zoom(zoom); prj1.set_axes_unit('AU'); prj1.set_font_size(15.0)
        prj1.set_xlabel(title[i]); prj1.set_ylabel(title[1])

        prj2.set_cmap(field=field_2, cmap=cmap2); prj2.set_colorbar_label(field_2, zlabel)
        #prj2.set_zlim(field=field, zmin=zmin, zmax=zmax)
        prj2.zoom(zoom); prj2.set_axes_unit('AU'); prj2.set_font_size(15.0)
        prj2.set_xlabel(title[i]); prj2.set_ylabel(title[1])

        if plot_particles:
            for j, particle in enumerate(ad[('PartType5', 'Coordinates')]):

                # Check to make sure particles are within plotting window.
                dx     = np.abs(c.v - coords[j, :].v)
                max_dx = half_width.v
                if (dx[0] > max_dx) or (dx[1] > max_dx) or (dx[2] > max_dx):
                    continue

                m_sink = masses[j]
                scale  = find_nearest_size(m_sink.v, min_size, max_size, mass_step)
                color  = 'white'

                prj1.annotate_marker(particle, marker='o', plot_args={'s':scale, 'color':'black'})
                prj1.annotate_marker(particle, marker='o', plot_args={'s':scale, 'color':color, 'linewidth':1.5,
                                                                     'edgecolor':'black'})

        if plot_unit_vec:
            p1 = c - 0.5 * ds.arr(unit_vec, 'code_length')  # Line endpoints.
            p2 = c + 0.5 * ds.arr(unit_vec, 'code_length')
            prj1.annotate_line(p1, p2, coord_system="data")
            prj2.annotate_line(p1, p2, coord_system="data")

        plot1 = prj1.plots[field_1]; plot1.figure = fig; plot1.axes = grid[i].axes; plot1.cax = grid.cbar_axes[i]
        plot2 = prj2.plots[field_2]; plot2.figure = fig; plot2.axes = grid[i+3].axes; plot2.cax = grid.cbar_axes[i+3]

        prj1._setup_plots(); prj1.run_callbacks()
        prj2._setup_plots(); prj2.run_callbacks()

    f_str = '{0:.3f} Myr'.format(t_myrs)
    t_str = '{0:.1f} t_cross'.format(s.t / s.t_cross0)
    if 'label' in kwargs:
        if kwargs['label'] is None:
            l_str = ''
        else:
            l_str = '{0:s}'.format(kwargs['label'])
    else:
        l_str = ''
    if 'sink_label' in kwargs:
        if kwargs['sink_label'] is None:
            s_str = ''
        else:
            s_str = '{0:s}'.format(kwargs['sink_label'])
    else:
        s_str = ''
    if 'extra_label' in kwargs:
        if kwargs['extra_label'] is None:
            e_str = ''
        else:
            e_str = '{0:s}'.format(kwargs['extra_label'])
    else:
        e_str = ''
    fig.text(0.87, 1.45, f_str, color='white', size=20)
    fig.text(0.87, 1.40, t_str, color='white', size=20)
    fig.text(0.11, 1.45, l_str, color='white', size=20)
    fig.text(0.11, 0.75, s_str, color='black', size=20)
    fig.text(0.11, 0.70, e_str, color='black', size=20)

    Fsize = fig.get_size_inches()
    fig.set_size_inches(11.5, 8)

    if kwargs['save_fig']:
        current_i = s.get_i()
        fname_out = os.path.join(kwargs['projdir'], 'zoom_{0:d}_proj_disk_{1:s}_snapshot_{2:03d}.png'.format(zoom, disk_name, current_i))
        plt.savefig(fname_out, bbox_inches='tight', facecolor='white', edgecolor='white')
        plt.close(fig)
    else:
        plt.show()
        
        
def plot_density_proj_gas(s, gas_ids, gas_name, USE_IDX=False, verbose=True, **kwargs):
    # Specify default plotting values: zoom, zmin, zmax, min_size, max_size, mass_step.
    if 'zoom' in kwargs:
        if kwargs['zoom'] is None:
            zoom = 200
        else:
            zoom = kwargs['zoom']
    else:
        zoom = 200
    if 'zmin' in kwargs:
        if kwargs['zmin'] is None:
            zmin = 2e-1
        else:
            zmin = kwargs['zmin']
    else:
        zmin = 2e-1
    if 'zmax' in kwargs:
        if kwargs['zmax'] is None:
            zmax = 2e1
        else:
            zmax = kwargs['zmax']
    else:
        zmax = 2e1
    if 'zmin2' in kwargs:
        if kwargs['zmin2'] is None:
            zmin2 = 1e-3
        else:
            zmin2 = kwargs['zmin2']
    else:
        zmin2 = 1e-3
    if 'zmax2' in kwargs:
        if kwargs['zmax2'] is None:
            zmax2 = 1e-1
        else:
            zmax2 = kwargs['zmax2']
    else:
        zmax2 = 1e-1
    if 'min_size' in kwargs:
        if kwargs['min_size'] is None:
            min_size = 0.1
        else:
            min_size = kwargs['min_size']
    else:
        min_size = 0.1
    if 'max_size' in kwargs:
        if kwargs['max_size'] is None:
            max_size = 5.0
        else:
            max_size = kwargs['max_size']
    else:
        max_size = 5.0
    if 'mass_step' in kwargs:
        if kwargs['mass_step'] is None:
            mass_step = 0.5
        else:
            mass_step = kwargs['mass_step']
    else:
        mass_step = 0.5
    if 'cmap' in kwargs:
        if kwargs['cmap'] is None:
            cmap = 'Greens'
        else:
            cmap = kwargs['cmap']
    else:
        cmap = 'Greens'
        
    # Plot line specified by 'unit_vec' array.
    plot_unit_vec = False
    if 'unit_vec' in kwargs:
        if kwargs['unit_vec'] is not None:
            plot_unit_vec = True
            unit_vec = kwargs['unit_vec']
        
    yt.set_log_level(50)
    unit_base = {'UnitMagneticField_in_gauss': s.B_unit,
                 'UnitLength_in_cm': s.l_unit,
                 'UnitMass_in_g': s.m_unit,
                 'UnitVelocity_in_cm_per_s': s.v_unit}

    if USE_IDX:
        gas_x = s.p0_x[gas_ids]
        def gas_selection(pfilter, data):
            filter = np.isin(data[('PartType0', 'Coordinates')][:, 0].v, gas_x)
            return filter
        yt.add_particle_filter(
            gas_name, function=gas_selection, filtered_type='PartType0', requires=['Coordinates']
        )
    else:
        def gas_selection(pfilter, data):
            filter = np.isin(data[('PartType0', 'ParticleIDs')], gas_ids)
            return filter
        yt.add_particle_filter(
            gas_name, function=gas_selection, filtered_type='PartType0', requires=['ParticleIDs']
        )
        
    ds = yt.load(s.fname, unit_base=unit_base); ds.add_particle_filter(gas_name)
    ad = ds.all_data(); plot_particles = False
    t_myrs = np.asarray(ds.current_time) * s.t_unit_myr

    # Check whether there are sink particles to plot.
    plot_particles = False
    if ('PartType5', 'Coordinates') in ds.field_list:
        plot_particles = True
        coords         = ad['PartType5', 'Coordinates']
        masses         = ad['PartType5', 'Masses']

    # Set plot coordinate origin; default to box center.
    if kwargs['set_center'] == 'custom_center':
        if kwargs['center_coordinates'] is None:
            c = ds.domain_center
        else:
            c = ds.arr(kwargs['center_coordinates'], 'code_length')
    elif kwargs['set_center'] == 'sink_center':
        if kwargs['center_sink_ids'] is None:
            # NEED TO UPDATE - no sink_id.
            m, cm_x, cm_v = s.sink_center_of_mass(sink_id)
        else:
            m, cm_x, cm_v = s.sink_center_of_mass(kwargs['center_sink_ids'])
        c = ds.arr(cm_x, 'code_length')
    else:
        c = ds.domain_center
        
    # Create a box for the projection plot (set by box_size and current zoom).
    if kwargs['use_box']:
        half_width = ds.quan(s.box_size/(2.0 * zoom), 'code_length')
        if verbose:
            print('Zoom = {0:d}:\tbox_half_width = {1:.5f} [code]'.format(zoom, s.box_size/(2.0 * zoom)))
        left_edge  = c - half_width
        right_edge = c + half_width
        box        = ds.region(c, left_edge, right_edge, fields=[('gas', 'density')], ds=ds)
        
    field_1 = [('gas', 'density')]
    field_2 = [(gas_name, 'Density')]
    dirs    = ['x', 'y', 'z']
    title   = ['', 'L [AU]', '']
    zlabel  = "Projected Density (g cm$^{-2}$)"

    figsize       = (40.0, 60.0)
    nrows         = 2
    ncols         = 3
    rect          = '111'
    direction     = 'row'
    axes_pad      = 0.2
    share_all     = False
    label_mode    = 'L'
    cbar_mode     = 'edge'
    cbar_location = 'right'
    cbar_size     = '5%'
    cbar_pad      = '5%'
    cbar_set_cax  = True
    
    fig = plt.figure()
    grid = ImageGrid(fig, rect=(0.075, 0.075, 1.5, 1.5),
                     nrows_ncols=(nrows, ncols), direction=direction, axes_pad=axes_pad,
                     share_all=share_all, aspect=True, label_mode=label_mode, cbar_mode=cbar_mode,
                     cbar_location=cbar_location, cbar_pad=cbar_pad, cbar_size=cbar_size,
                     cbar_set_cax=cbar_set_cax)

    for i, d in enumerate(dirs):
        print('Plotting column...')

        if kwargs['use_box']:
            prj1 = yt.ProjectionPlot(ds, d, field_1, center=c, data_source=box)
            prj2 = yt.ProjectionPlot(ds, d, field_2, center=c, data_source=box)
        else:
            prj1 = yt.ProjectionPlot(ds, d, field_1, center=c)
            prj2 = yt.ProjectionPlot(ds, d, field_2, center=c)

        prj1.set_cmap(field=field_1, cmap='plasma'); prj1.set_colorbar_label(field_1, zlabel)
        prj1.set_zlim(field=field_1, zmin=zmin, zmax=zmax)
        prj1.zoom(zoom); prj1.set_axes_unit('AU'); prj1.set_font_size(15.0)
        prj1.set_xlabel(title[i]); prj1.set_ylabel(title[1])

        prj2.set_cmap(field=field_2, cmap=cmap); prj2.set_colorbar_label(field_2, zlabel)
        #prj2.set_zlim(field=field_2, zmin=zmin2, zmax=zmax2)
        #if verbose:
        #    print('zmin2 = {0:.2g}\nzmax2 = {1:.2g}'.format(zmin2, zmax2))
        prj2.zoom(zoom); prj2.set_axes_unit('AU'); prj2.set_font_size(15.0)
        prj2.set_xlabel(title[i]); prj2.set_ylabel(title[1])

        if plot_particles:
            for j, particle in enumerate(ad[('PartType5', 'Coordinates')]):

                # Check to make sure particles are within plotting window.
                dx     = np.abs(c.v - coords[j, :].v)
                max_dx = half_width.v
                if (dx[0] > max_dx) or (dx[1] > max_dx) or (dx[2] > max_dx):
                    continue

                m_sink = masses[j]
                scale  = find_nearest_size(m_sink.v, min_size, max_size, mass_step)
                color  = 'white'

                prj1.annotate_marker(particle, marker='o', plot_args={'s':scale, 'color':'black'})
                prj1.annotate_marker(particle, marker='o', plot_args={'s':scale, 'color':color, 'linewidth':1.5,
                                                                     'edgecolor':'black'})

        if plot_unit_vec:
            p1 = c - 0.5 * ds.arr(unit_vec, 'code_length')  # Line endpoints.
            p2 = c + 0.5 * ds.arr(unit_vec, 'code_length')
            prj1.annotate_line(p1, p2, coord_system="data")
            prj2.annotate_line(p1, p2, coord_system="data")

        plot1 = prj1.plots[field_1]; plot1.figure = fig; plot1.axes = grid[i].axes; plot1.cax = grid.cbar_axes[i]
        plot2 = prj2.plots[field_2]; plot2.figure = fig; plot2.axes = grid[i+3].axes; plot2.cax = grid.cbar_axes[i+3]

        prj1._setup_plots(); prj1.run_callbacks()
        prj2._setup_plots(); prj2.run_callbacks()

    f_str = '{0:.3f} Myr'.format(t_myrs)
    t_str = '{0:.1f} t_ff'.format(s.t / s.t_ff0)
    if 'label' in kwargs:
        if kwargs['label'] is None:
            l_str = ''
        else:
            l_str = '{0:s}'.format(kwargs['label'])
    else:
        l_str = ''
    if 'sink_label' in kwargs:
        if kwargs['sink_label'] is None:
            s_str = ''
        else:
            s_str = '{0:s}'.format(kwargs['sink_label'])
    else:
        s_str = ''
    if 'extra_label' in kwargs:
        if kwargs['extra_label'] is None:
            e_str = ''
        else:
            e_str = '{0:s}'.format(kwargs['extra_label'])
    else:
        e_str = ''
    fig.text(0.87, 1.45, f_str, color='white', size=20)
    fig.text(0.87, 1.40, t_str, color='white', size=20)
    fig.text(0.11, 1.45, l_str, color='white', size=20)
    fig.text(0.11, 0.75, s_str, color='black', size=20)
    fig.text(0.11, 0.70, e_str, color='black', size=20)

    Fsize = fig.get_size_inches()
    fig.set_size_inches(11.5, 8)

    if kwargs['save_fig']:
        current_i = s.get_i()
        fname_out = os.path.join(kwargs['projdir'], 
                                 'zoom_{0:d}_proj_{1:s}_snapshot_{2:03d}.png'.format(zoom, gas_name, current_i))
        plt.savefig(fname_out, bbox_inches='tight', facecolor='white', edgecolor='white')
        plt.close(fig)
    else:
        plt.show()
        
def plot_nmhd_density_profiles(s, version=4):
    '''
    version 0: wrong sign on Z_grain.
    version 1: correct sign on Z_grain.
    version 2: new nu_i prefactor.
    version 3: new nu_i prefactor; WRONG positive_definite eta_A formulation.
    version 4: new nu_i prefactor; ALSO WRONG posdef sigma_A2.
    version 5: new nu_i prefactor; CORRECT (?) posdef sigma_A2.
    '''
    eta_O, eta_H, eta_A = s.get_nonideal_MHD_coefficients(s.p0_ids, version=version)
    
    rho = s.p0_rho * s.rho_unit
    n_H = s.p0_n_H
    
    # Get density ranges where eta_H < 0 and eta_H > 0.
    rho_H_pos, rho_H_neg, eta_H_pos, eta_H_neg = get_pos_neg(rho, eta_H)
    
    # Compute density profiles.
    x_O, y_O         = get_density_profile(np.log10(rho), eta_O, num_bins=100)
    x_H_pos, y_H_pos = get_density_profile(np.log10(rho_H_pos), eta_H_pos, num_bins=100)
    x_H_neg, y_H_neg = get_density_profile(np.log10(rho_H_neg), eta_H_neg, num_bins=100)
    x_A, y_A         = get_density_profile(np.log10(rho), eta_A, num_bins=100)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    lw = 2

    ax.plot(x_O, y_O, linewidth=lw, c='tab:red', label=r'$\eta_O$')
    ax.plot(x_H_pos, y_H_pos, linewidth=lw, c='tab:blue', label=r'$\eta_H > 0$')
    ax.plot(x_H_neg, np.abs(y_H_neg), linewidth=lw, c='cyan', label=r'$\eta_H < 0$')
    ax.plot(x_A, y_A, linewidth=lw, c='tab:green', label=r'$\eta_A$')

    ax.set_ylabel(r'$|\eta|$ [cm$^2$ s$^{-1}$]', fontsize=12)
    ax.set_xlabel(r'$\log_{10}(\rho)$ [g cm$^{-3}$]', fontsize=12)
    
    ax.set_title('SNAPSHOT {0:03d} ({1:.2f} t_cross)'.format(s.get_i(), s.t/s.t_cross0), fontsize=13)

    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--snapdir', type=str,
                        default='/home/naf999/turbsphere/M50_n5e6_nonideal/',
                        help='snapshot directory')                    
    parser.add_argument('-p', '--projdir', type=str,
                        default='/home/naf999/turbsphere/M50_n5e6_nonideal/proj/',
                        help='snapshot directory')                   
    parser.add_argument('-i', '--index', type=int,
                        default=113,
                        help='snapshot number') 
    parser.add_argument('-zoom', '--zoom', type=int,
                        default=50,
                        help='zoom factor')
    parser.add_argument('-zmin', '--zmin', type=float,
                        default=1e-2,
                        help='projected density min value')                    
    parser.add_argument('-zmax', '--zmax', type=float,
                        default=1e0,
                        help='projected density max value')
    args = parser.parse_args()

    plot_dict = {'zoom':args.zoom,
                 'zmin':args.zmin, 'zmax':args.zmax,
                 'set_center':'sink_center',
                 'center_sink_ids':None, 'center_coordinates':None,
                 'use_box':True,
                 'min_size':None, 'max_size':None, 'mass_step':None,
                 'label':'plot_defaults', 'save_fig':True,
                 'projdir':args.projdir}
                 
    cloud = Cloud(50.0, 0.3077, 1.0)     
    fname = os.path.join(args.snapdir, 'snapshot_{0:03d}.hdf5'.format(args.index))        
    s = Snapshot(fname, cloud)
                 
    plot_density_proj(s, **plot_dict)
