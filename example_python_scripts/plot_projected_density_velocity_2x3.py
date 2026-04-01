#!/usr/bin/env python3

'''
Make plots from YTProjectionPlot pickles with fields [('gas', 'density'), ('gas', 'velocity_dispersion')].
Calibrated to M0 = 4, R0 = 0.065, alpha0 = 0.5, dm = 1e-5 calculations at zoom = 20. 

Need to adjust:
  - dt_snap, for i_to_t() function.
  - snapdir, figdir
  - cloud parameters, for get_snap() function
  - mass_array vals, for plotting sink particles
  - all_sink_imin (list containing first snapshot each sink particle appears in)
'''
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os

sys.path.append('/home/naf999/protostellar_disk_analysis')
from analyze_snapshot import Cloud, Snapshot
from plot_yt_data import YTSlicePlotData, YTProjectionPlotData
M0, R0, alpha0 = 4.0, 0.065, 0.5
cloud          = Cloud(M0, R0, alpha0)

# Units.
m_unit     = 1.989e+33
l_unit     = 3.085678e+18
v_unit     = 100.0
B_unit     = 1e4
t_unit     = (l_unit/v_unit)
t_unit_myr = t_unit/(3600.0 * 24.0 * 365.0 * 1e6)
cm_to_AU   = 6.68459e-14
eta_unit   = l_unit**2/t_unit

plt.rcParams['font.family'     ] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'

# SPECIFY DATA DIRECTORIES HERE.
basedir = None
snapdir = None        
figdir  = None

# SPECIFY SINK PARTICLE MARKER SIZES HERE.
mass_array_min  = 0.1
mass_array_max  = 4.0
mass_array_step = 0.5
lw              = 1.0

# SPECIFY SNAPSHOT SPACING HERE.
dt_snap = 9.35537e-07

def i_to_t(i, dt_snap=dt_snap):
    return i*dt_snap

def get_fname_snap(i, snapdir):
    return os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))

def get_snap(i, snapdir, cloud=cloud):
    return Snapshot(get_fname_snap(i, snapdir), cloud)

def get_fname_pkl(i, snapdir, zoom, proj_ax):
    pkldir = os.path.join(snapdir, 'pickle/')
    return os.path.join(pkldir, 'snapshot_{0:03d}_zoom_{1:d}_ax_{2:s}_prj.pkl'.format(i, zoom, proj_ax))

def find_nearest_size(mass_value, mass_array_min, mass_array_max, mass_array_step):
    mass_vals = np.arange(mass_array_min, mass_array_max, mass_array_step)
    size_vals = np.full(len(mass_vals), 5, dtype=int)
    for j in range(len(size_vals)):
        size_vals[j] += 5*j
    mass_array = np.asarray(mass_vals)
    size_array = np.asarray(size_vals)
    idx   = (np.abs(mass_array - mass_value)).argmin()
    return size_array[idx]

def make_fig_density_velocity(i, snapdir, all_sink_imin, zoom=20, jet_mass='1em5', 
                              scalebar_w=1000.0, scalebar_unit='AU', savefig=False, figname=None):
    
    n_sinks  = np.sum(i >= np.asarray(all_sink_imin))
    
    fname_x = get_fname_pkl(i, snapdir, zoom, 'x')
    fname_y = get_fname_pkl(i, snapdir, zoom, 'y')
    fname_z = get_fname_pkl(i, snapdir, zoom, 'z')
    
    print('SNAPSHOT {0:03d}: {1:d} SINK PARTICLES'.format(i, n_sinks))
    fig  = plt.figure(figsize=(7.0, 4.67))
    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(2, 3),  
                     axes_pad=0.0,
                     cbar_location="right",
                     cbar_mode="edge",
                     cbar_size="6%",
                     cbar_pad=0.0,
                     cbar_set_cax=True,
                     share_all=True)
    
    xticks, yticks             = None, None 
    xtick_labels, ytick_labels = None, None
    fs_tick_labels = 8
    fs_text_labels = 8
    verbose        = False
    tl_maj         = 2
    tl_min         = 1
    tick_color     = 'black'
    text_color     = 'white'
    use_bbox       = True
    boxstyle       = 'square'
    cbar_location  = 'top'
    fs_cb_labels   = 10
    fs_tick_labels = 10
    
    # PROJECTED DENSITY:
    field1         = 'density'
    cmap1          = 'plasma'
    clabel1        = r'$\log \Sigma$ [g cm$^{-2}$]'
    data_unit1     = 1.0
    cbar_ticks1    = [1e-1, 1e0, 1e1]
    cbar_tl1       = ['-1', '0', '1']
    cmin1, cmax1   = 8e-2, 3e1
    # LINE-OF-SIGHT VELOCITY DISPERSION:
    cmap2          = 'inferno'
    clabel2        =  r'$\log \sigma_{1D}$ [km s$^{-1}$]'
    data_unit2     = 1e-5
    cbar_ticks2    = [1e-1, 1e0]
    cbar_tl2       = ['-1', '0']
    cmin2, cmax2   = 8e-2, 4e0
    
    
    # X-PROJECTION
    label1   = 'non-ideal 6946ff4'
    label2   = ''
    label1_x = 0.26
    label2_x = 0.18
    label1_y = 0.93
    label2_y = 0.83
    prj_data = YTProjectionPlotData(fname_pkl_stored=fname_x, init_from_pkl=True)
    im_11    = prj_data.draw_imshow(grid[0], field1, cmap1, cmin1, cmax1, 'log', data_unit=data_unit1,
                                    label1=label1, label2=label2, 
                                    label1_x=label1_x, label1_y=label1_y, 
                                    label2_x=label2_x, label2_y=label2_y,
                                    xticks=xticks, xtick_labels=xtick_labels, 
                                    yticks=yticks, ytick_labels=ytick_labels,
                                    fs_tick_labels=fs_tick_labels, fs_text_labels=fs_text_labels, 
                                    verbose=verbose, tl_maj=tl_maj, tl_min=tl_min, 
                                    tick_color=tick_color, text_color=text_color,
                                    use_bbox=use_bbox, boxstyle=boxstyle)
    field2   = 'velocity_x_dispersion'
    label1   = ''
    label2   = ''
    im_21    = prj_data.draw_imshow(grid[3], field2, cmap2, cmin2, cmax2, 'log', data_unit=data_unit2,
                                    label1=label1, label2=label2, 
                                    label1_x=label1_x, label1_y=label1_y, 
                                    label2_x=label2_x, label2_y=label2_y,
                                    xticks=xticks, xtick_labels=xtick_labels, 
                                    yticks=yticks, ytick_labels=ytick_labels,
                                    fs_tick_labels=fs_tick_labels, fs_text_labels=fs_text_labels, 
                                    verbose=verbose, tl_maj=tl_maj, tl_min=tl_min, 
                                    tick_color=tick_color, text_color=text_color,
                                    use_bbox=use_bbox, boxstyle=boxstyle)
    
    # Y-PROJECTION
    if (jet_mass in ['1em5', '1em6']):
        label1 = 't = {0:.3f} Myr'.format(i_to_t(i+152)*t_unit_myr) 
        label2 = 't = {0:.1f} t_ff'.format(i_to_t(i+152)/cloud.t_ff)
    else:
        label1 = 't = {0:.3f} Myr'.format(i_to_t(i)*t_unit_myr) 
        label2 = 't = {0:.1f} t_ff'.format(i_to_t(i)/cloud.t_ff)
    label1_x = 0.22
    label2_x = 0.82
    label1_y = 0.93
    label2_y = 0.93
    prj_data = YTProjectionPlotData(fname_pkl_stored=fname_y, init_from_pkl=True)
    im_12    = prj_data.draw_imshow(grid[1], field1, cmap1, cmin1, cmax1, 'log', data_unit=data_unit1,
                                    label1=label1, label2=label2, 
                                    label1_x=label1_x, label1_y=label1_y, 
                                    label2_x=label2_x, label2_y=label2_y,
                                    xticks=xticks, xtick_labels=xtick_labels, 
                                    yticks=yticks, ytick_labels=ytick_labels,
                                    fs_tick_labels=fs_tick_labels, fs_text_labels=fs_text_labels, 
                                    verbose=verbose, tl_maj=tl_maj, tl_min=tl_min, 
                                    tick_color=tick_color, text_color=text_color,
                                    use_bbox=use_bbox, boxstyle=boxstyle)
    label1   = ''
    label2   = ''
    field2   = 'velocity_y_dispersion'
    im_22    = prj_data.draw_imshow(grid[4], field2, cmap2, cmin2, cmax2, 'log', data_unit=data_unit2,
                                    label1=label1, label2=label2, 
                                    label1_x=label1_x, label1_y=label1_y, 
                                    label2_x=label2_x, label2_y=label2_y,
                                    xticks=xticks, xtick_labels=xtick_labels, 
                                    yticks=yticks, ytick_labels=ytick_labels,
                                    fs_tick_labels=fs_tick_labels, fs_text_labels=fs_text_labels, 
                                    verbose=verbose, tl_maj=tl_maj, tl_min=tl_min, 
                                    tick_color=tick_color, text_color=text_color,
                                    use_bbox=use_bbox, boxstyle=boxstyle)
    
    # Z-PROJECTION
    label1   = 'with CR att'
    label1_x = 0.18
    label1_y = 0.93
    label2_y = 0.93
    if (jet_mass == 'no jets'):
        label2   = 'no jets'
        label2_x = 0.88
    elif (jet_mass == '1em5'):
        label2   = 'dm_jet = 1e-5'
        label2_x = 0.78
    elif (jet_mass == '1em6'):
        label2   = 'dm_jet = 1e-6'
        label2_x = 0.78
    prj_data = YTProjectionPlotData(fname_pkl_stored=fname_z, init_from_pkl=True)
    im_13    = prj_data.draw_imshow(grid[2], field1, cmap1, cmin1, cmax1, 'log', data_unit=data_unit1,
                                    label1=label1, label2=label2, 
                                    label1_x=label1_x, label1_y=label1_y, 
                                    label2_x=label2_x, label2_y=label2_y,
                                    xticks=xticks, xtick_labels=xtick_labels, 
                                    yticks=yticks, ytick_labels=ytick_labels,
                                    fs_tick_labels=fs_tick_labels, fs_text_labels=fs_text_labels, 
                                    verbose=verbose, tl_maj=tl_maj, tl_min=tl_min, 
                                    tick_color=tick_color, text_color=text_color,
                                    use_bbox=use_bbox, boxstyle=boxstyle)
    label1   = ''
    label2   = ''
    field2   = 'velocity_z_dispersion'
    im_23    = prj_data.draw_imshow(grid[5], field2, cmap2, cmin2, cmax2, 'log', data_unit=data_unit2,
                                    label1=label1, label2=label2, 
                                    label1_x=label1_x, label1_y=label1_y, 
                                    label2_x=label2_x, label2_y=label2_y,
                                    xticks=xticks, xtick_labels=xtick_labels, 
                                    yticks=yticks, ytick_labels=ytick_labels,
                                    fs_tick_labels=fs_tick_labels, fs_text_labels=fs_text_labels, 
                                    verbose=verbose, tl_maj=tl_maj, tl_min=tl_min, 
                                    tick_color=tick_color, text_color=text_color,
                                    use_bbox=use_bbox, boxstyle=boxstyle)
    
    # PROJECTED DENSITY COLORBAR
    cax1     = grid.cbar_axes[0]
    cb1      = plt.colorbar(im_11, cax=cax1, orientation="vertical")
    cb1.ax.tick_params(which='both', direction='in')
    cb1.ax.xaxis.set_tick_params(pad=1)
    cb1.ax.tick_params(which='major', length=4, labelsize=fs_cb_labels, labelcolor='k')
    cb1.ax.tick_params(which='minor', length=2)
    cb1.mappable.set_clim(cmin1, cmax1)
    cax1.xaxis.set_ticks_position('top')
    cax1.xaxis.set_label_position('top')
    cax1.set_ylabel(clabel1, size=fs_cb_labels, rotation=-90, labelpad=15)
    cb1.ax.minorticks_off()
    cb1.set_ticks(cbar_ticks1, labels=cbar_tl1, fontsize=fs_cb_labels)

    # LINE-OF-SIGHT VELOCITY DISPERSION COLORBAR
    cax2     = grid.cbar_axes[1]
    cb2      = plt.colorbar(im_21, cax=cax2, orientation="vertical")
    cb2.ax.tick_params(which='both', direction='in')
    cb2.ax.xaxis.set_tick_params(pad=1)
    cb2.ax.tick_params(which='major', length=4, labelsize=fs_cb_labels, labelcolor='k')
    cb2.ax.tick_params(which='minor', length=2)
    cb2.mappable.set_clim(cmin2, cmax2)
    cax2.xaxis.set_ticks_position('top')
    cax2.xaxis.set_label_position('top')
    cax2.set_ylabel(clabel2, size=fs_cb_labels, rotation=-90, labelpad=15)
    cb2.ax.minorticks_off()
    cb2.set_ticks(cbar_ticks2, labels=cbar_tl2, fontsize=fs_cb_labels)
        
    # INSET SCALEBAR:
    fontprops = fm.FontProperties(size=fs_text_labels)
    for grid_idx in range(6):
        if grid_idx == 0:
            if (scalebar_unit == 'AU'):
                scalebar_label = '{0:d} AU'.format(int(scalebar_w))
            elif (scalebar_unit == 'pc'):
                scalebar_label = '{0:f} pc'.format(scalebar_w)
        else:
            scalebar_label = ''
        scalebar = AnchoredSizeBar(grid[grid_idx].transData,
                                   scalebar_w, scalebar_label, 'lower left',
                                   pad=0.8, sep=3,
                                   color='white',
                                   frameon=False,
                                   label_top=True,
                                   size_vertical=scalebar_w/10.0,
                                   fontproperties=fontprops)
        grid[grid_idx].add_artist(scalebar)
        
    # PLOT SINK PARTICLES.
    if (n_sinks > 0):
        xlim   = prj_data.plot_data['xlim']
        x1, x2 = xlim[0].v, xlim[1].v
        width  = prj_data.plot_data['width']
        sink_coord_data   = prj_data.plot_data['sink_coord_data']
        img_center_coords = np.asarray(sink_coord_data['center_coords'])

        
        if img_center_coords is None:
            img_center_coords = np.asarray([boxsize/2.0, boxsize/2.0, boxsize/2.0])
        x1 -= img_center_coords[0]
        x2 -= img_center_coords[1]
        for j in range(n_sinks):
            sink_data_label = 'sink_data_' + str(j+1)
            sink_data   = sink_coord_data[sink_data_label]
            sink_m      = sink_data['m']
            sink_size   = find_nearest_size(sink_m, mass_array_min, mass_array_max, mass_array_step)
            
            sink_coords = (sink_data['coords'] - img_center_coords)*l_unit*cm_to_AU
            # Check if sink particle is outside of image range:
            plot_sink = True
            for sc in sink_coords:
                if (sc < x1*l_unit*cm_to_AU) or (sc > x2*l_unit*cm_to_AU):
                    print('sink out of range')
                    plot_sink = False
            if plot_sink:
                grid[0].scatter(sink_coords[1], sink_coords[2], 
                                s=sink_size, c='white', lw=lw, edgecolor='black', marker='o')
                grid[1].scatter(sink_coords[2], sink_coords[0], 
                                s=sink_size, c='white', lw=lw, edgecolor='black', marker='o')
                grid[2].scatter(sink_coords[0], sink_coords[1], 
                                s=sink_size, c='white', lw=lw, edgecolor='black', marker='o')
                grid[3].scatter(sink_coords[1], sink_coords[2], 
                                s=sink_size, c='white', lw=lw, edgecolor='black', marker='o')
                grid[4].scatter(sink_coords[2], sink_coords[0], 
                                s=sink_size, c='white', lw=lw, edgecolor='black', marker='o')
                grid[5].scatter(sink_coords[0], sink_coords[1], 
                                s=sink_size, c='white', lw=lw, edgecolor='black', marker='o')
    
    if savefig:
        fig.savefig(figname, dpi=400, bbox_inches='tight', pad_inches=0.05)
        plt.close()
    else:
        plt.show()
