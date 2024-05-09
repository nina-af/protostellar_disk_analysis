#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from scipy import stats

import yt
import unyt

from analyze_snapshot import Cloud, Snapshot

# Get density ranges where eta > 0 and eta < 0.
def get_pos_neg(rho, eta):
    idx_pos = np.where(eta >= 0.0)
    idx_neg = np.where(eta < 0.0)
    rho_pos, rho_neg = rho[idx_pos], rho[idx_neg]
    eta_pos, eta_neg = eta[idx_pos], eta[idx_neg]
    return rho_pos, rho_neg, eta_pos, eta_neg

# Compute profile.
def get_density_profile(x_vals, y_vals, num_bins=100):
    # Using equal-spaced bins.
    y_mean, y_bin_edges, _ = stats.binned_statistic(x_vals, y_vals, statistic='mean', bins=num_bins)
    # Bin centers.
    x_centers = (y_bin_edges[:-1] + y_bin_edges[1:])/2
    return x_centers, y_mean

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
