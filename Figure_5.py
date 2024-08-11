# Project: HTTK-Japan
# Project start time: 2023/11/18
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot the pseudo warming HT-EAD simulations (Figure 5)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colorbar
from matplotlib import gridspec
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rpy2.robjects as ro


# mpl.use('Qt5Agg')
plt.close('all')
one_clm = 3.46
two_clm = 7.08

# induce some R function
rread = ro.r['load']
ls = ro.r('ls()')
levels = ro.r('levels')

###################################################################################################################
# Pseudo Prediction
pdt_dir = '/work/a07/qiang/Aging_Japan/02_model/Input_Data/'
fig_dir = '/work/a07/qiang/Aging_Japan/04_plot/fig_test/'
input_dir = '/work/a07/qiang/Aging_Japan/climate_data/'
plot_dir = '/work/a07/qiang/Aging_Japan/04_plot/'
jp_hsi_dir = '/work/a07/qiang/HSI_Japan/Japan_Data/HSIs_1980_2019/'

######################################################################################
# Load old and young prediction
rread(pdt_dir + 'Pseudo_old_young_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')
old_prediction = np.array(ro.r('old_prediction'))  # (47, 1220)
young_prediction = np.array(ro.r('young_prediction'))  # (47, 1220)
pseudo_old_prediction = np.array(ro.r('pseudo_old_prediction'))  # (2, 47, 1220)
pseudo_young_prediction = np.array(ro.r('pseudo_young_prediction'))  # (2, 47, 1220)

rread(pdt_dir + 'Pseudo_old_young_Prediction_10y_mean_k9_sp0_offset_dfg_lag3_1degree.Rdata')
warming_old_prediction = np.array(ro.r('pseudo_old_prediction'))  # (10, 47, 1220)
warming_young_prediction = np.array(ro.r('pseudo_young_prediction'))  # (10, 47, 1220)

warming_old_prediction_real = np.array(ro.r('pseudo_old_prediction_real'))  # (10, 47, 1220)
warming_young_prediction_real = np.array(ro.r('pseudo_young_prediction_real'))  # (10, 47, 1220)

# (6, 47, 1220)  'Newborn', 'Baby', 'Teenager', 'Adult', 'Elderly', 'Unclear'
age_htk = np.load(pdt_dir + 'model_input.npz')['age_htk']
old_htk = age_htk[4, :, :]
young_htk = np.nansum(age_htk[:4, :, :], axis=0)

old_htk_yr = np.zeros((47, 10)) * np.NaN
young_htk_yr = np.zeros((47, 10)) * np.NaN

old_prediction_yr = np.zeros((47, 10)) * np.NaN   # pref, year
young_prediction_yr = np.zeros((47, 10)) * np.NaN   # pref, year

pseudo_old_prediction_yr = np.zeros((2, 47, 10)) * np.NaN   # age, pref, year
pseudo_young_prediction_yr = np.zeros((2, 47, 10)) * np.NaN   # age, pref, year

warming_old_prediction_yr = np.zeros((10, 47, 10)) * np.NaN   # warming, pref, year
warming_young_prediction_yr = np.zeros((10, 47, 10)) * np.NaN   # warming, pref, year

warming_old_prediction_real_yr = np.zeros((10, 47, 10)) * np.NaN   # warming, pref, year
warming_young_prediction_real_yr = np.zeros((10, 47, 10)) * np.NaN   # warming, pref, year

for yy in range(10):

    # old simulation
    old_prediction_yr[:, yy] = np.nansum(old_prediction[:, yy * 122: (yy + 1) * 122], axis=1)
    pseudo_old_prediction_yr[:, :, yy] = np.nansum(pseudo_old_prediction[:, :, yy * 122: (yy + 1) * 122],
                                                   axis=2)
    warming_old_prediction_yr[:, :, yy] = np.nansum(warming_old_prediction[:, :, yy * 122: (yy + 1) * 122],
                                                    axis=2)
    warming_old_prediction_real_yr[:, :, yy] = np.nansum(warming_old_prediction_real[:, :, yy * 122: (yy + 1) * 122],
                                                         axis=2)

    # young simulation
    young_prediction_yr[:, yy] = np.nansum(young_prediction[:, yy * 122: (yy + 1) * 122], axis=1)
    pseudo_young_prediction_yr[:, :, yy] = np.nansum(pseudo_young_prediction[:, :, yy * 122: (yy + 1) * 122],
                                                     axis=2)
    warming_young_prediction_yr[:, :, yy] = np.nansum(warming_young_prediction[:, :, yy * 122: (yy + 1) * 122],
                                                      axis=2)
    warming_young_prediction_real_yr[:, :, yy] = np.nansum(warming_young_prediction_real[:, :, yy * 122: (yy + 1) * 122],
                                                           axis=2)

    # observations
    old_htk_yr[:, yy] = np.nansum(old_htk[:, yy * 122: (yy + 1) * 122], axis=1)
    young_htk_yr[:, yy] = np.nansum(young_htk[:, yy * 122: (yy + 1) * 122], axis=1)


pref_47 = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima',
           'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',
           'Niigata', 'Toyama', 'Ishikawa', 'Fukui', 'Yamanashi', 'Nagano', 'Gifu',
           'Shizuoka', 'Aichi', 'Mie', 'Shiga', 'Kyoto', 'Osaka', 'Hyogo',
           'Nara', 'Wakayama', 'Tottori', 'Shimane', 'Okayama', 'Hiroshima', 'Yamaguchi',
           'Tokushima', 'Kagawa', 'Ehime', 'Kochi', 'Fukuoka', 'Saga', 'Nagasaki',
           'Kumamoto', 'Oita', 'Miyazaki', 'Kagoshima', 'Okinawa', 'Japan']


#######################################################################################################################
# Warming VS Aging

total_prediction_yr = old_prediction_yr + young_prediction_yr
warming_total_prediction_yr = warming_old_prediction_yr + warming_young_prediction_yr
pseudo_total_prediction_yr = pseudo_old_prediction_yr + pseudo_young_prediction_yr


fig1 = plt.figure(1, figsize=(one_clm, 3), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig1,
                       nrows=1,
                       ncols=1,
                       height_ratios=[1],
                       width_ratios=[1])

ax = fig1.add_subplot(gs[0, 0])

ax.bar(np.linspace(1, 10, 10),
       np.nansum(warming_total_prediction_yr, axis=(1, 2)) / 10,
       facecolor=[0.8, 0.8, 0.8],
       edgecolor='k',
       lw=0.4)
plt.xticks(np.linspace(1, 10, 10),
           ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=8)
plt.yticks([np.nansum(pseudo_total_prediction_yr[0, :, :]) / 10, 55000, 65000, 75000, 85000],
           ['Scenario\n1995', '55000', '65000', '75000', '85000'],
           fontsize=8)
plt.xlabel(r'Warming level ($^{\mathrm{o}}$C)', fontsize=8)
plt.ylabel('Total HT-EADs of Japan (Cases/Yr)', fontsize=8)

plt.axis([0.5, 10.5,
          np.nansum(pseudo_total_prediction_yr[0, :, :]) / 10,
          85000])

# present
ax.axhline(np.nansum(total_prediction_yr) / 10,
           color='tab:orange', linewidth=0.8, linestyle='--',
           label='Realistic')
# 2045
ax.axhline(np.nansum(pseudo_total_prediction_yr[1, :, :]) / 10,
           color='tab:red', linewidth=0.8, linestyle='--',
           label='Scenario 2045')
plt.legend(fontsize='x-small',
           loc='upper left',)

plt.show()

fig1.savefig(fig_dir + 'Figure_5.svg',
             dpi=1200,
             format='svg')

print('All Finished.')

