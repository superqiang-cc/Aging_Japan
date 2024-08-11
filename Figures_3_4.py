# Project: HTTK-Japan
# Project start time: 2023/11/18
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot the pseudo prediction under Scenarios 1995 and 2045 (Figures 3 and 4)

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
plot_dir = '/work/a07/qiang/Aging_Japan/04_plot/'
jp_hsi_dir = '/work/a07/qiang/HSI_Japan/Japan_Data/HSIs_1980_2019/'
input_dir = '/work/a07/qiang/Aging_Japan/climate_data/'

######################################################################################
# Load old and young prediction
rread(pdt_dir + 'Pseudo_old_young_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')

old_prediction = np.array(ro.r('old_prediction'))  # (47, 1220)

pseudo_old_prediction = np.array(ro.r('pseudo_old_prediction'))  # (2, 47, 1220)

young_prediction = np.array(ro.r('young_prediction'))  # (47, 1220)

pseudo_young_prediction = np.array(ro.r('pseudo_young_prediction'))  # (2, 47, 1220)

# (6, 47, 1220)  'Newborn', 'Baby', 'Teenager', 'Adult', 'Elderly', 'Unclear'
age_htk = np.load(pdt_dir + 'model_input.npz')['age_htk']
old_htk = age_htk[4, :, :]
young_htk = np.nansum(age_htk[:4, :, :], axis=0)

old_htk_yr = np.zeros((47, 10)) * np.NaN
young_htk_yr = np.zeros((47, 10)) * np.NaN

old_prediction_yr = np.zeros((47, 10)) * np.NaN   # pref, year
pseudo_old_prediction_yr = np.zeros((2, 47, 10)) * np.NaN   # oy, pref, year

young_prediction_yr = np.zeros((47, 10)) * np.NaN   # pref, year
pseudo_young_prediction_yr = np.zeros((2, 47, 10)) * np.NaN   # oy, pref, year

for yy in range(10):

    # old simulation
    old_prediction_yr[:, yy] = np.nansum(old_prediction[:, yy * 122: (yy + 1) * 122], axis=1)
    pseudo_old_prediction_yr[:, :, yy] = np.nansum(pseudo_old_prediction[:, :, yy * 122: (yy + 1) * 122],
                                                   axis=2)

    # young simulation
    young_prediction_yr[:, yy] = np.nansum(young_prediction[:, yy * 122: (yy + 1) * 122], axis=1)
    pseudo_young_prediction_yr[:, :, yy] = np.nansum(pseudo_young_prediction[:, :, yy * 122: (yy + 1) * 122],
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
# Time Series
fig1 = plt.figure(1, figsize=(two_clm, 6), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig1,
                       nrows=2,
                       ncols=2,
                       height_ratios=[1, 1],
                       width_ratios=[1, 1])

oy = 0

###########################################################
# old
ax = fig1.add_subplot(gs[0, 0])
lw_set = 1.2
# Observations
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(old_htk_yr, axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='k', label='Observations')
# All prediction
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(old_prediction_yr[:, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:blue', label='Realistic')
# Pseudo prediction
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(pseudo_old_prediction_yr[0, :, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:green', label='Scenario 1995')
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(pseudo_old_prediction_yr[1, :, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:orange', label='Scenario 2045')
ax.axhline(y=20000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=30000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=40000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=50000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=60000, c='grey', linestyle='--', lw=0.6)
plt.axis([2009.5, 2019.5, 10000, 65000])
plt.text(2009.6, 62000, 'a', fontsize=8, fontweight='bold')

handles, labels = plt.gca().get_legend_handles_labels()

plt.legend(handles=[handles[0], handles[1], handles[2], handles[3]],
           loc='upper center', ncol=1, fontsize='small')

plt.xlabel('Year', fontsize=9)
plt.ylabel('Older HT-EADs (cases/y)', fontsize=9)
plt.xticks(np.linspace(2010, 2019, 10, dtype=int),
           np.linspace(2010, 2019, 10, dtype=int),
           fontsize=8)
plt.yticks([10000, 20000, 30000, 40000, 50000, 60000],
           ['10k', '20k', '30k', '40k', '50k', '60k'],
           fontsize=8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

###########################################################
# Young
ax = fig1.add_subplot(gs[0, 1])
# Observations
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(young_htk_yr, axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='k', label='Observations')
# All prediction
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(young_prediction_yr[:, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:blue', label='Realistic')
# Pseudo prediction
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(pseudo_young_prediction_yr[0, :, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:green', label='Scenario 1995')
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(pseudo_young_prediction_yr[1, :, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:orange', label='Scenario 2045')
ax.axhline(y=20000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=30000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=40000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=50000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=60000, c='grey', linestyle='--', lw=0.6)
plt.axis([2009.5, 2019.5, 10000, 65000])
plt.text(2009.6, 62000, 'b', fontsize=8, fontweight='bold')

plt.xlabel('Year', fontsize=9)
plt.ylabel('Younger HT-EADs (cases/y)', fontsize=9)
plt.xticks(np.linspace(2010, 2019, 10, dtype=int),
           np.linspace(2010, 2019, 10, dtype=int),
           fontsize=8)
plt.yticks([10000, 20000, 30000, 40000, 50000, 60000],
           ['10k', '20k', '30k', '40k', '50k', '60k'],
           fontsize=8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

###########################################################
# Young + old
ax = fig1.add_subplot(gs[1, 0])
# Observations
plt.plot(np.linspace(2010, 2019, 10, dtype=int), np.nansum(young_htk_yr, axis=0) + np.nansum(old_htk_yr, axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='k', label='Observations')
# All prediction
plt.plot(np.linspace(2010, 2019, 10, dtype=int),
         np.nansum(young_prediction_yr[:, :], axis=0) + np.nansum(old_prediction_yr[:, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:blue', label='Realistic')
# Pseudo prediction
plt.plot(np.linspace(2010, 2019, 10, dtype=int),
         np.nansum(pseudo_young_prediction_yr[0, :, :], axis=0)
         + np.nansum(pseudo_old_prediction_yr[0, :, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:green', label='Scenario 1995')
plt.plot(np.linspace(2010, 2019, 10, dtype=int),
         np.nansum(pseudo_young_prediction_yr[1, :, :], axis=0)
         + np.nansum(pseudo_old_prediction_yr[1, :, :], axis=0),
         'o', ls='-', ms=2,
         lw=lw_set, c='tab:orange', label='Scenario 2045')
ax.axhline(y=30000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=40000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=50000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=60000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=70000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=80000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=90000, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=100000, c='grey', linestyle='--', lw=0.6)
plt.axis([2009.5, 2019.5, 30000, 105000])
plt.text(2009.6, 102000, 'c', fontsize=8, fontweight='bold')

plt.xlabel('Year', fontsize=9)
plt.ylabel('Overall HT-EADs (Cases/y)', fontsize=9)
plt.xticks(np.linspace(2010, 2019, 10, dtype=int),
           np.linspace(2010, 2019, 10, dtype=int),
           fontsize=8)
plt.yticks([30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
           ['30k', '40k', '50k', '60k', '70k', '80k', '90k', '100k'],
           fontsize=8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# plt.show()
fig1.savefig(fig_dir + 'Figure_3.svg',
             dpi=1200,
             format='svg')

######################################################################################################################
# # # Relative difference between present and pseudo
fig3 = plt.figure(3, figsize=(two_clm, 7), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig3,
                       nrows=2,
                       ncols=2,
                       height_ratios=[1,
                                      0.03],
                       width_ratios=[1, 1])

all_prediction_yr = old_prediction_yr + young_prediction_yr
pseudo_all_prediction_yr = pseudo_old_prediction_yr + pseudo_young_prediction_yr

prediction_data = all_prediction_yr
pseudo_prediction_data = pseudo_all_prediction_yr

# 2010-2019
all_htk_1019 = np.mean(prediction_data[:, :], axis=1)
pseudo_htk_1019_1995 = np.mean(pseudo_prediction_data[0, :, :], axis=1)
pseudo_htk_1019_2045 = np.mean(pseudo_prediction_data[1, :, :], axis=1)

htk_sim_1019 = {
    'Realistic vs Scenario 1995': (all_htk_1019 - pseudo_htk_1019_1995) / pseudo_htk_1019_1995 * 100,
    'Scenario 2045 vs Scenario 1995': (pseudo_htk_1019_2045 - pseudo_htk_1019_1995) / pseudo_htk_1019_1995 * 100,
}

htk_sim_1019_jp = {
    'Realistic vs Scenario 1995': (np.mean(all_htk_1019) - np.mean(pseudo_htk_1019_1995)) / np.mean(pseudo_htk_1019_1995) * 100,
    'Scenario 2045 vs Scenario 1995': (np.mean(pseudo_htk_1019_2045) - np.mean(pseudo_htk_1019_1995)) / np.mean(pseudo_htk_1019_1995) * 100,
}

color_sum = ['tab:blue', 'tab:orange']

width = 0.4  # the width of the bars
multiplier = 0

ax = fig3.add_subplot(gs[0, 0])

x = np.arange(1, 48)
for attribute, measurement in htk_sim_1019.items():
    offset = width * multiplier
    rects = ax.barh(x + offset, measurement, width, label=attribute, facecolor=color_sum[multiplier],
                    edgecolor='k', lw=0.3)
    multiplier += 1

x = 48.5
multiplier = 0
for attribute, measurement in htk_sim_1019_jp.items():
    offset = 0.4 * multiplier
    rects = ax.barh(x + offset, measurement, 0.4, label=attribute, facecolor=color_sum[multiplier],
                    edgecolor='k', lw=0.3)
    multiplier += 1
ax.axhline(y=48, c='grey', linestyle='--', lw=0.6)

plt.yticks(np.append(np.linspace(1, 47, 47) + width / 2, x + 0.2), pref_47, fontsize=8)
plt.xticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], fontsize=8)
ax.axvline(x=10, c='grey', linestyle='--', lw=0.6)
ax.axvline(x=20, c='grey', linestyle='--', lw=0.6)
ax.axvline(x=30, c='grey', linestyle='--', lw=0.6)
ax.axvline(x=40, c='grey', linestyle='--', lw=0.6)
plt.xlabel('Relative increase (%)', fontsize=8)
plt.axis([0, 50, 0.5, 49.5])
plt.text(47.5, 1.5, 'a', fontsize=8, fontweight='bold')
plt.gca().invert_yaxis()

# 2018
all_htk_1719 = prediction_data[:, 8]
pseudo_htk_1719_1995 = pseudo_prediction_data[0, :, 8]
pseudo_htk_1719_2045 = pseudo_prediction_data[1, :, 8]

htk_sim_1719 = {
    'Realistic vs Scenario 1995': (all_htk_1719 - pseudo_htk_1719_1995) / pseudo_htk_1719_1995 * 100,
    'Scenario 2045 vs Scenario 1995': (pseudo_htk_1719_2045 - pseudo_htk_1719_1995) / pseudo_htk_1719_1995 * 100,
}

htk_sim_1719_jp = {
    'Realistic vs Scenario 1995': (np.mean(all_htk_1719) - np.mean(pseudo_htk_1719_1995)) / np.mean(pseudo_htk_1719_1995) * 100,
    'Scenario 2045 vs Scenario 1995': (np.mean(pseudo_htk_1719_2045) - np.mean(pseudo_htk_1719_1995)) / np.mean(pseudo_htk_1719_1995) * 100,
}

color_sum = ['tab:blue', 'tab:orange']

ax = fig3.add_subplot(gs[0, 1])

multiplier = 0
x = np.arange(1, 48)
for attribute, measurement in htk_sim_1719.items():
    offset = width * multiplier
    rects = ax.barh(x + offset, measurement, width, label=attribute, facecolor=color_sum[multiplier],
                    edgecolor='k', lw=0.3)
    multiplier += 1
handles_oy, labels = plt.gca().get_legend_handles_labels()

x = 48.5
multiplier = 0
for attribute, measurement in htk_sim_1719_jp.items():
    offset = 0.4 * multiplier
    rects = ax.barh(x + offset, measurement, 0.4, label=attribute, facecolor=color_sum[multiplier],
                    edgecolor='k', lw=0.3)
    multiplier += 1
ax.axhline(y=48, c='grey', linestyle='--', lw=0.6)

plt.yticks(np.append(np.linspace(1, 47, 47) + width / 2, x + 0.2), pref_47, fontsize=8)
plt.xticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], fontsize=8)
ax.axvline(x=10, c='grey', linestyle='--', lw=0.6)
ax.axvline(x=20, c='grey', linestyle='--', lw=0.6)
ax.axvline(x=30, c='grey', linestyle='--', lw=0.6)
ax.axvline(x=40, c='grey', linestyle='--', lw=0.6)
plt.xlabel('Relative increase (%)', fontsize=8)
plt.axis([0, 50, 0.5, 49.5])
plt.text(47.5, 1.5, 'b', fontsize=8, fontweight='bold')
plt.gca().invert_yaxis()

# legend
axlgd = fig3.add_subplot(gs[1, :])
lgd = plt.legend(handles=handles_oy,
                 fontsize='small', ncol=2,
                 loc='lower center',
                 borderpad=0.3, labelspacing=0.2, columnspacing=2, handletextpad=0.2)
axlgd.set_frame_on(False)
axlgd.set_xticks([])
axlgd.set_yticks([])

plt.show()

fig3.savefig(fig_dir + 'Figure_4.svg',
             dpi=1200,
             format='svg')


