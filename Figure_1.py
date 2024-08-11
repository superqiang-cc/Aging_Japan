# Project: HTTK-Japan
# Project start time: 2023/11/18
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot Figure 1


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

httk_data_dir = '/work/a07/qiang/Aging_Japan/httk_data/'
fig_dir = '/work/a07/qiang/Aging_Japan/04_plot/fig_test/'
plot_dir = '/work/a07/qiang/Aging_Japan/04_plot/'
jp_hsi_dir = '/work/a07/qiang/HSI_Japan/Japan_Data/HSIs_1980_2019/'
input_dir = '/work/a07/qiang/Aging_Japan/climate_data/'

# # Load heatstroke data
with np.load(httk_data_dir + 'heatstroke_2010-2023.npz') as file:
    all_htk = file['all_htk']  # (47, 1708): pref, date
    age_htk = file['age_htk']  # (6, 47, 1708): age class (新生児, 乳幼児, 少年, 成人, 高齢者, 不明), pref, date

young_htk = np.nansum(age_htk[:4, :, :], axis=0)
old_htk = age_htk[4, :, :]

# Load the population data:  (47, 14)
population_all = np.array(pd.read_excel(input_dir + 'Population_1995-2023.xlsx', header=0, sheet_name='All'))[:, 16:]
population_old = np.array(pd.read_excel(input_dir + 'Population_1995-2023.xlsx', header=0, sheet_name='Old'))[:, 16:]

pp_all_avg = np.nanmean(population_all, axis=1)  # (47,)
pp_old_avg = np.nanmean(population_old, axis=1)
pp_young_avg = pp_all_avg - pp_old_avg

pp_all_jp = np.array(np.nansum(population_all, axis=0), dtype=float)  # (14,)
pp_old_jp = np.array(np.nansum(population_old, axis=0), dtype=float)
pp_young_jp = pp_all_jp - pp_old_jp

# Obtain date matrix
start_date = np.datetime64('2010-01-01')
end_date = np.datetime64('2023-12-31')
dates = np.arange(start_date, end_date + np.timedelta64(1, 'D'), dtype='datetime64[D]')
years = np.array([np.datetime64(date, 'Y') for date in dates])
months = np.array([np.datetime64(date, 'M') for date in dates]) - years.astype('datetime64[M]')
days = dates - np.array([np.datetime64(date, 'M') for date in dates]).astype('datetime64[D]') + 1
date_1023 = np.column_stack((years.astype(int) + 1970, months.astype('datetime64[M]').astype(int) + 1, days.astype(int)))

date_1023 = date_1023[((date_1023[:, 1] > 5) & (date_1023[:, 1] < 10)), :]
date_1723 = date_1023[date_1023[:, 0] > 2016, :]
date_1019 = date_1023[date_1023[:, 0] < 2020, :]

# load Japan shp
jp_shp_file = plot_dir + '/JP_shp/gadm41_JPN_1.shp'
jp_shp = gpd.read_file(jp_shp_file)
jp_shp['geometry'] = jp_shp['geometry'].simplify(0.01)
jp_shp['NAME_1'][12] = 'Hyogo'
jp_shp['NAME_1'][26] = 'Nagasaki'

with np.load(jp_hsi_dir + 'jp_hsi.npz') as file:
    jp_lon = file['jp_lon']  # 47
    jp_lat = file['jp_lat']  # 47
    jp_city = file['jp_city']  # 47

shp_order_toshow_all = np.zeros(47)
shp_order_toshow_young = np.zeros(47)
shp_order_toshow_old = np.zeros(47)
for ct in range(47):
    shp_order = np.where(jp_city == np.array(jp_shp['NAME_1'], dtype='str')[ct])[0]
    shp_order_toshow_all[ct] = np.nansum(all_htk[shp_order, :]) / 14
    shp_order_toshow_young[ct] = np.nansum(young_htk[shp_order, :]) / 14
    shp_order_toshow_old[ct] = np.nansum(old_htk[shp_order, :]) / 14

jp_shp['ALL_STK'] = shp_order_toshow_all

# color_sum
color_sum = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
             'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lightcoral', 'teal']


######################################################################################################
# # # Figure 1


def per_capita(row, ts_day, ts_year, map_pf, od1, od2):
    # line
    ax = fig3.add_subplot(gs[row, 0:2])
    plt.plot(np.linspace(1, 1708, 1708, dtype=int), ts_day, c='k', linewidth=0.4)
    plt.plot(np.linspace(61, 1647, 14), ts_year, 'o', ms=1.5, ls='-', c='tab:blue', linewidth=1)  # all annual mean
    plt.axis([-20, 1730, 0, 5.5])
    for yy in range(1, 14):
        ax.axvline(yy * 122, ymin=0, ymax=5.5, linestyle='--', c='grey', linewidth=0.5)
    plt.xticks(np.linspace(61, 1647, 14), np.linspace(2010, 2023, 14, dtype=int), fontsize=8)
    plt.yticks([0, 1, 2, 3, 4, 5],
               [0, 1, 2, 3, 4, 5],
               fontsize=8)
    ax.set_ylabel('HT-EADs per 100,000 people / Day', fontsize=8)
    if row == 2:
        ax.set_xlabel('Year', fontsize=8)
    plt.text(-11, 5.2, od1, fontsize=8, fontweight='bold')

    # Japan map
    ax = fig3.add_subplot(gs[row, 2], projection=ccrs.LambertConformal())
    ax.set_frame_on(False)

    cmap_ovlp = plt.cm.get_cmap('OrRd', 10)
    bounds_ovlp = np.linspace(0, 1, 11)
    norm_ovlp = mpl.colors.BoundaryNorm(bounds_ovlp, cmap_ovlp.N)

    # Main part
    jp46_shp = jp_shp[jp_shp['NAME_1'] != 'Okinawa']
    jp46_shp.plot(ax=ax, linewidth=0.3, column=map_pf, cmap=cmap_ovlp, norm=norm_ovlp)
    plt.axis([127, 147, 29.7, 46])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([128, 131], [38.5, 38.5], c='k', linewidth=0.5)
    ax.plot([137, 137], [45, 42], c='k', linewidth=0.5)
    ax.plot([131, 137], [38.5, 42], c='k', linewidth=0.5)
    plt.text(128, 45, od2, fontsize=8, fontweight='bold')
    # Tokyo
    plt.scatter(jp_lon[12], jp_lat[12], c='k', s=2)
    plt.text(jp_lon[12] + 2, jp_lat[12] + 2, jp_city[12], fontsize='8')
    plt.plot([jp_lon[12], jp_lon[12] + 2], [jp_lat[12], jp_lat[12] + 2], c='k', linewidth=0.5)

    # Osaka
    plt.scatter(jp_lon[26], jp_lat[26], c='k', s=2)
    plt.text(jp_lon[26], jp_lat[26] - 3, jp_city[26], fontsize='8', horizontalalignment='center')
    plt.plot([jp_lon[26], jp_lon[26]], [jp_lat[26], jp_lat[26] - 2.2], c='k', linewidth=0.5)

    # Nagoya
    plt.scatter(jp_lon[22], jp_lat[22], c='k', s=2)
    plt.text(jp_lon[22] + 2, jp_lat[22] - 2, jp_city[22], fontsize='8')
    plt.plot([jp_lon[22], jp_lon[22] + 1.8], [jp_lat[22], jp_lat[22] - 1.5], c='k', linewidth=0.5)

    # Okinawa
    okinawa_shp = jp_shp[jp_shp['NAME_1'] == 'Okinawa']
    ax_oknw = ax.inset_axes([0.01, 0.5, 0.6, 0.6])
    okinawa_shp.plot(ax=ax_oknw, linewidth=0.3, column=map_pf, cmap=cmap_ovlp, norm=norm_ovlp)
    ax_oknw.set_xlim(123, 130)
    ax_oknw.set_ylim(24, 28)
    ax_oknw.set_xticks([])
    ax_oknw.set_yticks([])
    ax_oknw.set_frame_on(False)
    plt.text(129, 43, 'Okinawa', fontsize=8)

    # Colorbar
    ax_cb = ax.inset_axes([0.80, 0.02, 0.04, 0.3])
    cb = colorbar.ColorbarBase(ax_cb,
                               cmap=cmap_ovlp,
                               norm=norm_ovlp,
                               orientation='vertical',
                               ticks=list(np.linspace(0, 1, 6)))
    cb.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=7)
    # cb.set_label('Heatstoke cases', fontsize=8)

    ax_cb.set_frame_on(False)


####################################################################################################################
all_year = np.zeros(14) * np.NaN
young_year = np.zeros(14) * np.NaN
old_year = np.zeros(14) * np.NaN

all_jp = np.nansum(all_htk, axis=0)
young_jp = np.nansum(young_htk, axis=0)
old_jp = np.nansum(old_htk, axis=0)

for yy in range(2010, 2024):
    yy_idx = np.where(date_1023[:, 0] == yy)
    all_year[yy - 2010] = np.nanmean(all_jp[yy_idx])
    young_year[yy - 2010] = np.nanmean(young_jp[yy_idx])
    old_year[yy - 2010] = np.nanmean(old_jp[yy_idx])


jp_pp_ts = np.zeros((1708, 3)) * np.NaN  # all, old, young
for yy in range(2010, 2024):
    yy_index = np.where(date_1023[:, 0] == yy)
    jp_pp_ts[yy_index, 0] = pp_all_jp[yy - 2010]
    jp_pp_ts[yy_index, 1] = pp_old_jp[yy - 2010]
    jp_pp_ts[yy_index, 2] = pp_young_jp[yy - 2010]


shp_order_toshow_all = np.zeros(47)
shp_order_toshow_young = np.zeros(47)
shp_order_toshow_old = np.zeros(47)
for ct in range(47):
    shp_order = np.where(jp_city == np.array(jp_shp['NAME_1'], dtype='str')[ct])[0]
    shp_order_toshow_all[ct] = np.nansum(all_htk[shp_order, :]) / 14 / pp_all_avg[shp_order] * 100
    shp_order_toshow_young[ct] = np.nansum(young_htk[shp_order, :]) / 14 / pp_young_avg[shp_order] * 100
    shp_order_toshow_old[ct] = np.nansum(old_htk[shp_order, :]) / 14 / pp_old_avg[shp_order] * 100

jp_shp['ALL_STK'] = shp_order_toshow_all / 122
jp_shp['YOUNG_STK'] = shp_order_toshow_young / 122
jp_shp['OLD_STK'] = shp_order_toshow_old / 122

fig3 = plt.figure(3, figsize=(two_clm, 6.5), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig3,
                       nrows=3,
                       ncols=3,
                       height_ratios=[1,
                                      1,
                                      1],
                       width_ratios=[1, 1, 0.8])

# all
per_capita(0,
           np.nansum(all_htk, axis=0) / jp_pp_ts[:, 0] * 100,
           all_year / pp_all_jp * 100,
           'ALL_STK',
           'a', 'b')

# old
per_capita(1,
           np.nansum(old_htk, axis=0) / jp_pp_ts[:, 1] * 100,
           old_year / pp_old_jp * 100,
           'OLD_STK',
           'c', 'd')

# young
per_capita(2,
           np.nansum(young_htk, axis=0) / jp_pp_ts[:, 2] * 100,
           young_year / pp_young_jp * 100,
           'YOUNG_STK',
           'e', 'f')


plt.show()

fig3.savefig(fig_dir + 'Figure_1.svg',
             format='svg',
             dpi=1200)

print('All Finished.')




















