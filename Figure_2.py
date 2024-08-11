# Project: HTTK-Japan
# Project start time: 2023/11/18
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to show the performance of GAM in
# predicting all heatstroke (Figure 2)

# for Nature Journals (1 inch = 2.54 cm):
# one-column figure width = 88 mm ~ 3.46 inch,
# two-column figure width = 180 mm ~ 7.08 inch

import numpy as np
import rpy2.robjects as ro
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec


# mpl.use('Qt5Agg')
plt.close('all')
one_clm = 3.46
two_clm = 7.08

# induce some R function
rread = ro.r['load']
ls = ro.r('ls()')
levels = ro.r('levels')

model_input = '/work/a07/qiang/Aging_Japan/02_model/Input_Data/'
fig_dir = '/work/a07/qiang/Aging_Japan/04_plot/fig_test/'
input_dir = '/work/a07/qiang/Aging_Japan/climate_data/'

rread(model_input + 'Old_Young_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')
# Old
old_prediction = np.array(ro.r('old_prediction'))  # (10, 10, 47, 1220)
old_prediction[:, 1:9, :, :] = np.array(ro.r('old_prediction'))[:, 1:9, :, :][:, ::-1, :, :]
# Young
young_prediction = np.array(ro.r('young_prediction'))  # (10, 10, 47, 1220)
young_prediction[:, 1:9, :, :] = np.array(ro.r('young_prediction'))[:, 1:9, :, :][:, ::-1, :, :]


with np.load(model_input + 'model_input.npz') as file:
    date_array_summer = file['date_array_summer']  # (1220, 5)
    # 'Newborn', 'Baby', 'Teenager', 'Adult', 'Elderly', 'Unclear'
    age_htk = file['age_htk']  # (6, 47, 1220)

old_htk = age_htk[4, :, :]  # (47, 1220)
young_htk = np.sum(age_htk[[0, 1, 2, 3], :, :], axis=0)  # (47, 1220)

pref_47 = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima',
           'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',
           'Niigata', 'Toyama', 'Ishikawa', 'Fukui', 'Yamanashi', 'Nagano', 'Gifu',
           'Shizuoka', 'Aichi', 'Mie', 'Shiga', 'Kyoto', 'Osaka', 'Hyogo',
           'Nara', 'Wakayama', 'Tottori', 'Shimane', 'Okayama', 'Hiroshima', 'Yamaguchi',
           'Tokushima', 'Kagawa', 'Ehime', 'Kochi', 'Fukuoka', 'Saga', 'Nagasaki',
           'Kumamoto', 'Oita', 'Miyazaki', 'Kagoshima', 'Okinawa']


######################################################################################################################
# Time series
fig1 = plt.figure(1, figsize=(two_clm, 6), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig1,
                       nrows=3,
                       ncols=2,
                       height_ratios=[1,
                                      1,
                                      1],
                       width_ratios=[1, 1])

lw = 0.9
vali_period = 8
ct_sum = np.array([12, 26, 22])
ct_name_1 = ['Tokyo (Older)', 'Osaka (Older)', 'Aichi (Older)']
ct_name_2 = ['Tokyo (Younger)', 'Osaka (Younger)', 'Aichi (Younger)']
no_sum_1 = ['a', 'c', 'e']
no_sum_2 = ['b', 'd', 'f']
grey_it = 0.7

###############################################
# Old

for ii in range(3):
    ax = fig1.add_subplot(gs[ii, 0])
    plt.title(ct_name_1[ii], fontsize=8.6)

    plt.bar(np.linspace(0, 121, 122),
            old_htk[ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
            facecolor=[grey_it, grey_it, grey_it], label='OBS', edgecolor=[grey_it, grey_it, grey_it])
    plt.plot(np.linspace(0, 121, 122),
             old_prediction[vali_period, 6, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color='orange', label='WBGT', linewidth=lw)
    plt.plot(np.linspace(0, 121, 122),
             old_prediction[vali_period, 9, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color='g', label='CLM', linewidth=lw)

    plt.xticks([0, 30, 61, 92], ['Jun', 'Jul', 'Aug', 'Sep'], fontsize=8)
    plt.axis([0, 122, -10, 250])
    plt.text(4, 230, no_sum_1[ii], fontsize=8, fontweight='bold')
    plt.yticks([0, 100, 200], [0, 100, 200], fontsize=8)
    plt.ylabel('HT-EADs (cases/d)', fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.2)

    # legend
    if ii == 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        lgd = plt.legend(handles=[handles[2], handles[0], handles[1]],
                         fontsize='small', ncol=1,
                         loc='upper right',
                         borderpad=0.4, labelspacing=0.1, columnspacing=0.3, handletextpad=0.2)

#####################################################################
# Young

for ii in range(3):
    ax = fig1.add_subplot(gs[ii, 1])
    plt.title(ct_name_2[ii], fontsize=8.6)

    plt.bar(np.linspace(0, 121, 122),
            young_htk[ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
            facecolor=[grey_it, grey_it, grey_it], label='OBS', edgecolor=[grey_it, grey_it, grey_it])
    plt.plot(np.linspace(0, 121, 122),
             young_prediction[vali_period, 6, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color='orange', label='WBGT', linewidth=lw)
    plt.plot(np.linspace(0, 121, 122),
             young_prediction[vali_period, 9, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color='g', label='CLM', linewidth=lw)

    plt.xticks([0, 30, 61, 92], ['Jun', 'Jul', 'Aug', 'Sep'], fontsize=8)
    plt.axis([0, 122, -10, 250])
    plt.text(4, 230, no_sum_2[ii], fontsize=8, fontweight='bold')

    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)


plt.show()

fig1.savefig(fig_dir + 'Figure_2.svg',
             dpi=1200,
             format='svg')

print('All Finished.')






