import sys 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sswm import SM_C_H

from scipy.stats import spearmanr, pearsonr

from pickle import dump, load
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, percentileofscore
from sklearn.linear_model import LinearRegression


full_param_name_list = ['T_GS', 'rf_alpha', 'rf_lambda', 'Eo', 'Td',
                        'LAI', 'RAI', 'hc', 'Zm',
                        'Ps0', 'b', 'Ks', 'n', 's_h', 's_fc', 
                        'k_xl_max', 'Px50', 'Pg50']


soil_tex_list =[[11.4, 0.405, 1.28E-06, 0.482, 'CLAY'],
                [8.52, 0.63, 2.45E-06, 0.476, 'CLAY LOAM'],
                [7.12, 0.299, 6.30E-06, 0.42, 'SANDY CLAY LOAM'],
                [5.39, 0.478, 6.95E-06, 0.451, 'LOAM'],
                [4.9, 0.218, 3.47E-05, 0.435, 'SANDY LOAM'],
                [4.38, 0.09, 1.56E-04, 0.41, 'LOAMY SAND'],
                [4.05, 0.121 ,1.76E-04, 0.395, 'SAND']]

s_list = np.linspace(0, 1, 101)


def cal_nse(obs, mod):
    mo = np.nanmean(obs)
    a = np.nansum([(mi - oi) ** 2 for mi, oi in zip(mod, obs)])
    b = np.nansum([(oi - mo) ** 2 for oi in obs])
    return 1 - a / b


def cal_mape(obs, mod):
    mo = np.nanmean(obs)
    ape = [np.abs(mi - oi) / oi for mi, oi in zip(mod, obs)]
    return np.nanmean(ape)


def cal_rmse(obs, mod):
    mo = np.nanmean(obs)
    err = [(mi - oi)**2 for mi, oi in zip(mod, obs)]
    return (np.nanmean(err))**0.5


def cal_meanbias(obs, mod):
    b = [(mi - oi) for mi, oi in zip(mod, obs)]
    return np.nanmean(b)


def test_m(params, pii, test_n):
    params[test_n] = pii
    model_p = [params[p] for p in full_param_name_list]
    smpdf = SM_C_H(model_p, f_wilt=0.05, f_star=0.95, constraints=False)
    return smpdf, params


def test_mp(params, vii, test_v, pi_vary):

    pii0 = vii /params[test_v]
    params[test_v] = vii
    if pi_vary != 'PiR':
        params['Px50'] = params['Px50'] * pii0
    if pi_vary != 'PiF':
        params['k_xl_max'] = params['k_xl_max'] / pii0
    if pi_vary != 'PiT':
        params['Ks'] = params['Ks'] / pii0 
    if pi_vary != 'PiS':
        params['Ps0'] = params['Ps0'] * pii0
        params['b'] = (np.log(10) - np.log(-params['Ps0'])) / (np.log(1) - np.log(params['s_h']))

    model_p = [params[p] for p in full_param_name_list]
    smpdf = SM_C_H(model_p, f_wilt=0.05, f_star=0.95,  constraints=False)
    return smpdf, params


def cal_list_beta(smpdf):
    bs = [smpdf.loss_T(si) / (smpdf.Eo / (smpdf.n * smpdf.Zm)) / (1 - smpdf.phiE) for si in s_list]
    return bs


def get_baseline_params(soil_i=3):
    b, Ps0, Ksat, n, soiltex = soil_tex_list[soil_i]
    Ps0 = Ps0 * -9.8067 * 10 ** -3 # m to MPa
    Ksat = Ksat * 12 * 60 * 60
    s_h = (-10/Ps0)**(-1/b)
    s_fc = (-0.03/Ps0)**(-1/b)
    params0 = {
         'Zm': 0.5,
         'Td': 12 * 60 * 60,
         'Eo': 0.004,
         'rf_alpha': 0.007,
         'rf_lambda': 0.35,
         'b': b,
         'Ps0': Ps0 ,
         'Ks': Ksat,
         'n': n,
         's_h': s_h,
         's_fc': s_fc,
         'RAI': 10,
         'dr': 0.0005,
         'hc': 20,
         'k_xl_max': 0.0008,
         'Px50': -2.5,
         'Pg50': -1.5,
         'LAI': 2, 
         'T_GS': 180}
    
    return params0


def plot_SI_v_range(params, test_p, test_n, test_nn, legend=0, cmap='viridis_r'):
    cmap = plt.cm.get_cmap(cmap)
    fig = plt.figure(figsize=(6, 5))

    axB = plt.subplot2grid((5,6), (0, 0), colspan=5, rowspan=5)
    axA = plt.subplot2grid((5, 6), (0, 5), colspan=1, rowspan=1)
    axR = plt.subplot2grid((5, 6), (1, 5), colspan=1, rowspan=1)
    axF = plt.subplot2grid((5, 6), (2, 5), colspan=1, rowspan=1)
    axT = plt.subplot2grid((5, 6), (3, 5), colspan=1, rowspan=1)
    axS = plt.subplot2grid((5, 6), (4, 5), colspan=1, rowspan=1)
    
    s_list = np.linspace(0, 1, 101)

    for ax in [axA, axR, axF,axT, axS]:
        ax.yaxis.tick_right()
        ax.get_yaxis().set_label_coords(1.8,0.5)
        ax.yaxis.set_label_position('right')
        ax.tick_params(direction='inout')
        ax.set_xlim([test_p[0], test_p[-1]])
        ax.set_xticks([])

    for ix, pii in enumerate(test_p):
        try:
            color = cmap((1 + ix) / (np.float(len(test_p) + 1)))
            smpdf, xparams = test_m(params, pii, test_n)
            et_i = cal_list_beta(smpdf)

            axB.plot(s_list, et_i, color=color, lw=2)
            axB.set_xticks([0.2, 0.4, 0.6, 0.8])
            axB.set_yticks([0, 0.5, 1])
            axB.set_ylim([0, 1])
            axB.set_xlim([0.2, 0.8])

            axB.set_xlabel('Soil saturation', fontsize=14)
            axB.set_ylabel(r"$\beta(s) = T/E_0$", fontsize=14)
            axB.set_title('(a)', fontsize=16)

            axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
            axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
            axA.set_ylim([0, 1])
            axA.set_yticks([0, 0.5])
            axA.set_yticklabels(['0', '0.5'])
            axA.set_title('(b)', fontsize=16)

            axR.plot(pii, smpdf.pi_R, linestyle='', marker='.',  color=color)
            axR.set_ylabel(r'$\Pi_R$', fontsize=16, rotation=0)
            axR.set_ylim([0, 1])
            axR.set_yticks([0, 0.5])
            axR.set_yticklabels(['0', '0.5'])

            axF.plot(pii, smpdf.pi_F, linestyle='', marker='.', color=color)
            axF.set_ylabel(r'$\Pi_F$', fontsize=16, rotation=0)
            axF.set_ylim([0, 3])
            axF.set_yticks([0, 1.5])
            axF.set_yticklabels(['0', '1.5'])

            axT.plot(pii, smpdf.pi_T, linestyle='', marker='.', color=color)
            axT.set_ylabel(r'$\Pi_T$', fontsize=16, rotation=0)
            axT.set_ylim([0, 8*10**6])
            axT.set_yticks([0, 4*10**6])
            axT.set_yticklabels(['0', '3 $10^{6}$'])

            axS.plot(pii, smpdf.pi_S, linestyle='', marker='.', color=color)
            axS.set_ylabel(r'$\Pi_S$', fontsize=16, rotation=0)
            axS.set_yticks([0, 250])
            axS.set_ylim([0, 500])
            axS.set_yticklabels(['0', '250'])
            axS.set_xticks([test_p[0], test_p[-1]])
            axS.set_xlabel(test_nn, fontsize=14)
            axS.yaxis.set_label_position('right')
        except:
            pass
    
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_%srange' % test_n
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def plot_SI_pi_range_p(params, test_p, test_n, test_nn, pi_vary, cmap='viridis_r'):
    cmap = plt.cm.get_cmap(cmap)
    fig = plt.figure(figsize=(6, 5))

    axB = plt.subplot2grid((5, 6), (0, 0), colspan=5, rowspan=5)
    axA = plt.subplot2grid((5, 6), (0, 5), colspan=1, rowspan=1)
    axR = plt.subplot2grid((5, 6), (1, 5), colspan=1, rowspan=1)
    axF = plt.subplot2grid((5, 6), (2, 5), colspan=1, rowspan=1)
    axT = plt.subplot2grid((5, 6), (3, 5), colspan=1, rowspan=1)
    axS = plt.subplot2grid((5, 6), (4, 5), colspan=1, rowspan=1)
    
    s_list = np.linspace(params['s_h'], 1, 101)
    p_list = [-params['Ps0'] * s ** (- params['b']) for s in s_list]
    for ax in [axA, axR, axF,axT, axS]:
        ax.yaxis.tick_right()
        ax.get_yaxis().set_label_coords(1.8,0.5)
        ax.yaxis.set_label_position('right')
        ax.tick_params(direction='inout')
        ax.set_xlim([test_p[0], test_p[-1]])
        ax.set_xticks([])

    for ix, pii in enumerate(test_p):
        try:
            color = cmap((1 + ix) / (np.float(len(test_p) + 1)))
            smpdf, xparams = test_mp(params, pii, test_n, pi_vary)
            et_i = cal_list_beta(smpdf)

            axB.plot(p_list, et_i, color=color, lw=2)
            axB.tick_params(direction='inout')
            axB.set_yticks([0, 0.5, 1])
            axB.set_ylim([0, 1])
            axB.set_xlim([5, 0.01])
            axB.set_xscale('log')
            axB.set_xticks([1, 0.1])
            axB.set_xticklabels(['-1', '-0.1'])
            axB.set_xlabel('Soil water potential [MPa]', fontsize=14)
            axB.set_ylabel(r"$\beta(s) = T/E_0$ [-]", fontsize=14)
            axB.set_title('(a)', fontsize=16)

            axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
            axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
            axA.set_ylim([0, 1])
            axA.set_yticks([0, 0.5])
            axA.set_yticklabels(['0', '0.5'])
            axA.set_title('(b)', fontsize=16)

            axR.plot(pii, smpdf.pi_R, linestyle='', marker='.',  color=color)
            axR.set_ylabel(r'$\Pi_R$', fontsize=16, rotation=0)
            axR.set_ylim([0, 1])
            axR.set_yticks([0, 0.5])
            axR.set_yticklabels(['0', '0.5'])

            axF.plot(pii, smpdf.pi_F, linestyle='', marker='.', color=color)
            axF.set_ylabel(r'$\Pi_F$', fontsize=16, rotation=0)
            axF.set_ylim([0, 3])
            axF.set_yticks([0, 1.5])
            axF.set_yticklabels(['0', '1.5'])

            axT.plot(pii, smpdf.pi_T, linestyle='', marker='.', color=color)
            axT.set_ylabel(r'$\Pi_T$', fontsize=16, rotation=0)
            axT.set_ylim([0, 6*10**6])
            axT.set_yticks([0, 3*10**6])
            axT.set_yticklabels(['0', '3 $10^{6}$'])

            axS.plot(pii, smpdf.pi_S, linestyle='', marker='.', color=color)
            axS.set_ylabel(r'$\Pi_S$', fontsize=16, rotation=0)
            axS.set_yticks([0, 250])
            axS.set_ylim([0, 500])
            axS.set_yticklabels(['0', '250'])
            axS.set_xticks([-2, -1])
            axS.set_xlabel(test_nn, fontsize=14)
            axS.yaxis.set_label_position('right')
        except:
            pass
    
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_%s_range_p' % pi_vary
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def plot_SI_pi_range_s(params, test_p, test_n, test_nn, pi_vary, cmap='viridis_r'):
    cmap = plt.cm.get_cmap(cmap)
    fig = plt.figure(figsize=(6, 5))

    axB = plt.subplot2grid((5, 6), (0, 0), colspan=5, rowspan=5)
    axA = plt.subplot2grid((5, 6), (0, 5), colspan=1, rowspan=1)
    axR = plt.subplot2grid((5, 6), (1, 5), colspan=1, rowspan=1)
    axF = plt.subplot2grid((5, 6), (2, 5), colspan=1, rowspan=1)
    axT = plt.subplot2grid((5, 6), (3, 5), colspan=1, rowspan=1)
    axS = plt.subplot2grid((5, 6), (4, 5), colspan=1, rowspan=1)
    
    s_list = np.linspace(params['s_h'], 1, 101)
    p_list = [-params['Ps0'] * s ** (- params['b']) for s in s_list]
    
    for ax in [axA, axR, axF,axT, axS]:
        ax.yaxis.tick_right()
        ax.get_yaxis().set_label_coords(1.8,0.5)
        ax.yaxis.set_label_position('right')
        ax.tick_params(direction='inout')
        ax.set_xlim([test_p[0], test_p[-1]])
        ax.set_xticks([])

    for ix, pii in enumerate(test_p):
        try:
            color = cmap((1 + ix) / (np.float(len(test_p) + 1)))
            smpdf, xparams = test_mp(params, pii, test_n, pi_vary)
            et_i = cal_list_beta(smpdf)

            axB.plot(s_list, et_i, color=color, lw=2)
            axB.tick_params(direction='inout')
            axB.set_yticks([0, 0.5, 1])
            axB.set_ylim([0, 1])
            #axB.set_xlim([5, 0.01])
            #axB.set_xscale('log')
            #axB.set_xticks([1, 0.1])
            #axB.set_xticklabels(['-1', '-0.1'])

            axB.set_xticks([0.4, 0.6, 0.8])
            axB.set_xticklabels([ 0.4, 0.6, 0.8])
            axB.set_xlim([0.35, 0.85])


            axB.set_xlabel('Soil saturation [-]', fontsize=14)
            axB.set_ylabel(r"$\beta(s) = T/E_0$ [-]", fontsize=14)
            axB.set_title('(a)', fontsize=16)

            axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
            axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
            axA.set_ylim([0, 1])
            axA.set_yticks([0, 0.5])
            axA.set_yticklabels(['0', '0.5'])
            axA.set_title('(b)', fontsize=16)

            axR.plot(pii, smpdf.pi_R, linestyle='', marker='.',  color=color)
            axR.set_ylabel(r'$\Pi_R$', fontsize=16, rotation=0)
            axR.set_ylim([0, 1])
            axR.set_yticks([0, 0.5])
            axR.set_yticklabels(['0', '0.5'])

            axF.plot(pii, smpdf.pi_F, linestyle='', marker='.', color=color)
            axF.set_ylabel(r'$\Pi_F$', fontsize=16, rotation=0)
            axF.set_ylim([0, 3])
            axF.set_yticks([0, 1.5])
            axF.set_yticklabels(['0', '1.5'])

            axT.plot(pii, smpdf.pi_T, linestyle='', marker='.', color=color)
            axT.set_ylabel(r'$\Pi_T$', fontsize=16, rotation=0)
            axT.set_ylim([0, 6*10**6])
            axT.set_yticks([0, 3*10**6])
            axT.set_yticklabels(['0', '3 $10^{6}$'])

            axS.plot(pii, smpdf.pi_S, linestyle='', marker='.', color=color)
            axS.set_ylabel(r'$\Pi_S$', fontsize=16, rotation=0)
            axS.set_yticks([0, 250])
            axS.set_ylim([0, 500])
            axS.set_yticklabels(['0', '250'])
            axS.set_xticks([-2, -1])
            axS.set_xlabel(test_nn, fontsize=14)
            axS.yaxis.set_label_position('right')
        except:
            pass
    
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_%s_range_s' % pi_vary
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def plot_SI_soiltex(soil_tex_list, cmap='viridis'):
    cmap = plt.cm.get_cmap(cmap)
    fig = plt.figure(figsize=(6, 5))

    axB = plt.subplot2grid((5,6), (0, 0), colspan=5, rowspan=5)
    axA = plt.subplot2grid((5, 6), (0, 5), colspan=1, rowspan=1)
    axR = plt.subplot2grid((5, 6), (1, 5), colspan=1, rowspan=1)
    axF = plt.subplot2grid((5, 6), (2, 5), colspan=1, rowspan=1)
    axT = plt.subplot2grid((5, 6), (3, 5), colspan=1, rowspan=1)
    axS = plt.subplot2grid((5, 6), (4, 5), colspan=1, rowspan=1)
    
    s_list = np.linspace(0, 1, 101)

    for ax in [axA, axR, axF,axT, axS]:
        ax.yaxis.tick_right()
        ax.get_yaxis().set_label_coords(1.8,0.5)
        ax.yaxis.set_label_position('right')
        ax.tick_params(direction='inout')
        ax.set_xlim([-1, len(soil_tex_list)])
        ax.set_xticks([])

    for ix, (b, Ps0, Ksat, n, soiltex) in enumerate(soil_tex_list):
        pii = ix
        params = get_baseline_params(soil_i=ix)
    
        color = cmap((1 + ix)/(np.float(len(soil_tex_list)+1)))
        smpdf, xparams = test_m(params, pii, 'soil_tex_id')
        et_i = cal_list_beta(smpdf)

        axB.plot(s_list, et_i, color=color, lw=2)
        axB.axvline(smpdf.s_h, color=color, linestyle=':')
        axB.axvline(smpdf.s_fc, color=color, linestyle='--')
        axB.set_yticks([0, 0.5, 1])
        axB.set_ylim([0, 1])
        axB.set_xlim([0, 1])
        axB.set_xlabel('Soil saturation', fontsize=14)
        axB.set_ylabel(r"$\beta(s) = T/E_0$", fontsize=14)
        axB.set_title('(a)', fontsize=16)

        axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
        axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
        axA.set_ylim([0, 1])
        axA.set_yticks([0, 0.5])
        axA.set_yticklabels(['0', '0.5'])
        axA.set_title('(b)', fontsize=16)

        axR.plot(pii, smpdf.pi_R, linestyle='', marker='.',  color=color)
        axR.set_ylabel(r'$\Pi_R$', fontsize=16, rotation=0)
        axR.set_ylim([0, 1])
        axR.set_yticks([0, 0.5])
        axR.set_yticklabels(['0', '0.5'])

        axF.plot(pii, smpdf.pi_F, linestyle='', marker='.', color=color)
        axF.set_ylabel(r'$\Pi_F$', fontsize=16, rotation=0)
        axF.set_ylim([0, 3])
        axF.set_yticks([0, 1.5])
        axF.set_yticklabels(['0', '1.5'])

        axT.plot(pii, smpdf.pi_T, linestyle='', marker='.', color=color)
        axT.set_ylabel(r'$\Pi_T$', fontsize=16, rotation=0)
        axT.set_ylim([0, 60*10**6])
        axT.set_yticks([0, 30*10**6])
        axT.set_yticklabels(['0', '3 $10^{7}$'])

        axS.plot(pii, smpdf.pi_S, linestyle='', marker='.', color=color)
        axS.set_ylabel(r'$\Pi_S$', fontsize=14, rotation=0)
        axS.set_yticks([0, 750])
        axS.set_ylim([0, 1500])
        axS.set_yticklabels(['0', '750'])
        
        axS.set_xticks(range(len(soil_tex_list)))
        axS.set_xticklabels(list(zip(*soil_tex_list))[-1], rotation =90, fontsize=7)

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_soilTex_range'
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def plot_piF_piR_range(params, piF_l, piR_l):
    cmap = plt.cm.get_cmap('viridis_r')
    fig = plt.figure(figsize=(4, 5.5))
    axA = plt.subplot2grid((2, 3),(0, 0), colspan=2, rowspan=1)
    axE = plt.subplot2grid((2, 3),(1, 0), colspan=2, rowspan=1)
    axb1 = plt.subplot2grid((5, 3),(0, 2), colspan=1, rowspan=1)
    axb2 = plt.subplot2grid((5, 3),(1, 2), colspan=1, rowspan=1)
    axb3 = plt.subplot2grid((5, 3),(2, 2), colspan=1, rowspan=1)
    axb4 = plt.subplot2grid((5, 3),(3, 2), colspan=1, rowspan=1)
    axb5 = plt.subplot2grid((5, 3), (4, 2), colspan=1, rowspan=1)
    axb_l = [axb1, axb2, axb3, axb4, axb5]

    for ir, piR in enumerate(piR_l):
        result = []
        c = (piR-np.min(piR_l))/(np.max(piR_l)-np.min(piR_l))
        for aa, piF in enumerate(piF_l):
            aa = aa / np.float(len(piF_l))
            params['Px50'] = params['Pg50'] / piR
            K_p_max = - params['Eo'] / (piF * params['Pg50'])
            k_xl_max = K_p_max / (params['LAI'] / params['hc'] * params['Td'] / 1000)
            params['k_xl_max'] = k_xl_max
            model_p = [params[p] for p in full_param_name_list]
            smpdf = SM_C_H(model_p, f_wilt=0.05, f_star=0.95, constraints=True)
            smpdf_f = SM_C_H(model_p, f_wilt=0.05, f_star=0.95, constraints=False)
            et_i = cal_list_beta(smpdf)
            et_i_f = cal_list_beta(smpdf_f)
            swet2=smpdf.epsilon
            AA2 = smpdf.AA
            piF2 = piF
            axb_l[ir].plot(s_list, et_i_f, color=cmap(c), alpha=aa)

            swet2 = smpdf_f.epsilon
            AA2 = smpdf_f.AA
            piF2 = piF
            result.append([smpdf.AA, AA2, smpdf.epsilon, swet2, piF, piF2])
                
        try:
            AA, AA2, swet, swet2, piF, piF2 = zip(*result)
            axA.plot(piF, AA2, color=cmap(c),  linestyle='-')
            axE.plot(piF, swet2, color=cmap(c), linestyle='-')
            #axA.plot(piF, AA, color=cmap(c), label=r'%-5.2f $\Pi_R$'%(piR), linestyle='-')
            #axE.plot(piF, swet, color=cmap(c), label=r'%-5.2f $\Pi_R$'%(piR), linestyle='-')
        except:
            pass

    for ir, piR in enumerate(piR_l):
        axb_l[ir].text(0.07, 0.25,  r'$\Pi_R$=%-5.2f'%(piR), fontsize=8, rotation=90)

    for ax in [axA, axE]:
        ax.tick_params(direction='inout')
        ax.set_xlim([0, np.max(piF_l)])
        ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([0.25,'', 0.75], fontsize=8)
        ax.set_ylim([None, 0.75])
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
    
    axE.set_xticklabels([0, '', 1, '', 2, '', 3], fontsize=8)
    axA.set_xticklabels([], fontsize=8)
  
    axA.set_ylabel(r'$\sigma$', fontsize=14, rotation=0)
    axE.set_ylabel(r'$\epsilon$', fontsize=14, rotation=0)
    axE.set_xlabel(r'$\Pi_F$', fontsize=14)
    
    
    for axb in axb_l:
        axb.set_xticklabels([])
        axb.set_xticks([0, 0.5, 1])
        axb.set_yticks([0,  1])
        axb.set_yticklabels(['0',  '1'], fontsize=8)
        axb.set_ylim([0, 1])
        axb.set_xlim([0, 1])
        axb.tick_params(direction='inout')
        axb.yaxis.tick_right()
        axb.set_ylabel(r"$\beta$", rotation=0, fontsize=12)
        axb.get_yaxis().set_label_coords(1.2, 0.3)

    axb5.set_xticklabels(['0', '0.5', '1'], fontsize=8)
    axb5.set_xlabel(r'$s$', fontsize=14)

    axA.set_title('(a)', fontsize=12)
    axE.set_title('(b)', fontsize=12)
    axb1.set_title('(c)', fontsize=12)
    
    fig.subplots_adjust(wspace=0.22, hspace=0.22)
    
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/3_piRpiF_range'
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def plot_Pg50_range(params, test_p, test_n, test_nn, legend=0, cmap='viridis_r'):
    cmap = plt.cm.get_cmap(cmap)
    fig = plt.figure(figsize=(8.5, 4))

    axB = plt.subplot2grid((4, 7), (0, 0), colspan=4, rowspan=2)
    axps = plt.subplot2grid((4, 7), (2, 0), colspan=4, rowspan=2)
    axA = plt.subplot2grid((4, 7), (0, 4), colspan=2, rowspan=2)
    axE = plt.subplot2grid((4, 7), (2, 4), colspan=2, rowspan=2)
    axR = plt.subplot2grid((4, 7), (0, 6), colspan=1, rowspan=1)
    axF = plt.subplot2grid((4, 7), (1, 6), colspan=1, rowspan=1)
    axT = plt.subplot2grid((4, 7), (2, 6), colspan=1, rowspan=1)
    axS = plt.subplot2grid((4, 7), (3, 6), colspan=1, rowspan=1)
    
    for ix, pii in enumerate(test_p):
        try:
            c = (1 + ix)/(np.float(len(test_p)+1))
            color = cmap((c - 0.05)/(0.9))
            smpdf, xparams = test_m(params, -pii, test_n)
            et_i = cal_list_beta(smpdf)
            
            axB.plot(s_list, et_i, color=color, lw=2)
            axB.set_ylabel(r"$\beta(s)$", fontsize=13, rotation=0)
            axB.set_ylim([0, 1])
            axB.set_yticks([0, 0.5,  1])
            axB.set_yticklabels(['0','', '1'], fontsize=8)

            axps.plot(s_list, smpdf.p0, color=color)
            axps.set_ylabel('$p(s)$', fontsize=13, rotation=0)
            axps.set_ylim([0, 0.11])
            axps.set_yticks([0, 0.05, 0.1])
            axps.set_yticklabels(['0', '', '0.1'], fontsize=8)
            axps.set_xlabel('$s$',  fontsize=12)

            axA.plot(pii, smpdf.AA, linestyle='', marker='o', color=color)
            axA.set_xticklabels(['', ''])
            axA.set_ylim([0.1, 0.7])
            axA.set_yticks([0.3, 0.6])
            axA.set_yticklabels(['0.3', '0.6'], fontsize=8)
            axA.set_ylabel(r'$\sigma$', fontsize=13, rotation=0)

            axE.plot(pii, smpdf.epsilon, linestyle='', marker='o', color=color)
            axE.set_ylabel(r'$\epsilon$', fontsize=13, rotation=0)
            axE.set_ylim([0.35, 0.65])
            axE.set_yticks([0.4, 0.5, 0.6])
            axE.set_yticklabels(['0.4','0.5', '0.6'], fontsize=8)
            axE.set_xticklabels(['-2', '-1'], fontsize=8)
            axE.set_xlabel(test_nn, fontsize=12)

            axR.plot(pii, smpdf.pi_R, linestyle='', marker='.',  color=color)
            axR.set_ylabel(r'$\Pi_R$', fontsize=12, rotation=0)
            axR.set_ylim([0, 1])
            axR.set_yticks([0, 0.5])
            axR.set_yticklabels(['0', '0.5'], fontsize=8)

            axF.plot(pii, smpdf.pi_F, linestyle='', marker='.', color=color)
            axF.set_ylabel(r'$\Pi_F$', fontsize=12, rotation=0)
            axF.set_ylim([0, 3])
            axF.set_yticks([0, 1.5])
            axF.set_yticklabels(['0', '1.5'], fontsize=8)

            axT.plot(pii, smpdf.pi_T, linestyle='', marker='.', color=color)
            axT.set_ylabel(r'$\Pi_T$', fontsize=12, rotation=0)
            axT.set_ylim([0, 5*10**6])
            axT.set_yticks([0, 3*10**6])
            axT.set_yticklabels(['0', '3 $10^{6}$'], fontsize=8)

            axS.plot(pii, smpdf.pi_S, linestyle='', marker='.', color=color)
            axS.set_ylabel(r'$\Pi_S$', fontsize=12, rotation=0)
            axS.set_ylim([0, 549])
            axS.set_yticks([0, 250])
            axS.set_yticklabels(['0', '250'], fontsize=8)
            axS.set_xticks([2, 1])
            axS.set_xticklabels(['-2', '-1'], fontsize=8)
            axS.set_xlabel(test_nn, fontsize=12)

        except:
            print('xxx', pii)

    for ax in [axB, axps, axA, axE, axR, axF, axT, axS]:
        ax.tick_params(direction='inout')

    for ax in [axB, axps, ]:
        ax.get_yaxis().set_label_coords(-0.11, 0.4)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels([0.2, 0.4, 0.6, 0.8], fontsize=8)
        ax.set_xlim([0.2, 0.85])

    for ax in [axA, axE, ]:
        ax.get_yaxis().set_label_coords(-0.18, 0.5)
        ax.set_xticks([2, 1])
        ax.set_xlim([test_p[-1]-0.1, test_p[0]+0.1])
    
    for ax in [axR, axF, axT, axS]:
        ax.yaxis.tick_right()
        ax.get_yaxis().set_label_coords(-0.32, 0.3)
        ax.set_xticks([2, 1])
        ax.set_xlim([test_p[-1]-0.1, test_p[0]+0.1])

    for ax in [axB, axA, axR, axF, axT]:
        ax.set_xticklabels([])

    axB.set_title('(a)', fontsize=10)
    axps.set_title('(c)', fontsize=10)
    axA.set_title('(b)', fontsize=10)
    axE.set_title('(d)', fontsize=10)
    axR.set_title('(e)', fontsize=10)

    fig.subplots_adjust(wspace=0.9, hspace=0.5)
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/2_%s_range' % test_n
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def scatter_all_metrics(res_opt, res_nse, figname):
    max_AI = 2.25
    min_AI = 0.5
    lmin = 0
    fig = plt.figure(figsize=(11.5, 5.5))
    cmap = plt.cm.get_cmap('viridis')

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=1)
    ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=1, rowspan=1)
    ax6 = plt.subplot2grid((2, 4), (1, 1), colspan=1, rowspan=1)
    ax7 = plt.subplot2grid((2, 4), (1, 2), colspan=1, rowspan=1)
    ax8 = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1)

    for ax, v, subi in zip([ax1, ax2, ax3, ax5, ax6, ax7, ax8], 
                            [ ['beta_ww', r'$f_{ww}$'], ['s_star', '$s*$'], ['s_wilt', '$s_w$'], 
                            ['pi_F', r'$\Pi_F$'], ['pi_R', r'$\Pi_R$'], 
                            ['AA', r'$\sigma$'], ['epsilon', r'$\epsilon$']],
                            ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']):
        vv, vn = v
        if vv == 'pi_F':
            lmax = 2.75
            ticks = [0, 0.5, 1, 1.5, 2, 2.5]
            tickl = [0, '', 1, '', 2, '', ]
        else:
            lmax = 1
            ticks = [0, 0.25, 0.5, 0.75, 1]
            tickl = ['0', '', '', '', '1']
        ax.plot([lmin, lmax], [lmin,lmax], linestyle='--', color='silver')
        for site in res_opt['siteID']:
            x = res_nse[res_nse['siteID']==site][vv].values
            y = res_opt[res_opt['siteID']==site][vv].values
            nse = res_opt[res_opt['siteID']==site]['NSE'].values
            a = res_nse[res_nse['siteID']==site]['AI'].values
            pft = res_nse[res_nse['siteID']==site]['pft'].values[0]

            if pft == 'NL':
                    marker='^'
            elif pft == 'BL':
                    marker = 's'
            elif pft=='H':
                marker='o'
            elif pft=='M':
                marker='*'

            if a < min_AI:
                ac = 0
            elif a > max_AI:
                ac = max_AI
            else:
                ac = a
            ac = (ac - min_AI) / (max_AI - min_AI)
            if nse > 0:
                color = cmap(ac)
            else:
                color='none'
            ecolor = cmap(ac)

            cb = ax.scatter([x,], [y,], color=color, edgecolor=ecolor, marker=marker)

        reg = LinearRegression(fit_intercept=False).fit(np.array(res_nse[vv].values).reshape(-1, 1), res_opt[vv].values)
        ax.plot([lmin,lmax], [lmin, lmax*reg.coef_[0]], linestyle="-", color='k', lw=0.75)
        

        if ax == ax3:
            
            ax.plot([], [], marker='', lw=0, color='none', label='\n')
            ax.plot([], [], marker='', lw=0, color='none', label='\n')
            ax.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
            ax.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
            ax.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
            ax.plot([], [], marker='*', lw=0, color='k', label='Mixed')
            ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.2, 0.75))
        
        ax.set_xlabel(vn, fontsize=12)
        ax.set_ylabel(vn, fontsize=12)
        ax.set_ylim([lmin,lmax])
        ax.set_xlim([lmin,lmax])
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_yticklabels(tickl, fontsize=10)
        ax.set_xticklabels(tickl, fontsize=10)
        ax.tick_params(direction='inout')
        ax.set_title(subi)
        ax.get_yaxis().set_label_coords(-0.07,0.5)
        ax.get_xaxis().set_label_coords(0.5, -0.07)

        r2 = pearsonr(res_nse[vv], res_opt[vv])
        r2_ai = pearsonr(res_nse[vv], res_opt['AI'])
        biasd = cal_meanbias(res_nse[vv], res_opt[vv]) / np.mean(res_opt[vv]) * 100
        mapd = cal_mape(res_nse[vv], res_opt[vv]) * 100
        rmsd = cal_rmse(res_nse[vv], res_opt[vv])
        print( '%s  R2: %5.2f (%5.2f); RMSD:%-5.2f; mean bias:%-5.1f' % (vv, r2[0], r2[1], rmsd, biasd))

    fig.text(0.5, 0, 'Data-driven estimates', ha='center', fontsize=14)
    fig.text(0.06, 0.5, 'Optimality-based estimates', va='center', rotation='vertical', fontsize=14)

    #cbar = plt.colorbar(cb)
    #cbar.ax.set_ylabel('Aridity')

    fig.subplots_adjust(wspace=0.33, hspace=0.44)

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/4_results_%s' % figname
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def get_sm_gof(res_opt, res_dd):
    all_nse_dd = []
    all_nse_opt = []
    all_smean_opt = []
    all_smean_dd = []
    all_smean_obs = []

    model_pnames = ['T_GS', 'rf_alpha', 'rf_lambda', 'Eo', 'Td', 'LAI', 'RAI',
                    'hc', 'Zmi', 'Ps0', 'b', 'Ks', 'n', 's_h', 's_fc', 'k_xl_max', 'Px50', 'Pg50']
    for si, site in enumerate (res_opt['siteID'].values):
        params_opt = res_opt[res_opt['siteID']==site]
        params_dd = res_dd[res_dd['siteID']==site]
        with open('../../DATA/WUS/WUS_output/opt/files/_1/%s.pickle' %site, 'rb') as f:
            loc_res = load(f)
        s_obs = loc_res['s_obs']
        nbins=100
        s_list = np.linspace(0, 1, nbins + 1)

        theta_opt = [params_opt[k].values[0] for k in model_pnames]

        smpdf = SM_C_H(theta_opt, nbins=nbins)
        p0_opt = smpdf.p0*nbins
        
        cdf_opt = np.cumsum(p0_opt/np.sum(p0_opt))
        cdf_opt_n = cdf_opt / np.max(cdf_opt)
        s_opt = [(np.abs(cdf_opt_n - qi / 100.)).argmin() / np.float(nbins) for qi in range(1, 101)]
        obs_l = [np.percentile(s_obs, qi / 100. * 100) for qi in range(1, 101)]
        nse_opt = cal_nse(obs_l, s_opt)
        f = interp1d(cdf_opt, np.linspace(0, 1, nbins+1))
        random_p = [np.random.uniform(0, 1) for r in range(10000)]
        fit_s_opt = np.array(f(random_p))
        
        theta_dd = [params_dd[k].values[0] for k in model_pnames]

        smpdf = SM_C_H(theta_dd, nbins=nbins)
        p0_dd = smpdf.p0*nbins

        cdf_dd = np.cumsum(p0_dd/np.sum(p0_dd))
        cdf_dd_n = cdf_dd / np.max(cdf_dd)
        s_dd = [(np.abs(cdf_dd_n - qi / 100.)).argmin() / np.float(nbins) for qi in range(1, 101)]
        obs_l = [np.percentile(s_obs, qi / 100. * 100) for qi in range(1, 101)]
        nse_dd = cal_nse(obs_l, s_dd)
        f = interp1d(cdf_dd, np.linspace(0, 1, nbins+1))
        random_p = [np.random.uniform(0, 1) for r in range(10000)]
        fit_s_dd = np.array(f(random_p))
        
        if (np.isnan(np.mean(fit_s_opt))==0) and nse_opt>0:
            all_nse_opt.append(nse_opt)
            all_nse_dd.append(nse_dd)
        else:
            all_nse_opt.append(0)
            all_nse_dd.append(nse_dd)

        if (np.isnan(np.mean(fit_s_opt))==0):
            all_smean_opt.append(np.mean(fit_s_opt))
            all_smean_dd.append(np.mean(fit_s_dd))
            all_smean_obs.append(np.mean(s_obs))
            

    return all_nse_dd, all_nse_opt, all_smean_opt, all_smean_dd, all_smean_obs


def plot_sm_gof(res_opt, res_dd, figname='sm_gof'):
    all_nse_dd, all_nse_opt, all_smean_opt, all_smean_dd, all_smean_obs = get_sm_gof(res_opt, res_dd)
    fig = plt.figure(figsize=(3, 6))
    ax_h = fig.add_subplot(2, 1, 1)
    ax_s = fig.add_subplot(2, 1, 2)

    r = res_opt
    cmap = plt.cm.get_cmap('viridis')

    ax_h.hist(all_nse_dd, bins=np.linspace(0, 1, 6), color=cmap(1), alpha=1, label='data\ndriven')
    ax_h.hist(all_nse_opt, bins=np.linspace(0, 1, 6), color=cmap(0.5), alpha=0.75, label='optimality\nbased')
    #ax_h.set_yticks([0, 5, 10, 15])
    ax_h.set_xticklabels(['0', '0.25', '0.50', '0.75', '1'])
    ax_h.set_xticks([0, 0.25, 0.50, 0.75, 1])
    ax_h.set_xlabel('NSE between empirical \n and theoretical $p(s)$', fontsize=12)
    ax_h.set_ylabel('Number of sites', fontsize=12)
    ax_h.set_xlim([0,1])
    ax_h.set_title('(a)')
    ax_h.legend(frameon=False)

    ax_s.scatter(all_smean_dd, all_smean_obs, color=cmap(1), alpha=1)
    rho_dd = pearsonr(all_smean_dd, all_smean_obs)[0]
    rmse_dd = cal_rmse(all_smean_obs, all_smean_dd)
    mape_dd = cal_mape(all_smean_obs, all_smean_dd) * 100
    rmse_dd = cal_rmse(all_smean_obs, all_smean_dd)
    bias_dd = cal_meanbias(all_smean_obs, all_smean_dd) / np.mean(all_smean_obs) * 100
    

    ax_s.scatter(all_smean_opt, all_smean_obs, color=cmap(0.5), alpha=0.75)
    rho_opt = pearsonr(all_smean_opt, all_smean_obs)[0]
    rmse_opt = cal_rmse(all_smean_obs, all_smean_opt)
    mape_opt = cal_mape(all_smean_obs, all_smean_opt) * 100
    bias_opt = cal_meanbias(all_smean_obs, all_smean_opt) / np.mean(all_smean_obs) * 100

    reg_dd = LinearRegression(fit_intercept=False).fit(np.array(all_smean_dd).reshape(-1, 1), all_smean_obs)
    reg_opt = LinearRegression(fit_intercept=False).fit(np.array(all_smean_opt).reshape(-1, 1), all_smean_obs)
    

    ax_s.plot([0, 1], [0, 1*reg_dd.coef_[0]], linestyle="-", color=cmap(1), label='r=%-5.2f' %rho_dd, lw=0.75)
    ax_s.plot([0, 1], [0, 1*reg_opt.coef_[0]], linestyle="-", color=cmap(0.5), label='r=%-5.2f' %rho_opt, lw=0.75)

    ax_s.plot([0,1], [0,1], linestyle="--", color='silver')
    ax_s.set_xlim([0.1, 0.85])
    ax_s.set_ylim([0.1, 0.85])
    ax_s.set_xlabel(r'$\langle s \rangle$ (modeled)', fontsize=12)
    ax_s.set_ylabel(r'$\langle s \rangle$ (observed)', fontsize=12)
    ax_s.set_title('(b)')

    plt.tight_layout()
    print('p(s) NSE data-driven: %-5.2f; p(s) NSE  optimality-based: %-5.2f' %(np.median(all_nse_dd), np.median(all_nse_opt)))
    print('rho <s> data-driven: %-5.2f; rho <s>  optimality-based: %-5.2f' %(rho_dd, rho_opt))
    print('rmse <s> data-driven: %-5.2f; rmse <s>  optimality-based: %-5.2f' %(rmse_dd, rmse_opt))
    print('mean bias <s> data-driven: %-5.1f %%; bias <s>  optimality-based: %-5.1f %%' % (bias_dd, bias_opt))

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/%s' % figname
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def flux_scatter(res_opt, res_dd, figname='flux_scatter'):

    for k in ['GPP_mean', 'EF']:
        k_data = []
        for site in res_opt['siteID']:
            with open('../../DATA/WUS/WUS_input_data_2/%s_params.pickle' % site, 'rb') as f:
                p_s = load(f)
            k_data.append(p_s[k])
        res_opt[k] = k_data
        res_dd[k] = k_data

    lmin_ET = 0
    lmax_ET = 4.3
    lmax_sT = 3.3
    lmin_GPP = 0
    lmax_GPP = 13.5
    vmin = 0.5
    vmax = 2.25

    fig = plt.figure(figsize=(13, 3.5))
    ax_GPP = fig.add_subplot(1, 3, 1)
    ax_ET = fig.add_subplot(1, 3, 2)

    cmap = plt.cm.get_cmap('viridis')
    for r, colori, name in [[res_dd, 1, 'data-driven'], [res_opt, 0.5, 'optimality-based']]:
        r['GPP'] = r['GPP_mean'] * 12.0107 * 24 * 3600
        r['Delta'] = r['LAI'] * 2 * 10 ** - 4
        r['ET_obs'] = r['EF'] * r['RF'] * 1000
        r['I_mod'] = r['rf_alpha'] * r['rf_lambda'] *(1- np.exp(-r['Delta'] / r['rf_alpha'])) * 1000
        r['ET_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] * 1000 + r['I_mod']
        r['T_mod'] = r['mean_tf'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] * 1000
        r['(1-s) * T'] = r['epsilon'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda'] *1000 

        for site in r['siteID']:
            r_site = r[r['siteID']==site]
            pft = r_site['pft'].values[0]
            if pft == 'NL':
                marker='^'
            elif pft == 'BL':
                    marker = 's'
            elif pft=='H':
                marker='o'
            elif pft=='M':
                marker='*'
            c = [r_site['AI'].values,]

            cb = ax_ET.scatter([r_site['ET_mod'].values,], [r_site['ET_obs'].values ,],  
                               color = cmap(colori), marker=marker, vmin=vmin, vmax=vmax, alpha=0.75)
            ax_GPP.scatter([r_site['(1-s) * T'].values,], [r_site['GPP'].values ,],  
                           color = cmap(colori), marker=marker, vmin=vmin, vmax=vmax, alpha=0.75)
        reg = LinearRegression(fit_intercept=False).fit(np.array(r['(1-s) * T'].values).reshape(-1, 1), r['GPP'].values)
        rho_GPP = pearsonr(r['(1-s) * T'], r['GPP'] )[0]

        ax_GPP.plot([0, lmax_sT], [0, lmax_sT*reg.coef_[0]], linestyle="-", color=cmap(colori), lw=0.75)
        ax_GPP.set_ylim([lmin_GPP, lmax_GPP])
        ax_GPP.set_xlim([lmin_ET, lmax_sT])

        rho_GPP = pearsonr(r['GPP'], r['(1-s) * T'] )[0]

        ax_GPP.set_xlabel(r'$(1 — \langle\Theta\rangle) \langle T \rangle$ (modeled)', fontsize=12)
        ax_GPP.set_ylabel(r'$\langle GPP \rangle$ (observed)', fontsize=12)
        ax_GPP.set_yticks([0, 3, 6, 9, 12])
        ax_GPP.set_xticks([0, 1, 2, 3])
        ax_GPP.set_title('(a)')
        print(name, ': GPP rho = %-5.2f' % rho_GPP)

        
        reg = LinearRegression(fit_intercept=False).fit(np.array(r['ET_mod'].values).reshape(-1, 1), r['ET_obs'].values)
        rho_ET = pearsonr(r['ET_obs'], r['ET_mod'] )[0]
        rmse_ET = cal_rmse(r['ET_obs'], r['ET_mod'])
        mape_ET = cal_mape(r['ET_obs'], r['ET_mod']) * 100

        bias_ET = cal_meanbias(r['ET_obs'], r['ET_mod']) / np.mean(r['ET_obs']) * 100


        ax_ET.plot([0, lmax_ET], [0, lmax_ET*reg.coef_[0]], linestyle="-", color=cmap(colori), lw=0.75)
        ax_ET.plot([0, lmax_ET], [0, lmax_ET], linestyle="--", color='silver')
        ax_ET.set_ylim([lmin_ET, lmax_ET])
        ax_ET.set_xlim([lmin_ET, lmax_ET])

        ax_ET.set_xlabel(r'$\langle ET \rangle$ (modeled)', fontsize=12)
        ax_ET.set_ylabel(r'$\langle ET \rangle$ (observed)', fontsize=12)
        ax_ET.set_yticks([0, 1, 2, 3, 4])
        ax_ET.set_xticks([0, 1, 2, 3, 4])
        ax_ET.set_title('(b)')
        print(name, ': ET rho = %-5.2f' % rho_ET)
        print(name, ': ET rmse = %-5.2f' % rmse_ET)
        print(name, ': ET bias = %-5.1f %%' % bias_ET)

    ax_ET.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
    ax_ET.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
    ax_ET.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
    ax_ET.plot([], [], marker='*', lw=0, color='k', label='Mixed')
    ax_ET.plot([], [], marker='', lw=0, color='none', label='')
    ax_ET.plot([], [], marker='', lw=0, color='none', label='')
    ax_ET.plot([], [], marker='', lw=10, color=cmap(1), label='data-driven')
    ax_ET.plot([], [], marker='', lw=10, color=cmap(0.5), label='optimality-based')
    ax_ET.legend(frameon=False, loc='center left', bbox_to_anchor=(1.2, 0.75))


    plt.subplots_adjust(wspace=0.45)
    #plt.tight_layout()
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/%s' % figname
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def wb_aridity(res_opt, res_dd, figname='budyko_f'):
    vmin = 0
    vmax = 2
    cmap = plt.cm.get_cmap('viridis')

    fig = plt.figure(figsize=(5, 7))
    ax_B = fig.add_subplot(2, 1, 1)
    ax_tB = fig.add_subplot(2, 1, 2)

    r = res_dd

    r['Delta'] = r['LAI'] * 2 * 10 ** - 4
    r['I_mod'] = r['rf_alpha'] * r['rf_lambda'] *(1- np.exp(-r['Delta'] / r['rf_alpha'])) 
    r['ET_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] + r['I_mod']
    r['T_mod'] = r['mean_tf'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda']
    r['E_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda']  - r['T_mod']
    r['sT_mod'] = r['epsilon'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda'] 

    for site in r['siteID']:
        r_site = r[r['siteID']==site]
        pft = r_site['pft'].values[0]
        if pft == 'NL':
            marker='^'
        elif pft == 'BL':
                marker = 's'
        elif pft=='H':
            marker='o'
        elif pft=='M':
            marker='*'
        c = [r_site['pi_F'].values,]
        
        for nn, axn in zip(['ET_mod',  'T_mod', ], [ax_B,  ax_tB, ]):
            cb = axn.scatter([r_site['AI'].values,], 
                             [r_site[nn].values/r_site['rf_alpha'].values/r_site['rf_lambda'].values ,], 
                             edgecolor=cmap(r_site['pi_F'].values[0]/2), facecolor='none', marker=marker)
            
    r = res_opt

    r['Delta'] = r['LAI'] * 2 * 10 ** - 4
    r['I_mod'] = r['rf_alpha'] * r['rf_lambda'] *(1- np.exp(-r['Delta'] / r['rf_alpha'])) 
    r['ET_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] + r['I_mod']
    r['T_mod'] = r['mean_tf'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda']
    r['E_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda']  - r['T_mod']
    r['sT_mod'] = r['epsilon'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda'] 

    for site in r['siteID']:
        r_site = r[r['siteID']==site]
        pft = r_site['pft'].values[0]
        if pft == 'NL':
            marker='^'
        elif pft == 'BL':
                marker = 's'
        elif pft=='H':
            marker='o'
        elif pft=='M':
            marker='*'
        c = [r_site['pi_F'].values,]
        
        for nn, axn in zip(['ET_mod',  'T_mod'], [ax_B,  ax_tB]):
            cb = axn.scatter([r_site['AI'].values,], 
                             [r_site[nn].values / r_site['rf_alpha'].values / r_site['rf_lambda'].values ,], 
                             color=cmap(r_site['pi_F'].values[0]/2), marker=marker)

    d = np.linspace(0.2, 3.5, 101)
    ax_B.plot(d, ((1 - np.exp(-d)) * d * np.tanh(1/d))**0.5, color='silver', lw=0.5, linestyle='-', marker='')

    for ip, nn, axn in zip(['(a)', '(b)'], [r'$\langle ET \rangle / \langle P \rangle$ (modeled)', r'$\langle T \rangle / \langle P \rangle$ (modeled)'], [ax_B,  ax_tB]):
        axn.set_ylim([0.35, 1.02])
        axn.set_xlim([0, 3.5])
        axn.set_yticks([ 0.4, 0.6, 0.8, 1])
        axn.set_xticks([0, 1, 2, 3])
        axn.set_title(ip)
        
        axn.set_ylabel(nn, fontsize=12)
    ax_tB.set_xlabel(r'$\langle E_0 \rangle / \langle P \rangle$', fontsize=12)

    ax_B.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
    ax_B.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
    ax_B.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
    ax_B.plot([], [], marker='*', lw=0, color='k', label='Mixed')
    ax_B.legend(frameon=False, loc='lower right', fontsize=10)

    #plt.colorbar(cb, ticks=[0, 0.25, 0.5, 0.75, 1], label=r'$\Pi_F$')
    #cb.ax_tB.set_yticklabels([0, 5, 1, 1.5, '>2'])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
    cb_ = fig.colorbar(cb, cax=cbar_ax)
    cb_.set_label(label = r'$\Pi_F$', fontsize=16, rotation=0)

    cb_.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb_.set_ticklabels(['0','0.5', '1', '1.5', '>2'])

    #plt.tight_layout()
   
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/%s' % figname
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def wb_aridity_sigma(res_opt, res_dd, figname='budyko_s'):
    vmin = 0
    vmax = 2
    cmap = plt.cm.get_cmap('viridis_r')

    fig = plt.figure(figsize=(5, 7))
    ax_B = fig.add_subplot(2, 1, 1)
    ax_tB = fig.add_subplot(2, 1, 2)

    r = res_dd

    r['Delta'] = r['LAI'] * 2 * 10 ** - 4
    r['I_mod'] = r['rf_alpha'] * r['rf_lambda'] *(1- np.exp(-r['Delta'] / r['rf_alpha'])) 
    r['ET_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] + r['I_mod']
    r['T_mod'] = r['mean_tf'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda']
    r['E_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda'] - r['T_mod']
    r['sT_mod'] = r['epsilon'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda'] 

    for site in r['siteID']:
        r_site = r[r['siteID']==site]
        pft = r_site['pft'].values[0]
        if pft == 'NL':
            marker='^'
        elif pft == 'BL':
                marker = 's'
        elif pft=='H':
            marker='o'
        elif pft=='M':
            marker='*'
        c = (r_site['AA'].values[0] - 0.25) / (1-0.25)
        for nn, axn in zip(['ET_mod',  'T_mod', ], [ax_B,  ax_tB, ]):
            cb = axn.scatter([r_site['AI'].values,], 
                             [r_site[nn].values/r_site['rf_alpha'].values/r_site['rf_lambda'].values ,], 
                             edgecolor=cmap(c), facecolor='none', marker=marker)
            
    r = res_opt

    r['Delta'] = r['LAI'] * 2 * 10 ** - 4
    r['I_mod'] = r['rf_alpha'] * r['rf_lambda'] *(1- np.exp(-r['Delta'] / r['rf_alpha'])) 
    r['ET_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] + r['I_mod']
    r['T_mod'] = r['mean_tf'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda']
    r['E_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda']  - r['T_mod']
    r['sT_mod'] = r['epsilon'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda'] 

    for site in r['siteID']:
        r_site = r[r['siteID']==site]
        pft = r_site['pft'].values[0]
        if pft == 'NL':
            marker='^'
        elif pft == 'BL':
                marker = 's'
        elif pft=='H':
            marker='o'
        elif pft=='M':
            marker='*'
        c = (r_site['AA'].values[0] - 0.25) / (1-0.25)
        for nn, axn in zip(['ET_mod',  'T_mod'], [ax_B,  ax_tB]):
            cb = axn.scatter([r_site['AI'].values,], 
                             [r_site[nn].values / r_site['rf_alpha'].values / r_site['rf_lambda'].values ,], 
                             color=cmap(c), marker=marker)

    d = np.linspace(0.2, 3.5, 101)
    ax_B.plot(d, ((1 - np.exp(-d)) * d * np.tanh(1/d))**0.5, color='silver', lw=0.5, linestyle='-', marker='')

    for ip, nn, axn in zip(['(a)', '(b)'], [r'$\langle ET \rangle / \langle P \rangle$ (modeled)', r'$\langle T \rangle / \langle P \rangle$ (modeled)'], [ax_B,  ax_tB]):
        axn.set_ylim([None, None])
        axn.set_xlim([0, 3.5])
        axn.set_yticks([ 0.4, 0.6, 0.8, 1])
        axn.set_xticks([0, 1, 2, 3])
        axn.set_title(ip)
        
        axn.set_ylabel(nn, fontsize=12)
    ax_tB.set_xlabel(r'$\langle E_0 \rangle / \langle P \rangle$', fontsize=12)

    ax_B.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
    ax_B.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
    ax_B.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
    ax_B.plot([], [], marker='*', lw=0, color='k', label='Mixed')
    ax_B.legend(frameon=False, loc='lower right', fontsize=10)

    #plt.colorbar(cb, ticks=[0.25, 0.5, 0.75], label=r'$\Pi_F$')
    #cb.ax_tB.set_yticklabels([0, 5, 1, 1.5, '>2'])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
    cb_ = fig.colorbar(cb, cax=cbar_ax)
    cb_.set_label(label = r'$\sigma$', fontsize=20, rotation=0)

    cb_.set_ticks([0, 0.33, 0.66,  1])
    cb_.set_ticklabels([1, 0.75, 0.5, 0.25])

    #plt.tight_layout()
   
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/%s' % figname
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def et_part(r,):
    vmin = 0.5
    vmax = 2.25
    r['Delta'] = r['LAI'] * 2 * 10 ** - 4
    r['I_mod'] = r['rf_alpha'] * r['rf_lambda'] *(1- np.exp(-r['Delta'] / r['rf_alpha'])) 
    r['ET_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda'] + r['I_mod']
    r['T_mod'] = r['mean_tf'] * (r['rf_alpha'] - r['Delta']) * r['rf_lambda']
    r['E_mod'] = r['mean_ef'] * (r['rf_alpha'] - r['Delta'] ) * r['rf_lambda']  - r['T_mod']
    
    fig = plt.figure(figsize=(5.5, 12))

    for isp, (tname, vname, ylim, legend) in enumerate([['(a)', 'E_mod', [0, 0.6], True], 
                                                 ['(b)', 'T_mod', [0.4, 1], False], 
                                                 ['(c)', 'I_mod', [0, 0.15], False]]):
        ax = fig.add_subplot(3, 1, 1+isp)
        for site in r['siteID']:
            x = r[r['siteID']==site]['LAI'].values
            #y = r[r['siteID']==site]['tet_part'].values
            y = r[r['siteID']==site][vname].values / r[r['siteID']==site]['ET_mod'].values
            c = r[r['siteID']==site]['AI'].values
            pft = r[r['siteID']==site]['pft'].values[0]
           
            if pft == 'NL':
                marker='^'
            elif pft == 'BL':
                    marker = 's'
            elif pft=='H':
                marker='o'
            elif pft=='M':
                marker='*'

            cb = ax.scatter([x,], [y,], c=[c,], vmin=vmin, vmax=vmax, cmap='viridis', marker=marker)
        ax.set_ylim(ylim)
        if legend:
            ax.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
            ax.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
            ax.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
            ax.plot([], [], marker='*', lw=0, color='k', label='Mixed')
            ax.legend(frameon=False, loc='upper right', fontsize=11)
        if isp == 2:
            ax.set_xlabel('$LAI$', fontsize=14)
            ax.set_yticks([0, 0.05, 0.1, 0.15])
        ax.set_ylabel(r'$\langle %s \rangle / \langle ET \rangle$' % vname[0], fontsize=14)
        cbar = plt.colorbar(cb, ticks=[1,  2])
        cbar.ax.set_ylabel(r'$\langle E_0 \rangle / \langle P \rangle$', fontsize=14)
        ax.set_title(tname)

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_ET_partitioning'
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()


def piF_piR_scatter(res_nse, res_opt):
    from scipy import stats
    cmap = plt.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    for r, color in zip([res_nse, res_opt],  [cmap(1), cmap(0.5)]):
        for site in res_opt['siteID']:
            x = r[r['siteID']==site]['pi_F'].values
            y = r[r['siteID']==site]['pi_R'].values
            pft = r[r['siteID']==site]['pft'].values[0]
            nse = r[r['siteID']==site]['NSE'].values[0]
            if pft == 'NL':
                marker='^'
            elif pft == 'BL':
                    marker = 's'
            elif pft=='H':
                marker='o'
            elif pft=='M':
                marker='*'
               
            if (color == cmap(0.5)) and (nse<0):
                c='none'
                ec = color
            else:
                c = color
                ec = color
            cb = ax.scatter([x,], [y,], edgecolor = ec, color=c, marker=marker)
        
        slope, intercept, r_value, \
                p_value, std_err = stats.linregress(r['pi_F'], r['pi_R'])
        print('%-5.2f (%-5.2f)'% (slope, pearsonr(r['pi_F'], r['pi_R'])[1])) 
        ax.plot([0, np.max(r['pi_F']) * 1.1], [intercept, intercept + np.max(r['pi_F']) * 1.1 * slope], color= color, linestyle="--") 
    
    ax.set_xlabel(r'$\Pi_F$', fontsize=12)
    ax.set_ylabel(r'$\Pi_R$', fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim([0, 1.05])
    ax.set_yticklabels(['0', 0.25, '0.5', 0.75, 1], fontsize=9)
    ax.set_xticklabels(['0', 1, 2, 3, 4], fontsize=9)
    ax.tick_params(direction='inout')

    ax.plot([], [], marker='o', lw=0, color=cmap(1), label='Data-driven')
    ax.plot([], [], marker='o', lw=0, color=cmap(0.5), label='Optimality-based')
    ax.legend(frameon=False, loc='upper right', fontsize=9)

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_piRpiF'
    #plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.pdf' % fig_name)
    plt.show()
