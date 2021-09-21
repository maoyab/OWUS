import sys 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sswm import SM_C_H

from scipy.stats import spearmanr, pearsonr


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
    ape = [(mi - oi)**2 for mi, oi in zip(mod, obs)]
    return (np.nanmean(ape))**0.5


def cal_meanpctbias(obs, mod):
    b = [(oi - mi) / mi for mi, oi in zip(mod, obs)]
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
            axB.set_title('A', fontsize=16)

            axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
            axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
            axA.set_ylim([0, 1])
            axA.set_yticks([0, 0.5])
            axA.set_yticklabels(['0', '0.5'])
            axA.set_title('B', fontsize=16)

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
    plt.savefig('%s.eps' % fig_name)
    plt.show()


def plot_SI_pi_range(params, test_p, test_n, test_nn, pi_vary, cmap='viridis_r'):
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
            axB.set_xticklabels(['1', '0.1'])
            axB.set_xlabel('Soil water potential [MPa]', fontsize=14)
            axB.set_ylabel(r"$\beta(s) = T/E_0$ [-]", fontsize=14)
            axB.set_title('A', fontsize=16)

            axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
            axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
            axA.set_ylim([0, 1])
            axA.set_yticks([0, 0.5])
            axA.set_yticklabels(['0', '0.5'])
            axA.set_title('B', fontsize=16)

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
            axS.set_xticks([test_p[0], test_p[-1]])
            axS.set_xlabel(test_nn, fontsize=14)
            axS.yaxis.set_label_position('right')
        except:
            pass
    
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_%s_range' % pi_vary
    plt.savefig('%s.eps' % fig_name)
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
        axB.set_xlim([0,1])
        axB.set_xlabel('Soil saturation', fontsize=14)
        axB.set_ylabel(r"$\beta(s) = T/E_0$", fontsize=14)
        axB.set_title('A', fontsize=16)

        axA.plot(pii, smpdf.AA, linestyle='', marker='.', color=color)
        axA.set_ylabel(r'$\sigma$', fontsize=16, rotation=0)
        axA.set_ylim([0, 1])
        axA.set_yticks([0, 0.5])
        axA.set_yticklabels(['0', '0.5'])
        axA.set_title('B', fontsize=16)

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
    plt.savefig('%s.eps' % fig_name)
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

    axA.set_title('A', fontsize=12)
    axE.set_title('B', fontsize=12)
    axb1.set_title('C', fontsize=12)
    
    fig.subplots_adjust(wspace=0.22, hspace=0.22)
    
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/3_piRpiF_range'
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.svg' % fig_name)
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
            axps.set_yticks([0, 0.05,  0.1])
            axps.set_yticklabels(['0', '', '0.1'], fontsize=8)
            axps.set_xlabel('Soil saturation ($s$)',  fontsize=12)

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

    axB.set_title('A', fontsize=10)
    axps.set_title('C', fontsize=10)
    axA.set_title('B', fontsize=10)
    axE.set_title('D', fontsize=10)
    axR.set_title('E', fontsize=10)

    fig.subplots_adjust(wspace=0.9, hspace=0.5)
    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/2_%s_range' % test_n
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)
    plt.show()


def scatter_all_metrics(res_opt, res_nse, figname):
    print(len(res_nse.index), len(res_opt[res_opt['NSE']<0].index), len(res_nse[res_nse['NSE']<0].index),
        'NSE data-driven: %-5.2f; NSE optimality-based: %-5.2f' %(np.median(res_nse['NSE']), np.median(res_opt['NSE'])))
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
                            ['A', 'B', 'C', 'D', 'E', 'F', 'G']):
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
        bias = cal_meanpctbias(res_nse[vv], res_opt[vv])
        mapd = cal_mape(res_nse[vv], res_opt[vv])
        rmsd = cal_rmse(res_nse[vv], res_opt[vv])
        print( '%s  R2: %5.2f (%5.2f); RMSD:%-5.2f' % (vv, r2[0], r2[1], rmsd))

    fig.text(0.5, 0, 'Data-driven estimates', ha='center', fontsize=14)
    fig.text(0.06, 0.5, 'Optimality-based estimates', va='center', rotation='vertical', fontsize=14)

    #cbar = plt.colorbar(cb)
    #cbar.ax.set_ylabel('Aridity')

    fig.subplots_adjust(wspace=0.33, hspace=0.44)

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/4_results_%s' % figname
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)
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
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.svg' % fig_name)
    plt.show()


def budyko(r, vv, vn, vlim=None):
    def __th_B(d):
        a = (1 - np.exp(-d)) * d * np.tanh(1/d)
        return a **0.5

    if vlim is None:
        vmin = np.percentile(r[vv], 5)
        vmax = np.percentile(r[vv], 95)
    else:
        [vmin, vmax] = vlim
    fig = plt.figure(figsize=(5.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    for site in r['siteID']:
        x = r[r['siteID']==site]['mean_di'].values
        delta =  2 * 10 ** - 4 * r[r['siteID']==site]['LAI'].values
        y = r[r['siteID']==site]['mean_ef'].values
        c = r[r['siteID']==site][vv].values
        pft = r[r['siteID']==site]['pft'].values[0]
        if pft == 'NL':
                marker='^'
        elif pft == 'BL':
                marker = 's'
        elif pft=='H':
            marker='o'
        elif pft=='M':
            marker='*'
        cb = ax.scatter([x,], [y,], c=[c,], vmin=vmin, vmax=vmax, cmap='viridis_r', marker=marker)
    max_di=3
    min_di=0.25
    ax.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
    ax.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
    ax.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
    ax.plot([], [], marker='*', lw=0, color='k', label='Mixed')
    ax.plot([min_di, 1], [min_di, 1], linestyle='-', color='silver')
    ax.plot([1, max_di], [1, 1], linestyle='-', color='silver')
    di = np.linspace(min_di, max_di)
    ef = [__th_B(diii) for diii in di]
    ax.plot(di, ef, linestyle='--', color='silver')
    ax.set_xlabel('$<E_0>$/$<P>$', fontsize=14)
    ax.set_ylabel('$<E+T>$/$<P>$', fontsize=14)
    #ax.set_xscale('log')
    cbar = plt.colorbar(cb, ticks=[0.25, 0.5, 0.75])
    cbar.ax.set_ylabel(vn, fontsize=14, rotation=0)
    #ax.legend(frameon=False)

    plt.show()


def tet_part(r):
    vmin = 0.5
    vmax = 2.25
    fig = plt.figure(figsize=(5.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    for site in r['siteID']:
        x = r[r['siteID']==site]['LAI'].values
        y = r[r['siteID']==site]['tet_part'].values
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
    
    ax.plot([], [], marker='o', lw=0, color='k', label='Herbaceous')
    ax.plot([], [], marker='s', lw=0, color='k', label='Broadleaf')
    ax.plot([], [], marker='^', lw=0, color='k', label='Needleleaf')
    ax.plot([], [], marker='*', lw=0, color='k', label='Mixed')
    ax.legend(frameon=False, loc='lower right', fontsize=11)
    ax.set_xlabel('$LAI$', fontsize=14)
    ax.set_ylabel(r'$\langle T \rangle / \langle E+T \rangle$', fontsize=14)
    cbar = plt.colorbar(cb, ticks=[1,  2])
    cbar.ax.set_ylabel(r'$\langle E_0 \rangle / \langle P \rangle$', fontsize=14)

    fig_name = '../../PROJECTS/optimal_water_use_strategies/Figures/SI_TET'
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)
    plt.show()
