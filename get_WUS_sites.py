import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pickle import dump, load
from sswm import SM_C_H
from param_sm_pdf import Processor


def go_mhmcmc(dfi, unknown_params, save_params, epsi=None, stress_type='dynamic', num_pl=3, nbins=100):
    
    mhmcmc = Processor(unknown_params, save_params, nbins=nbins, epsi=epsi, stress_type=stress_type)
    
    dt0 = datetime.now()
    print(dt0)
    model_param_names = ['T_GS', 'rf_alpha', 'rf_lambda', 'Eo', 'Td',
                          'LAI', 'RAI', 'hc', 'Zm',
                          'Ps0', 'b', 'Ks', 'n', 's_h', 's_fc', 
                          'k_xl_max', 'Px50', 'Pg50']

    theta0 = {}
    for p in model_param_names:
        theta0[p] = dfi[p]
    pl_results, n_it, fail_conv_count, fail_eff_count = mhmcmc.get_mcmc_mh_results(
                                                                dfi['s_obs'], 
                                                                theta0, p_ranges, 
                                                                num_pl=num_pl)
    dfi['n_it'] = n_it
    dfi['num_pl'] = len(pl_results)
    dfi['fail_conv_count'] = fail_conv_count
    dfi['fail_eff_count'] = fail_eff_count

    if dfi['num_pl'] == num_pl:
        dfi = mhmcmc.process_raw_results(dfi, pl_results, p_ranges, outfile_format='full')

    dfi['ctime'] = (datetime.now() - dt0).seconds / 60.
    dfi['fail_conv_count'] = fail_conv_count + dfi['fail_conv_count']
    dfi['fail_eff_count'] = fail_eff_count + dfi['fail_eff_count']
    print('ctime', dfi['ctime'])
    return dfi


def plot_ps(params, nbins=30, out_fig_file=None , title=None, stress_type='dynamic'):
    s_list = np.linspace(0, 1, nbins + 1)

    theta = [params['T_GS'], params['rf_alpha'], params['rf_lambda'], params['Eo'], params['Td'],
                                      params['LAI'], params['RAI'], params['hc'], params['Zmi'],
                                      params['Ps0'], params['b'], params['Ks'], params['n'], params['s_h'], params['s_fc'], 
                                      params['k_xl_max'], params['Px50'], params['Pg50']]

    smpdf = SM_C_H(theta, nbins=nbins, stress_type=stress_type)
    p0 = smpdf.p0*nbins
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(params['s_obs'], density=True, color='silver', bins=s_list)
    ax.axvline(smpdf.s_wilt,  color='dodgerblue', linestyle=':', label='$S_w$')
    ax.axvline(smpdf.s_star, color='dodgerblue', linestyle='-', label='$S^*$')
    ax.axvline(smpdf.s_h,  color='k', linestyle=':', label='$S_h$')
    ax.axvline(smpdf.s_fc, color='k', linestyle='-', label='$S_{fc}$')
    ax.plot(s_list, p0, color='tomato', label='best-fit')
    plt.legend()
    plt.suptitle(title, fontsize=18)    
    if out_fig_file is None:
        plt.show()
    else:
        plt.savefig(out_fig_file)
    plt.close()


def plot_mcmc_results(loc_res, plot_params, burnin=0, out_fig_file=None , title=None, epsi=None):
    colors = ['gold', 'limegreen', 'dodgerblue', 'purple', 'navy']
    lp = len(plot_params)
    fig = plt.figure(figsize=(3*lp, 9))
    for i, p in enumerate(plot_params):
        if p == 'k_xl_max':
            scale = 1000
        else:
            scale = 1
        
        try:
            xlim = [p_ranges[p][0] * scale, p_ranges[p][1] * scale] 
        except:
            xlim = [None, None] 

        ax = fig.add_subplot(3, lp, i + 1 )
        estimates = np.array(loc_res['%s_estimates' % p]) * scale
        loglikelihood = loc_res['loglikelihood_estimates']
        estimates = [e[int(burnin*len(e)):] for e in estimates]
        loglikelihood = [l[int(burnin*len(l)):] for l in loglikelihood]
        for e ,c in zip(estimates, colors):
            ax.hist(e, bins=10, alpha=0.75, color=c)
            ax.axvline(np.median(e), linestyle=':', color=c)
        ax.axvline(np.median(estimates),  color='k')
        ax.set_xlim(xlim)
        ax.set_xticks([])
        if i == 0:
            ax.set_ylabel('n random samples')
        else: 
            ax.set_yticks([])
        
        ax = fig.add_subplot(3, lp, i + 1 + lp)
        for e, ll, c in zip(estimates, loglikelihood, colors):
            ax.scatter(e, ll, alpha=0.75, marker='.', s=1, color=c)
            ax.axvline(np.median(e), linestyle=':', color=c)
        ax.axvline(np.median(estimates), color='k')
        ax.axvline(loc_res['%s_maxmaxlike' % p] * scale, color='tomato', lw=2)
        ax.axvline(loc_res[p] * scale, color='orange', lw=2)

        ax.set_xlim(xlim)
        ax.set_xticks([])
        if i == 0:
            if epsi is None:
                ax.set_ylabel('loglikelihood')
            else:
                ax.set_ylabel(r'$/epsilon$')

        else: 
            ax.set_yticks([])
        
        ax = fig.add_subplot(3, lp, i + 1 + 2*lp)
        for e ,c in zip(estimates, colors):
            ax.plot(e, range(len(e)), alpha=0.75, marker='', color=c)
            ax.axvline(np.median(e), linestyle=':', color=c)
        ax.axvline(np.median(estimates),  color='k')
        ax.axvline(loc_res['%s_maxmaxlike' % p] * scale, color='tomato', lw=2)
        ax.set_xlim(xlim)

        if i == 0:
            ax.set_ylabel('simulation t')
        else: 
            ax.set_yticks([])
        ax.set_xlabel('%s \n [%-5.2f; %-5.2f; %-5.2f] [%-5.2f ; %-5.2f]' % (p, np.median(estimates), 
                                                                    loc_res['%s_maxmaxlike' % p] * scale, loc_res[p] * scale, 
                                                                    np.std(estimates)/np.median(estimates),
                                                                    loc_res['%s_grd' % p]))
        

    plt.suptitle(title, fontsize=18) 
    plt.tight_layout()   
    if out_fig_file is None:
        plt.show()
    else:
        plt.savefig(out_fig_file)
    plt.close()


p_ranges = {'k_xl_max': [0.001 * 10**(-3), 2 * 10**(-3)], 
            'Px50': [-10, -0.1], 
            'Pg50': [-10, -0.1],
            'RAI': [1, 80], 
            'Zm': [0.05, 1], 
            'pi_R': [0, 1],
            'pi_F': [0, 6], 
            'beta_ww': [0, 1],
            's_wilt': [0, 1],
            's_star': [0, 1],
            'LAI': [0, 10], 
            'epsilon': [0, 1],
            'AA': [0, 1]}

sites = pd.read_csv('sel_sites.csv')

if __name__ == "__main__":

# MH-MCMC fit .................................................................................
    
    data_path = '../../DATA/WUS/WUS_input_data'
    fn = '_1'
    stress_type='dynamic'

    for criteria in ['bf', 'opt']:
        resultpath_plots =  '../../DATA/WUS/WUS_output/%s/figs/%s' % (criteria, fn)
        resultpath_pickles =  '../../DATA/WUS/WUS_output/%s/files/%s' % (criteria, fn)
        
        try:
            os.mkdir(resultpath_pickles)
        except:
            pass
        try:
            os.mkdir(resultpath_plots)
        except:
            pass
    for site in sites['siteID']: 
        for criteria, epsi in [['bf', None], ['opt', 1], ]:
            resultpath_plots =  '../../DATA/WUS/WUS_output/%s/figs/%s' % (criteria, fn)
            resultpath_pickles =  '../../DATA/WUS/WUS_output/%s/files/%s' % (criteria, fn)
            try:
                print(site, '.................')
                f_params = os.path.join(data_path, '%s_params.pickle' % (site))
                with open(f_params, 'rb') as f:
                    params = load(f)
                #print(params)

                unknown_params = ['Pg50', 'Px50', 'k_xl_max', 'RAI',]
                save_params = ['pi_T', 'pi_S', 'pi_R', 'pi_F', 
                                'epsilon', 'AA', 'tet_part', 'mean_ef', 'mean_tf', 'mean_di',
                                'Px50', 'Pg50', 'k_xl_max', 'RAI',  'beta_ww', 
                                's_wilt', 's_star', 'psi_wilt', 'psi_star']  

                save_params1 = ['pi_T', 'pi_S', 'pi_R', 'pi_F',  'epsilon', 'AA']
                save_params2 = ['Px50', 'Pg50', 'k_xl_max', 'RAI', 'beta_ww', 
                            's_wilt', 's_star',  'psi_wilt', 'psi_star'] 
                
                params['Zmi'] = params['Zm']
                params['Zm'] = params['Zm'] * 1.5
                loc_res = go_mhmcmc(params, unknown_params, save_params, epsi=epsi, stress_type=stress_type, num_pl=3, nbins=100)
                title='%s [%s]; NSE=%-5.2f; piF=%-5.2f piR=%-5.2f' % (site, loc_res['pft'],  loc_res['NSE'],
                                                                  loc_res['pi_F'], loc_res['pi_R'])
                
                out_fig_file = os.path.join(resultpath_plots, '%s_%s_1.png' % (site, params['IGBP']))
                plot_mcmc_results(loc_res, save_params1, burnin=0, out_fig_file=out_fig_file, title=title)

                out_fig_file = os.path.join(resultpath_plots, '%s_%s_2.png' % (site, params['IGBP']))
                plot_mcmc_results(loc_res, save_params2, burnin=0, out_fig_file=out_fig_file, title=title)

                out_fig_file = os.path.join(resultpath_plots, '%s_%s_ps.png' % (site, params['IGBP']))
                plot_ps(loc_res, out_fig_file=out_fig_file, title=title)

                outfile_res = os.path.join(resultpath_pickles, '%s.pickle'% site)
                with open(outfile_res, 'wb') as f:
                    dump(loc_res, f)
                print(title)
            except:
                print('xxx', site)
        

# combine results ................................................................................
    
    fn = '_1'

    for criteria in ['bf', 'opt']:
        res_dir = '../../DATA/WUS/WUS_output/%s/files/%s' % (criteria, fn)
        outfile = '../../DATA/WUS/WUS_output/combined/results_%s_%s.csv' % (criteria, fn)
        for ix, site in enumerate(sites['siteID'].values[:1]):
            print(site, '.................')
            outfile_res = os.path.join(res_dir, '%s.pickle' % site)
            with open(outfile_res, 'rb') as f:
                loc_res = load(f)
            for k in loc_res.keys():
                if k.endswith('_estimates'):
                    loc_res[k] = np.nan
            loc_res['p_obs'] = len(loc_res['p_obs'])
            loc_res['obs_l'] = len(loc_res['s_obs'])
            loc_res['s_obs'] = np.median(loc_res['s_obs'])
            loc_res['DOY_GS'] = loc_res['T_GS']
            results_all = pd.DataFrame(loc_res, index=[ix,])

        for ix, site in enumerate(sites['siteID'].values[1:]):
            try:
                print(site, '.................')
                outfile_res = os.path.join(res_dir, '%s.pickle' % site)

                with open(outfile_res, 'rb') as f:
                    loc_res = load(f)
                for k in loc_res.keys():
                    if k.endswith('_estimates'):
                        loc_res[k] = np.nan
                loc_res['p_obs'] = len(loc_res['p_obs'])
                loc_res['obs_l'] = len(loc_res['s_obs'])
                loc_res['s_obs'] = np.median(loc_res['s_obs'])
                loc_res['DOY_GS'] = loc_res['T_GS']
                loc_res = pd.DataFrame(loc_res, index=[ix+1,])
                results_all = pd.concat([results_all, loc_res])
            except:
                print(site, 'XXX.................')
        print(results_all)
        results_all.to_csv(outfile)
      

