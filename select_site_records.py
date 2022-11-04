import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
from pickle import dump, load

from data_management import *

warnings.filterwarnings('ignore')


sites_params = pd.read_csv('sel_sites.csv')


def plot_check_swci():
    plotpath = '../../PROJECTS/optimal_water_use_strategies/_figures/data_plots/swc_all_z'
    for site in sites_params['siteID'].values:
        print(site)
        fig_name =  os.path.join(plotpath, '%s_check_swc.png' % site)
        fig = plt.figure(figsize=(15, 10))
        for i in [1, 2, 3]:
            try:
                data = get_site_data_flx(site, swc_i=i)
                data_gs, doy_gs = select_growing_season(data, lai_th=[100, 1/3.], t_th=2)
                params = get_site_params(site, data_gs, swc_i=i)
                soil_tex_id = get_map_soil_text(site, params['lat'], params['lon'])
                data_gs['S'] = data_gs['SWC'] / params['n']
                years = [yi for yi in range(np.min(data_gs.index.year), np.max(data_gs.index.year) + 1) if yi in data_gs.index.year]
                data_gs = data_gs.resample('D').mean().dropna()
                data_gs['DOY'] = data_gs.index.dayofyear
                data_gs['Y'] = data_gs.index.year

                for y in years:
                    ll = len(data_gs[data_gs['Y']==y].dropna().index)
                    if ll < 0.8*len(list(doy_gs)):
                        print(ll, len(list(doy_gs)))
                        data_gs = data_gs[data_gs['Y']!=y]
                years = [yi for yi in range(np.min(data_gs.index.year), np.max(data_gs.index.year) + 1) if yi in data_gs.index.year]

                ax = fig.add_subplot(3, 2, 1 + (i-1)*2)
                
                for y in years:
                    ax.hist(data_gs[data_gs['Y']==y]['S'], density=True,  bins=np.linspace(0, 1, 31), alpha=0.33)
                ax.set_xlabel('S [-]', fontsize=16)
                ax.set_xlim([0, 1])
                ax.set_ylabel('p(S) [%s; %s] %s' % (i, params['Zm'], len(years)), fontsize=16)
                ax.hist(data_gs['S'], density=True,  bins=np.linspace(0, 1, 31), color = 'k', alpha = 0.5)
                ax.axvline(params['s_fc_rec'], linestyle=':', color = 'k')


                ax = fig.add_subplot(3, 2, 2 + (i-1)*2)
                for y in years:
                    ll = len(data_gs[data_gs['Y']==y].dropna().index)
                    ax.plot(data_gs[data_gs['Y']==y]['DOY'], data_gs[data_gs['Y']==y]['S'], label= '%s [%s]' % (y, ll), marker='.', lw=0)
                ax.set_ylim([0, 1])
                ax.set_xlim([1, 365])
                ax.set_ylabel('S  [%s; %s] %s' % (i, params['Zm'], len(years)), fontsize=16)
                ax.set_xlabel('Day of year', fontsize=16)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
            except:
                pass
            
        title = '%s ; %s (%s); %s / %d [%s]'% (site,
                                            params['pft'], params['Zr_m'],
                                            params['soil_tex_id'], soil_tex_id,
                                            params['swc_i_sel'])
        plt.suptitle(title, fontsize=18)
        plt.savefig(fig_name)
        plt.close()


def plot_check_ysel():
    #plotpath = '../../PROJECTS/optimal_water_use_strategies/_figures/data_plots/swc_nHH_DD'
    plotpath = '../../PROJECTS/optimal_water_use_strategies/_figures/data_plots/swc_ysel'
    for site in sites_params['siteID'].values:
        print(site, 'sel')
        fig_name =  os.path.join(plotpath, '%s_check_data_sel_fc2.png' % site)
        data = get_site_data_flx(site, x_years=True)

        fig = plt.figure(figsize=(15, 10))

        data_gs, doy_gs = select_growing_season(data, data, lai_th=[90, 0.5], t_th=2)
        params = get_site_params(site, data_gs, data)
        data_ = data
        data_['Y'] = data_.index.year
        for y in list(set(data_['Y'])):
            if y not in list(set(data_gs.index.year)):
                data_ = data_[data_['Y'] != y]
        n = np.ceil(np.nanmax(data_['SWC'] * 100)) / 100.
        data['S'] = data['SWC'] / params['n']
        data_gs['S'] = data_gs['SWC'] / params['n']
        print(n, params['n'])

        years = [yi for yi in range(np.min(data_gs.index.year), np.max(data_gs.index.year) + 1) if yi in data_gs.index.year]
        selected_datetimes = [dt for dt in data.index if dt.year in years]
        data = data.ix[selected_datetimes]
        
        ax = fig.add_subplot(2, 2, 1)
        for y in years:
            ax.hist(data_gs[data_gs['Y'] == y]['S'], density=True,  bins=np.linspace(0, 1, 31), alpha=0.33)
        ax.set_xlabel('S [-]', fontsize=16)
        ax.set_xlim([0, 1])
        ax.set_ylabel('p(S) [%s] %s' % ( params['Zm'], len(years)), fontsize=16)
        ax.hist(data_gs['S'], density=True,  bins=np.linspace(0, 1, 31), color = 'k', alpha = 0.5)
        ax.axvline(params['s_fc'], linestyle='--', color = 'k')
        ax.axvline(params['s_fc_ref'], linestyle='-', color = 'k')
        ax.axvline(params['s_h'], linestyle=':', color = 'k')

        ax = fig.add_subplot(2, 2, 2)
        for y in years:
            ax.plot(data_gs[data_gs['Y']==y]['DOY'], data_gs[data_gs['Y']==y]['S'], label= '%s' % (y), marker='.', lw=0)
        ax.set_ylim([0, 1])
        ax.set_xlim([1, 365])
        ax.set_ylabel('S  [%s]' % (len(years)), fontsize=16)
        ax.set_xlabel('Day of year', fontsize=16)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(data['S'], color= 'k', marker='o', lw=0)
        ax.plot(data_gs['S'], color= 'dodgerblue', marker='o', lw=0)
        ax.set_ylim([0, 1])
            
        title = '%s [%s, %s] ; %s [%s]; '% (site, params['swc_i_sel'], params['Zm'],
                                            params['pft'], params['Zr_m'],
                                            )
        plt.suptitle(title, fontsize=18)
        plt.savefig(fig_name)
        plt.close()


def plot_data(data_gs, data, params, site, title=None, out_fig_file=None):
    pft = sites_params[sites_params['siteID'] == site]['pft'].values[0]
    zm = sites_params[sites_params['siteID'] == site]['Zm'].values[0]

    gs_days = list(set(data_gs['DOY'].values))
    n_years = np.max(data_gs.index.year) - np.min(data_gs.index.year) + 1
    
    data_gs['DOY'] = data_gs.index.dayofyear
    data['DOY'] = data.index.dayofyear
    
    S_annual = data.groupby('DOY').mean()['S']
    S_gs = data_gs.groupby('DOY').mean()['S']
    
    lai_annual = data.groupby('DOY').mean()['LAI']
    lai_gs = data_gs.groupby('DOY').mean()['LAI']

    GPP_annual = data.groupby('DOY').median()['GPP']
    GPP_gs = data_gs.groupby('DOY').median()['GPP']

    E0_annual = data.groupby('DOY').median()['ETo_m']
    E0_gs = data_gs.groupby('DOY').median()['ETo_m']


    fig = plt.figure(figsize=(10, 12))
    
    ax = fig.add_subplot(3, 2, 1)
    ax.plot(S_annual, color='k', label='avg by DOY')
    ax.plot(S_gs, color='green', lw=2, label='growing season')
    ax.axhline(np.nanmean(data_gs['S']), linestyle='--', color='green', label='avg grow. seas')
    ax.set_ylabel('S [-] Zm=%s' % zm, fontsize=14)
    ax.set_xlabel('Day of year', fontsize=14)
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 365])
    ax.legend(frameon=False)

    ax = fig.add_subplot(3, 2, 2)
    ax.hist(data_gs['S'], density=True, color='green', bins=np.linspace(0, 1, 31), label='growing season')
    ax.hist(data['S'], density=True, color='k', alpha=0.5, bins=np.linspace(0, 1, 31), label='full record')
    ax.axvline(params['s_h'], color='tomato')
    ax.axvline(params['s_fc'], color='tomato')
    ax.axvline(params['s_fc_ref'], color='tomato', linestyle=':')
    ax.set_xlabel('S [-]', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylabel('p(S)', fontsize=14)
    ax.legend(frameon=False)


    ax = fig.add_subplot(3, 2, 3)
    ax.plot(E0_annual * 1000, color='k', label='avg by DOY')
    ax.plot(E0_gs * 1000, color='green', label='growing season')
    ax.set_ylabel('E0 [mm/d]', fontsize=14)
    ax.set_xlabel('Day of year', fontsize=14)
    ax.set_ylim([0, np.max(E0_gs) * 1.1*1000])
    ax.set_xlim([1, 365])
    ax.legend(frameon=False)

    ax = fig.add_subplot(3, 2, 4)
    m_df = data.resample('M').sum()
    m_rf = [np.nanmean(m_df[m_df.index.month == m]['RF'] * 1000) for m in range(1, 13)]
    ax.bar(range(1, 13), m_rf, color='k')
    m_df = data_gs.resample('M').sum()
    m_rf = [np.nanmean(m_df[m_df.index.month == m]['RF'] * 1000) for m in range(1, 13)]
    ax.bar(range(1, 13), m_rf, color='green')
    ax.set_ylabel('RF [mm]', fontsize=14)
    ax.set_xlabel('Month of year', fontsize=14)


    ax = fig.add_subplot(3, 2, 5)
    ax.plot(lai_annual, color='k', label='avg by DOY')
    ax.plot(lai_gs, color='green',  lw=2, label='growing season')
    ax.axhline(np.nanmean(data_gs['LAI']), linestyle='--', color='green', label='avg grow. seas')
    ax.axhline(np.percentile(data['LAI'].dropna(), 90), linestyle='--', color='limegreen', label='peak')
    ax.set_ylabel('LAI [m2/m2]', fontsize=14)
    ax.set_xlabel('Day of year', fontsize=14)
    ax.set_ylim([0, np.max(lai_annual) * 1.1])
    ax.set_xlim([1, 365])
    ax.legend(frameon=False)

    ax = fig.add_subplot(3, 2, 6)
    ax.plot(GPP_annual * 1000000, color='k', label='avg by DOY')
    ax.plot(GPP_gs * 1000000, color='green',  lw=2, label='growing season')
    ax.set_ylabel('GPP [umol/m2/s]', fontsize=14)
    ax.set_xlabel('Day of year', fontsize=14)
    ax.set_ylim([0, np.max(GPP_gs) * 1.1 * 1000000])
    ax.set_xlim([1, 365])
    ax.legend(frameon=False)

    if title is None:
        title = '%s [%s] nyears =%s; GSdays=%s ; ndata=%s' \
                 % (site, pft, n_years, len(gs_days), len(data_gs.index))
    plt.suptitle(title, fontsize=18)
    
    if out_fig_file is None:
        plt.show()
    else:
        plt.savefig(out_fig_file)


def pool_plot_data(site):
    plotpath = '../../DATA/OWUS/plots/data_growing_seasons'
    data_path = '../../DATA/WUS/WUS_input_data'
    pft = sites_params[sites_params['siteID'] == site]['pft'].values[0]

    igbp = sites_params[sites_params['siteID'] == site]['IGBP'].values[0]
    data = get_site_data_flx(site)
    data_gs, doy_gs = select_growing_season(data, lai_th=[90, 0.5], t_th=2)

    data['Y'] = data.index.year
    for y in list(set(data['Y'])):
        if y not in list(set(data_gs.index.year)):
            data = data[data['Y'] != y]

    params = get_site_params(site, data_gs, data)

    data['S'] = data['SWC'] / params['n']
    data_gs['S'] = data_gs['SWC'] / params['n']

    out_fig_file =  os.path.join(plotpath, '%s_%s.png' % (site, igbp))
    plot_data(data_gs, data, params, site, out_fig_file=out_fig_file)
    outfile_data_gs = os.path.join(data_path, '%s_gs_data.csv' % (site))
    outfile_data = os.path.join(data_path, '%s_data.csv' % (site))
    outfile_params = os.path.join(data_path, '%s_params.pickle' % (site))

    data_gs.to_csv(outfile_data_gs)
    data.to_csv(outfile_data)
    with open(outfile_params, 'wb') as f:
        dump(params, f)
    print('dump', site)



if __name__ == "__main__":
         

# plot check all SWC z records....................................................................................
    '''
    plot_check_swci()
    '''    


# plot check sel y SWC records....................................................................................
    '''
    plot_check_ysel()
    '''

# plots data & save sel data inputs ..............................................................................
    pool = multiprocessing.Pool(processes=2)
    pool.map(pool_plot_data, sites_params['siteID'].values)



        
    

