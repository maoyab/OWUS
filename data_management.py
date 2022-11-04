import sys 
import os
import numpy as np
from pandas import date_range
import pandas as pd
import xarray as xr
from datetime import datetime


Mw = 0.018          # [kg mol-1] Molar mass of water          
rho_w = 1000        # [kg m-3] Density of liquid water
mwratio = 0.622     # - Ratio molecular weight of water vapor/dry air
cp_air = 1005       # J/kg/k specific heat of air
Lv = 2.5008 * 10**6 # (J/kg) is the latent heat of vaporization of water

drive_w = '/Volumes/SLU_maoya/E'

sites_params = pd.read_csv( 'sel_sites.csv')
soil_params_file_RUC = '../../DATA/WUS//NLDAS_soilParams_RUC.csv'
soil_params = pd.read_csv(soil_params_file_RUC)
pft_params = pd.read_csv( '../../DATA/WUS//selected_pft_params_refs.csv')


data_directory = os.path.join(drive_w, 'Original_data/FLUXNET/FLUXNET2015_2020dl/unzipped')
sites_files = [f for f in os.listdir(data_directory) if f.startswith('FLX')]

lai_file = os.path.join(drive_w, 'Original_data/FLUXNET/lai-combined-1/lai-combined-1-MCD15A3H-006-results.csv')
df_lai = pd.read_csv(lai_file)

fv_file = os.path.join(drive_w, 'Original_data/FLUXNET/LAI/veg-pct-yearly/veg-pct-yearly-MOD44B-006-results.csv')
df_fv = pd.read_csv(fv_file)

BIF_f = os.path.join(drive_w, 'Original_data/FLUXNET/FLUXNET2015_2020dl/FLX_AA-Flx_BIF_ALL_20200217/FLX_AA-Flx_BIF_HH_20200217.csv')
df_BIF = pd.read_csv(BIF_f, error_bad_lines=False, engine='python')

def get_raster_data():
    #from osgeo import gdal
    #dataset = gdal.Open(filename)
    #transform = dataset.GetGeoTransform()
    #x_origin = transform[0]
    #y_origin = transform[3]
    #pixel_width = transform[1]
    #pixel_height = -transform[5]
    #cols = dataset.RasterXSize
    #rows = dataset.RasterYSize
    #band = dataset.GetRasterBand(1)
    #data = band.ReadAsArray(0, 0, cols, rows)

    zr_file_d = os.path.join(drive_w, 'Original_Data/Global_effective_plant_rooting_depth/zr_data.npy')
    zr_file_t = os.path.join(drive_w, 'Original_Data/Global_effective_plant_rooting_depth/zr_transform.npy')
    data = np.load(zr_file_d)
    transform = np.load(zr_file_t)

    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = -transform[5]
    return data, x_origin, y_origin, pixel_width, pixel_height

def get_est_Zrmap(lat, lon):
    data, x_origin, y_origin, \
    pixel_width, pixel_height =  get_raster_data()
    col = int((lon - x_origin) / pixel_width)
    row = int((y_origin - lat ) / pixel_height)
    try:
        x = data[row][col]
        if x >= 0:
            return x
        else:
            return np.nan
    except:
        print( row, col)

def get_map_soil_text(site, lat, lon):
    soil_tex_map_file = os.path.join(drive_w, 'Original_Data/GLDAS/GLDASp4_soiltexture_025d.nc4')
    ds_soil_map = xr.open_dataset(soil_tex_map_file)
    def __c_coor():
        # lat/lon sel. grid move because of empty gridcells
        if site in ['US-KS2', 'US-KS1',  'GF-Guy']:
            lon_c = lon - 0.125
            lat_c = lat
        elif site in ['IT-Cpz', 'IT-Noe', 'AR-SLu']:
            lon_c = lon + 0.125
            lat_c = lat
        elif site in ['CA-TP3', 'CA-TP4']:
            lon_c = lon
            lat_c = lat + 0.125
        else:
            lon_c = lon
            lat_c = lat
        return lat_c, lon_c

    lat_c, lon_c = __c_coor()
    ds_ti = ds_soil_map.sel(lat=lat_c, lon=lon_c, method='nearest')
    soil_tex_id = ds_ti['GLDAS_soiltex'].values[0]
    return soil_tex_id

def get_fluxnet_hm(site):
    zm_multi = {'AT-Neu': 2.5,
            'BE-Lon': 2.7,
            'BE-Vie': 40,
            'CA-SF3': 20,
            'CA-TP1': 5,
            'CA-TP3': 18,
            'CZ-BK1': 18,
            'DE-Geb': 3.5,
            'DK-Fou': 2.6,
            'FI-Hyy': 24,
            'FR-Gri': 3,
            'FR-LBr': 41.5,
            'IT-BCi': 2.25,
            'IT-CA1': 6.7,
            'IT-CA3': 5.5,
            'IT-Col': 30,
            'IT-Cpz': 25,
            'IT-PT1': 30,
            'IT-Ren': 33,
            'IT-SRo': 22,
            'IT-Tor': 2.5,
            'US-Blo': 10.5,
            'US-KS2': 3.5,
            'US-Ne1': 6,
            'US-Ne2': 6,
            'US-Ne3': 6,
            'US-Oho': 33,
            'US-PFa': 30,
            'US-Whs': 6.5,
            'US-Wkg': 6.4,
            }
    dfi = df_BIF[df_BIF['SITE_ID']==site]
    z_ids = dfi[dfi['DATAVALUE']=='USTAR']['GROUP_ID'].values 
    hm = []
    t_zm = []
    for z_id in z_ids:
        zmi = np.float(dfi[(dfi['GROUP_ID']==z_id) & (dfi['VARIABLE']=='VAR_INFO_HEIGHT')]['DATAVALUE'].values[0])
        hm.append(zmi)
        t_zmi = dfi[(dfi['GROUP_ID']==z_id) & (dfi['VARIABLE']=='VAR_INFO_DATE')]['DATAVALUE'].values[0]
        t_zm.append(t_zmi)
    if np.mean(hm)==hm[0]:
        hm = np.mean(hm)
    else:
        hm = zm_multi[site]
    return hm

def get_fluxnet_hc(site):
    dfi = df_BIF[df_BIF['SITE_ID']==site]
    hc = np.round(np.mean([np.float(i) for i in dfi[dfi['VARIABLE']=='HEIGHTC']['DATAVALUE'].values]), 1)
    return  hc

def get_fluxnet_zm(site, i):
    dfi = df_BIF[df_BIF['SITE_ID']==site]
    z_id = dfi[dfi['DATAVALUE']=='SWC_F_MDS_%s' % i]['GROUP_ID'].values[0]
    zm = np.float(dfi[(dfi['GROUP_ID']==z_id) & (dfi['VARIABLE']=='VAR_INFO_HEIGHT')]['DATAVALUE'].values[0])
    return -zm

def get_gldas_soil_data(soil_tex_id):
    MAXSMC = soil_params[soil_params['ID']==soil_tex_id]['MAXSMC'].values[0]
    DRYSMC = soil_params[soil_params['ID']==soil_tex_id]['DRYSMC'].values[0]
    BB = soil_params[soil_params['ID']==soil_tex_id]['BB'].values[0]
    SATPSI = soil_params[soil_params['ID']==soil_tex_id]['SATPSI'].values[0]  * -9.8067 * 10 ** -3 # m to MPa
    SATDK = soil_params[soil_params['ID']==soil_tex_id]['SATDK'].values[0] # m s-1
    return BB, SATPSI, SATDK, MAXSMC, DRYSMC

def s_to_pot_BC(s, b, Ps0):
    psi = Ps0 * (s ** - b)  
    return psi

def pot_to_s_BC(psi, b, Ps0):
    s = (psi / Ps0) ** (- 1 / b)
    return s

def stochastic_rf_char(rf):
    mu = np.nanmean(rf)
    var = np.nanvar(rf)
    l = 2 * mu ** 2 / var
    a = mu / l
    return a, l

def psychrometric_cste(atm_pressure):
    # Pa/C
    return cp_air * atm_pressure / (Lv * mwratio)

def vapor_pressure_slope(air_temp):
    # Pa/C
    a =  np.exp((17.27 * air_temp) / (air_temp + 237.3)) 
    b = (air_temp + 237.3) ** 2
    delta = a / b * 2.504 * 10 ** 6 
    return delta

def pt_ETo(Qn, air_temp, atm_pressure):
    #preistley Taylor
    delta = vapor_pressure_slope(air_temp)
    pt = 1.26 * delta / (delta + psychrometric_cste(atm_pressure)) * Qn
    return pt

def insert_LAI(df, site, df_p, df_lai):

    if site=='IT-Ro1':
        df_lai = df_lai[(df_lai['siteID']=='IT-Ro2')
                        & (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description']=='Good quality (main algorithm with or without saturation)') 
                        & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description']=='Significant clouds NOT present (clear)') 
                        & (df_lai['MCD15A3H_006_FparExtra_QC_Aerosol_Description']=='No or low atmospheric aerosol levels detected')
                        & (df_lai['MCD15A3H_006_FparLai_QC_SCF_QC_Description']=='Main (RT) method used, best result possible (no saturation)')
                        ]
    else:
        df_lai = df_lai[(df_lai['siteID'] == site)
                        & (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description']=='Good quality (main algorithm with or without saturation)') 
                        & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description']=='Significant clouds NOT present (clear)') 
                        & (df_lai['MCD15A3H_006_FparExtra_QC_Aerosol_Description']=='No or low atmospheric aerosol levels detected')
                        &(df_lai['MCD15A3H_006_FparLai_QC_SCF_QC_Description']=='Main (RT) method used, best result possible (no saturation)')
                        ]

    lai = df_lai['MCD15A3H_006_Lai_500m'].values

    date = [datetime.strptime(dt, '%m/%d/%Y')
                             for dt in df_lai['Date'].values]

    df_lai = pd.DataFrame({'LAI': lai}, index=date)
    df_lai['smooth_LAI'] = np.nan
    df_lai = df_lai.resample('D').mean()
    df_lai = df_lai.interpolate()
    df_lai = df_lai.reindex(pd.date_range(start='2003-01-01',
                                        end='2014-12-31',
                                        freq='D'))

    df_lai = df_lai[~((df_lai.index.month == 2) & (df_lai.index.day == 29))]
    df_lai['Year'] = df_lai.index.year
    years = [y for y in list(set( df_lai['Year'].values))]
    doy = []
    for y in years:
        doy = np.concatenate((doy, range(1,366)), axis=None)
    doy = [np.int(di) for di in doy]
    df_lai['DOY'] = doy
    
    m_avg_lai = []
    for c, year in enumerate(years):
        df_lai_ii = df_lai[df_lai['Year']==year]
        m_avg_lai.append(df_lai_ii['LAI'].values)
    m_avg_lai = zip(*m_avg_lai)
    m_avg_lai = [np.nanmean(mi) for mi in m_avg_lai]
    df_lai['smooth_LAI'] = [m_avg_lai[di-1] for di in df_lai['DOY'].values]
    m_avg_lai_r = np.convolve(df_lai['smooth_LAI'].values, np.ones(31) / 31, mode='same')
    m_avg_lai_r = list(m_avg_lai_r[365:365*2])
    m_avg_lai_r.insert(31 + 28, m_avg_lai_r[31+27])
    lai_dates = pd.date_range(start='2000-01-01',
                              end='2000-12-31',
                              freq='1D')

    df['LAI'] =  np.nan
    for month, day, lai in zip(lai_dates.month, lai_dates.day, m_avg_lai_r):
        df.loc[(df.index.day == day) & (df.index.month == month), 'LAI'] = lai
    return df

def insert_fpar(df, site, df_p, df_lai):

    if site=='IT-Ro1':
        df_lai = df_lai[(df_lai['siteID']=='IT-Ro2')
                        & (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description']=='Good quality (main algorithm with or without saturation)') 
                        & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description']=='Significant clouds NOT present (clear)') 
                        & (df_lai['MCD15A3H_006_FparExtra_QC_Aerosol_Description']=='No or low atmospheric aerosol levels detected')
                        & (df_lai['MCD15A3H_006_FparLai_QC_SCF_QC_Description']=='Main (RT) method used, best result possible (no saturation)')
                        ]
    else:
        df_lai = df_lai[(df_lai['siteID'] == site)
                        & (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description']=='Good quality (main algorithm with or without saturation)') 
                        & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description']=='Significant clouds NOT present (clear)') 
                        & (df_lai['MCD15A3H_006_FparExtra_QC_Aerosol_Description']=='No or low atmospheric aerosol levels detected')
                        & (df_lai['MCD15A3H_006_FparLai_QC_SCF_QC_Description']=='Main (RT) method used, best result possible (no saturation)')
                        ]

    fpar = df_lai['MCD15A3H_006_Fpar_500m'].values
    
    date = [datetime.strptime(dt, '%m/%d/%Y')
                             for dt in df_lai['Date'].values]

    df_fpar = pd.DataFrame({'fpar': fpar}, index=date)
    df_fpar['smooth_fpar'] = np.nan
    df_fpar = df_fpar.resample('D').mean()
    df_fpar = df_fpar.interpolate()
    df_fpar = df_fpar.reindex(pd.date_range(start='2003-01-01',
                                        end='2014-12-31',
                                        freq='D'))

    df_fpar = df_fpar[~((df_fpar.index.month == 2) & (df_fpar.index.day == 29))]
    df_fpar['Year'] = df_fpar.index.year
    years = [y for y in list(set( df_fpar['Year'].values))]
    doy = []
    for y in years:
        doy = np.concatenate((doy, range(1,366)), axis=None)
    doy = [np.int(di) for di in doy]
    df_fpar['DOY'] = doy
    
    m_avg_fpar = []
    for c, year in enumerate(years):
        df_fpar_ii = df_fpar[df_fpar['Year']==year]
        m_avg_fpar.append(df_fpar_ii['fpar'].values)
    m_avg_fpar = zip(*m_avg_fpar)
    m_avg_fpar = [np.nanmean(mi) for mi in m_avg_fpar]
    df_fpar['smooth_fpar'] = [m_avg_fpar[di-1] for di in df_fpar['DOY'].values]
    m_avg_fpar_r = np.convolve(df_fpar['smooth_fpar'].values, np.ones(31) / 31, mode='same')
    m_avg_fpar_r = list(m_avg_fpar_r[365:365*2])
    m_avg_fpar_r.insert(31 + 28, m_avg_fpar_r[31+27])
    fpar_dates = pd.date_range(start='2000-01-01',
                              end='2000-12-31',
                              freq='1D')

    df['fpar'] =  np.nan
    for month, day, fpar in zip(fpar_dates.month, fpar_dates.day, m_avg_fpar_r):
        df.loc[(df.index.day == day) & (df.index.month == month), 'fpar'] = fpar
    return df

def get_veg_fraction(site, df_fv0):
    if site=='IT-Ro1':
        df_fv = df_fv0[(df_fv0['siteID'] == 'IT-Ro2')]
    else:
        df_fv = df_fv0[(df_fv0['siteID'] == site)]

    f_h = np.round(np.mean(df_fv['MOD44B_006_Percent_NonTree_Vegetation'].values/100), 3)
    f_t = np.round(np.mean(df_fv['MOD44B_006_Percent_Tree_Cover'].values/100), 3)
    f_b = np.round(np.mean(df_fv['MOD44B_006_Percent_NonVegetated'].values/100), 3)
    f_v = 1 - f_b
    
    return f_h, f_t, f_b, f_v

def get_minmaxSWC_HH(site, swc_i, sel_years):
    f_site = [f for f in os.listdir(data_directory) if (f.startswith('FLX')) and (site in f)][0]
    dir_i = os.path.join(data_directory, f_site)
    [flx, site_name, flx15, fullset, years, suffix] = f_site.split('_')
    try:
        timestep = 'HH'
        f = '%s_%s_%s_%s_%s_%s_%s.csv' % (flx, site_name, flx15, fullset, timestep, years, suffix)
        data = pd.read_csv(os.path.join(dir_i, f), 
                          header = 0, index_col = 'TIMESTAMP_START', parse_dates = True, 
                          infer_datetime_format = True, na_values=-9999, error_bad_lines=False)
    except:
        timestep = 'HR'
        f = '%s_%s_%s_%s_%s_%s_%s.csv' % (flx, site_name, flx15, fullset, timestep, years, suffix)
        data = pd.read_csv(os.path.join(dir_i, f), 
                          header = 0, index_col = 'TIMESTAMP_START', parse_dates = True, 
                          infer_datetime_format = True, na_values=-9999, error_bad_lines=False)

    data['SWC_F_MDS'] = data['SWC_F_MDS_%s' % swc_i]
    data['SWC_F_MDS_QC'] = data['SWC_F_MDS_%s_QC' % swc_i]
    data.loc[data['SWC_F_MDS_QC'] > 0, 'SWC_F_MDS'] = np.nan
    data['SWC'] = data['SWC_F_MDS'] / 100.

    data = data[['SWC',]].dropna()
    data['Y'] = data.index.year

    for y in list(set(data['Y'])):
        if y not in sel_years:
            data = data[data['Y'] != y]
    print(np.percentile(data['SWC'].dropna(), 1), np.nanmax(data['SWC']), 'hh', site_name)
    return np.nanmax(data['SWC']), np.nanmin(data['SWC'])

def get_length_of_day(latitude, jd):
    n = jd - 2451545
    l = 280.46 + 0.9856474 * n
    while l < 0 :
        l = l + 360
    while l > 360:
        l = l - 360
    g = 357.528 + 0.9856003 * n
    while g < 0 :
        g = g + 360
    while g > 360:
        g = g - 360

    ecliptic_lon = l + 1.915 * np.sin(g) + 0.02 * np.sin(2 * g)
    obliquity = 23.439 - 0.0000004 * n
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_lon))

    lod = np.arccos(- np.tan(latitude) * np.tan(declination)) *  24 / np.pi
    if np.isnan(lod):
        lod = 12
    return lod

def exclude_years(df, site_name):
    if site_name == 'AT-Neu':   #
        x_year = [2007, ]       ## different porosity
    elif site_name == 'CA-Obs': ## different porosity
        x_year = [1999, ]
    elif site_name =='CA-SF2':  ## different porosity
        x_year = [2002,]
    elif site_name =='DE-Hai':  ## different porosity
        x_year = [2003, 2004] 
    elif site_name =='DE-Lnf':  ## different porosity
        x_year = [2002, 2003] 
    elif site_name =='FR-LBr':  ## GW
        x_year = [2006, 2007]
    elif site_name =='DK-Sor':  ## change in sensor positions
        x_year = [1997, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014] 
    elif site_name =='IT-Ren':  ## different porosity
        x_year = [2010, 2011, 2012, 2013]
    elif site_name =='IT-SRo':  ## different porosity
        x_year = [2012, ]
    elif site_name =='RU-Fyo':  ## different porosity
        x_year = [2006, ]
    elif site_name =='US-Prr':  ## possible gw
         x_year = [2014, ]
    elif site_name =='US-Syv':  ## different porosity
        x_year = [2012, 2013, 2014]
    elif site_name =='US-Var':  ## different porosity
        x_year = [2012, 2013, 2014, ]
    else:
        x_year = None
    #elif site_name =='GF-Guy':  ## at 0.2 m different porosity
    #    x_year = [2014, 2013, 2012, 2011, 2010, ]

    if x_year is not None:
        selected_datetimes = [dt for dt in df.index if dt.year not in x_year]
        df = df.loc[selected_datetimes]
    return df

def get_site_data_flx(site, swc_i=None, x_years=True):
    df_p = sites_params[sites_params['siteID'] == site]
    if swc_i is None:
        swc_i = df_p['swc_i'].values[0]

    soil_tex_id = df_p['soil_tex_id'].values[0]
        
    f_site = [f for f in os.listdir(data_directory) if (f.startswith('FLX')) and (site in f)][0]
    dir_i = os.path.join(data_directory, f_site)
    [flx, site_name, flx15, fullset, years, suffix] = f_site.split('_')
    timestep = 'DD'
    f = '%s_%s_%s_%s_%s_%s_%s.csv' % (flx, site_name, flx15, fullset, timestep, years, suffix)
    data = pd.read_csv(os.path.join(dir_i, f), 
                      header = 0, index_col = 'TIMESTAMP', parse_dates = True, 
                      infer_datetime_format = True, na_values=-9999, error_bad_lines=False)
    if x_years:
        data = exclude_years(data, site)

    data['SWC_F_MDS'] = data['SWC_F_MDS_%d' % swc_i]
    data['SWC_F_MDS_QC'] = data['SWC_F_MDS_%d_QC' % swc_i]
    data.loc[data['SWC_F_MDS_QC'] < 0.5, 'SWC_F_MDS'] = np.nan
    data['SWC'] = data['SWC_F_MDS'] / 100.                                # [-]

    data['VPD_air'] =  data['VPD_F_MDS'] / 10. * 1000                     # [Pa]
    data['P_air'] = data['PA_F'] * 1000                                   # [Pa]
    data['T_air'] = data['TA_F_MDS']                                      # [C]
    data['Qn'] = data['LE_F_MDS'] + data['H_F_MDS']                       # [W m-2]
    data.loc[data['P_F'] < 0.1, 'P_F'] = 0
    data['RF'] = data['P_F'] / 1000                                       # [m d-1]
    data['GPP'] = data['GPP_NT_VUT_REF'] * 10 ** (-6)                     # [mol/m2/s]
    
    data['ET_obs_mol'] = data['LE_F_MDS'] / Lv /  Mw                      # mol H20/m2/s
    data['ET_obs_m'] = data['LE_F_MDS'] / 28.94 / 1000                    # w/m2 to m/day

    data['LE_o'] = pt_ETo(data['Qn'], data['T_air'], data['P_air'])
    data['ETo_m'] = data['LE_o'] / 28.94 / 1000                           # w/m2 to m/day
    data['ETo_mol'] = data['LE_o'] / Lv /  Mw                             # mol H20/m2/s
    
    data = data[~((data.index.month == 2) & (data.index.day == 29))]
    
    data = insert_LAI(data, site_name, df_p, df_lai)
    data = insert_fpar(data, site_name, df_p, df_lai)

    data = data[['RF', 'T_air', 'VPD_air', 'SWC', 
                 'ET_obs_mol', 'ETo_mol', 'GPP',
                 'ET_obs_m', 'ETo_m', 'LAI', 'fpar']]

    data = data.dropna()
    return data

def select_growing_season_doy(data, gpp_th=[95, 0.10], t_th=2, lai_th=None):
    #select growing season DOY
    data['DOY'] = data.index.dayofyear
    data['Y'] = data.index.year
    data_doy = data[['GPP', 'LAI', 'T_air', 'DOY']].groupby('DOY').median().dropna()
    if lai_th is None:
        gpp_th = np.percentile(data_doy['GPP'].dropna(), gpp_th[0]) * gpp_th[1]
        data_doy_gs = data_doy[(data_doy['GPP'] > gpp_th) & (data_doy['T_air'] > t_th)]
        doy_gs = [dd for dd, ddi in zip(data_doy_gs.index[:-1], data_doy_gs.index[1:]) if (ddi - dd) == 1]
        doy_gs = range(doy_gs[0], doy_gs[-1] + 1)
    else:
        lai_th = np.percentile(data_doy['LAI'].dropna(), lai_th[0]) * lai_th[1]
        #lai_th = np.percentile(data_doy['LAI'].dropna(), lai_th[0])
        data_doy_gs = data_doy[(data_doy['LAI'] > lai_th) & (data_doy['T_air'] > t_th)]
        doy_gs = [dd for dd, ddi in zip(data_doy_gs.index[:-1], data_doy_gs.index[1:]) if (ddi - dd) == 1]
        doy_gs = range(doy_gs[0], doy_gs[-1] + 1)


    selected_days = [i for i in data.index if i.dayofyear in doy_gs]
    data_gs = data.loc[selected_days]
    
    years = [yi for yi in range(np.min(data_gs.index.year), np.max(data_gs.index.year) + 1) if yi in data_gs.index.year]
    data_gs = data_gs.resample('D').mean().dropna()
    
    for y in years:
        ll = len(data_gs[data_gs['Y'] == y].dropna().index)
        if ll < 0.8 * len(list(doy_gs)):
            data_gs = data_gs[data_gs['Y'] != y]
            data = data[data['Y'] != y]

    return data_gs, doy_gs

def select_growing_season(data, gpp_th=[95, 0.10], t_th=2, lai_th=None):
    # select growing season months
    data['DOY'] = data.index.dayofyear
    data['M'] = data.index.month
    data['Y'] = data.index.year
    data_doy = data[['GPP', 'LAI', 'T_air', 'DOY', 'M']].groupby('DOY').mean().dropna()
    data_m = data[['GPP', 'LAI', 'T_air', 'M']].groupby('M').mean().dropna()
    
    if lai_th is None:
        th_m =  np.percentile(data_m['GPP'].dropna(), gpp_th[0]) * gpp_th[1]
        data_m = data_m[(data_m['GPP'] >= th_m) & (data_m['T_air'] >= t_th) ]
        months_gs = list(data_m.index)
    else:
        th_m =  np.percentile(data_m['LAI'].dropna(), lai_th[0]) * lai_th[1]
        th_m = np.max([0.25, th_m])
        data_m = data_m[(data_m['LAI'] >= th_m) & (data_m['T_air'] >= t_th) ]
        months_gs = list(data_m.index)

    selected_days = [i for i in data.index if i.month in months_gs]
    data_gs = data.loc[selected_days]
    doy_gs = [dd for dd in data_doy['M'] if dd in months_gs]
    
    years = list(data_gs.index.year)
    data_gs = data_gs.resample('D').mean().dropna()
    
    for y in years:
        ll = len(data_gs[data_gs['Y'] == y].dropna().index)
        if ll < 0.8 * len(list(doy_gs)):
            data_gs = data_gs[data_gs['Y'] != y]
            data = data[data['Y'] != y]

    return data_gs, doy_gs

def get_site_params(site, data, data_0, swc_i=None):

    lat = sites_params[sites_params['siteID'] == site]['lat'].values[0]
    lon = sites_params[sites_params['siteID'] == site]['lon'].values[0]

    swc_i_sel =  sites_params[sites_params['siteID'] == site]['swc_i'].values[0]
    soil_tex_id = sites_params[sites_params['siteID'] == site]['soil_tex_id'].values[0]
    if swc_i is None:
        swc_i = swc_i_sel
        zm = sites_params[sites_params['siteID'] == site]['Zm'].values[0]
    else:
        zm = get_fluxnet_zm(site, swc_i)   # [m] measurement depth
    
    pft = sites_params[sites_params['siteID'] == site]['pft_0'].values[0]
    pft0 = sites_params[sites_params['siteID'] == site]['pft'].values[0]
    igbp = sites_params[sites_params['siteID'] == site]['IGBP'].values[0]
    df_pft = pft_params[pft_params['PFT'] == pft0]
    clim =  sites_params[sites_params['siteID'] == site]['climate'].values[0]
    
    params = {'siteID': site,
              'pft': pft,
              'IGBP': igbp,
              'clim': clim,
              'soil_tex_id': soil_tex_id,
              'swc_i': swc_i,
              'swc_i_sel': swc_i_sel,
              'Zm': -zm,
              'lat': lat,
              'lon': lon,
              }

    gs_days = list(set(data['DOY'].values))
    td = [get_length_of_day(params['lat'], jd) for jd in gs_days]
    params['Td'] = np.ceil(np.nanmean(td) * 3600.)
    params['T_GS'] = len(gs_days)
    params['DOY_GS'] = gs_days
    
    # climate characteristics
    params['Eo']  =  np.nanmean(data['ETo_m'])                                    # [m d-1] potential evaporation
   
    params['D_mean']  =  np.nanmean(data['VPD_air'])                              # [Pa] Vapor pressure deficit
    params['D_peak']  =  np.percentile(data['VPD_air'].dropna(), 95)              # [Pa] Vapor pressure deficit
    
    rf_alpha, rf_lambda = stochastic_rf_char(data['RF'].values)
    params['rf_alpha'] = rf_alpha                                                  # [m d-1] rainfall depth
    params['rf_lambda'] = rf_lambda                                                # [d-1] rainfall frequency 
    params['RF_a'] = np.nanmean(data_0['RF'])                                      # [m d-1] Mean annual precipitation
    params['RF'] = np.nanmean(data['RF'])                                          # [m d-1] Mean gs precipitation
    
    params['AI_a'] = np.nanmean(data_0['ETo_m']) / np.nanmean(data_0['RF'])         # aridity index (annual)
    params['EF_a'] = np.nanmean(data_0['ET_obs_m']) / np.nanmean(data_0['RF'])      # evaporative fraction (annual)
    params['AI'] = np.nanmean(data['ETo_m']) / np.nanmean(data['RF'])               # aridity index (growing season)
    params['EF'] = np.nanmean(data['ET_obs_m']) / np.nanmean(data['RF'])            # evaporative fraction (growing season)
    params['GPP_mean'] = np.nanmean(data['GPP'])
    params['ET_mean'] = np.nanmean(data['ET_obs_m'])
   
    # soil physical characteristics
    BB, SATPSI, SATDK, MAXSMC, DRYSMC = get_gldas_soil_data(soil_tex_id)
    
    params['b'] = BB
    params['Ps0'] = SATPSI
    params['Ks'] = SATDK * params['Td'] 

    swcmax = np.ceil(np.nanmax(data_0['SWC']) * 100) / 100.
    params['n'] = swcmax
    data['S'] = data['SWC'] / params['n']
    data_0['S'] = data_0['SWC'] / params['n']

    if s_to_pot_BC(np.nanmin(data_0['S'].values), params['b'], params['Ps0']) < -10:     
        params['b'] = (np.log(10) - np.log(-params['Ps0'])) / (np.log(1) - np.log(np.nanmin(data_0['S'].values)))
    
    sh = pot_to_s_BC(-10, params['b'], params['Ps0']) 
    params['s_h'] = np.floor(sh * 100) / 100. - 0.01
    
    params['s_fc_ref'] = pot_to_s_BC(-0.03, params['b'], params['Ps0'])
    data_0['ds/dt'] = data_0['S'].diff()
    sm_peaks = [smi for dsi, smi in zip(data_0['ds/dt'].values,
                                        data_0['S'].values)
                                        if dsi > 0]
    s_fc_r = np.percentile(sm_peaks, 95) 
    params['s_fc_rec'] = s_fc_r
    params['s_fc'] = np.max([params['s_fc_ref'], params['s_fc_rec']])

    # Root characteristics
    zr_est = get_est_Zrmap(params['lat'], params['lon'])  # Yang 2016 WRR dataset
    if np.isnan(zr_est):
        zr_est = get_est_Zrmap(params['lat'], params['lon'] + 0.25)
    if np.isnan(zr_est):
        zr_est = get_est_Zrmap(params['lat'] + 0.25, params['lon'])
    params['Zr_est'] = zr_est
    params['Zr_m'] = df_pft['Zr_mean'].values[0]        # [m] mean rooting depth - PFT refs           
    params['RAI'] = df_pft['RAI'].values[0]             # root area index m2 roots/ m2 ground - PFT ref
    params['dr'] = 0.0005                               # [m] Fine root diameter Manzoni 2014 

    # plant characteristics
    params['hc'] = get_fluxnet_hc(site)                 # [m] canopy height fluxnet metadata

    # plant hydraulic characteristics  - PFT references
    params['k_xl_max'] = df_pft['kl_max'].values[0]     # [kg m-1 s-1 MPa-1] Maximum xylem conductivity, leaf specific
    params['Px50'] = df_pft['Px50'].values[0]           # [MPa] Water potential at 50% loss of xylem conductivity
    
    # leaf characteristics  - PFT ref
    params['gs_max_ref'] = df_pft['g_max'].values[0]    # [mol m-2 s-1] Maximum stomatal conductance
    params['Pg50'] = -1.5                               # [MPa] Water potential at 50% stomatal closure placeholder
    
    # surface characteristics
    f_h, f_t, f_b, f_v = get_veg_fraction(site, df_fv)
    params['frac_H'] = f_h
    params['frac_T'] = f_t
    params['frac_B'] = f_b
    params['frac_V'] = f_v

    params['LAI_s90'] = np.nanpercentile(data_0['LAI'], 90)  # surface leaf area index m2 leaf/ m2 ground - peak of growing season
    params['LAI_s'] = np.nanmean(data['LAI'])                # surface leaf area index m2 leaf/ m2 ground - mean of growing season
    params['LAI_v'] = params['LAI_s'] / params['frac_V']     # surface leaf area index m2 leaf/ m2 veg area 
    params['LAI'] = params['LAI_s']

    params['fpar_s90'] = np.nanpercentile(data_0['fpar'], 90)  # surface leaf area index m2 leaf/ m2 ground - peak of growing season
    params['fpar_s'] = np.nanmean(data['fpar'])                # surface leaf area index m2 leaf/ m2 ground - mean of growing season
    params['fpar_v'] = params['fpar_s'] / params['frac_V']     # surface leaf area index m2 leaf/ m2 veg area 
    params['fpar'] = params['fpar_s']

    params['s_obs'] = data['S'].dropna().values
    p_obs = np.histogramdd([params['s_obs']], [np.linspace(0, 1, 101)])[0]
    params['p_obs'] = p_obs / np.sum(p_obs)
    params['obs_l'] = [np.percentile(params['s_obs'], qi / 365. * 100) for qi in range(1, 365)]
    params['l_obs'] = len(params['s_obs'])
    params['nYear'] = len(list(set(data['Y'])))

    return params



if __name__ == "__main__":
    pass