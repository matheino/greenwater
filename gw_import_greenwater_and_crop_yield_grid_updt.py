import os
import gzip
import numpy as np
import h5py
import rasterio
import netCDF4
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def import_irrig_area_mat(): 
# Import data for irrigated areas.
    irrig_area = np.swapaxes(h5py.File(r'D:\work\research\greenwater\data\irrig\hyde_final_hyde_05_annual.mat')['hyde_final_hyde_05_int'][()],0,2)
    years = np.arange(1900,2006,1)

# Since irrig area data is only until 2005, repeat irrig area values of 2005 six times, so that the data expands to 2011.
    irrig_area = np.dstack((irrig_area[:,:,years >= 1981],np.repeat(irrig_area[:,:,-1][:,:,np.newaxis], 6, axis = 2)))
# Calculate and irrigated area factor to scale the harvested irrigated area data relative to irrigated areas in year 2000.
    years_updt = np.hstack((years[years >= 1981],np.array((2006,2007,2008,2009,2010,2011))))
    
    irrig_area_2000 = irrig_area[:,:,years_updt == 2000]
    irrig_area_factor = irrig_area / irrig_area_2000
    
    # correct exceptions: if no change observed set factor to 1.0 and,
    # if there's an infinite change, relative to year 2000
    # change to large number to avoid nan when multiplying
    irrig_area_factor[np.isnan(irrig_area_factor),...] = 1.0
    irrig_area_factor[np.isinf(irrig_area_factor),...] = 10**20

    return irrig_area_factor[:,:,1:-1]


def import_mirca_total_harvested_area():
# Import harvested areas data (MIRCA2000).
    os.chdir('D:\work\data\MIRCA2000\harvested_area_grids')
              
    crop_ids = {'maize':'crop02',
                'rice':'crop03',
                'soy':'crop08',
                'wheat':'crop01'}

    mirca_ha_irc = {}
    mirca_ha_rfc = {}
    for crop in crop_ids.keys():
        file_name_irc = 'annual_area_harvested_irc_'+crop_ids[crop]+'_ha_30mn.asc.gz'
        file_name_rfc = 'annual_area_harvested_rfc_'+crop_ids[crop]+'_ha_30mn.asc.gz'
        
        with gzip.open(file_name_irc, 'rb') as file_in:            
            with rasterio.open(file_in) as raster_in:
                ha_per_crop = np.squeeze(raster_in.read())
                ha_per_crop[np.isnan(ha_per_crop)] = 0.0
                mirca_ha_irc[crop] = ha_per_crop
                raster_in.close()          
            file_in.close()
                
        with gzip.open(file_name_rfc, 'rb') as file_in:            
            with rasterio.open(file_in) as raster_in:
                ha_per_crop = np.squeeze(raster_in.read())
                ha_per_crop[np.isnan(ha_per_crop)] = 0.0
                mirca_ha_rfc[crop] = ha_per_crop
                raster_in.close()
            file_in.close()
        

    return mirca_ha_irc, mirca_ha_rfc

def import_iizumi_crop_data(crop):
# Import crop yield raster data.
    path = os.path.join(r'D:\work\data\iizumis_data\Iizumi_2019\dias\data\GDHYv1.2',crop)
    os.chdir(path)    
    
    file_list = []
    yield_data = np.zeros((360,720,31))

    for i, file_name in enumerate(os.listdir(path), 0):
        
        nc = netCDF4.Dataset(file_name, mode = 'r', format = 'NETCDF4')
        lon = nc['lon'][:]
        lat = nc['lat'][:]
        yield_annual = np.flip(nc['var'][:],axis=0)
        yield_annual = np.roll(yield_annual, np.sum(lon < 180),axis = 1)
        yield_data[:,:,i] = yield_annual
        file_list.append(file_name)
        
    yield_data[yield_data < 0] = np.nan
    
    yield_data = yield_data[:,:,np.argsort(file_list)]
    
    lon = np.roll(lon, np.sum(lon < 180))
    lon = ((lon+180) % 360)-180
    
    # print(lat)
    # print(lon)
    
    return yield_data[:,:,1:-1]


def import_iizumi_crop_data_all_crops():
# Combine crop yield data into a single dictionary.
    crop_yield_data = {'maize':import_iizumi_crop_data('maize_major'),
                       'rice':import_iizumi_crop_data('rice_major'),
                       'soy':import_iizumi_crop_data('soybean'),
                       'wheat':import_iizumi_crop_data('wheat')}
    return crop_yield_data


def import_greenwater_data():
# Import and standardize greenwater data
    os.chdir(r'D:\work\research\greenwater\data')    
    file_name = 'lpjml_princeton_hist_pressoc_co2_rainfusegreen_global_monthly_1971_2012.nc'
    
    gw_xr_monthly = xr.open_dataarray(file_name, decode_times = False)
    
    months_included = pd.date_range(pd.datetime(1971,1,1), pd.datetime(2012,12,31), freq = 'm')
    
    gw_xr_monthly['time'] = months_included
    
    seconds_per_month =  months_included.daysinmonth * 24 * 60 * 60
    seconds_per_month = xr.DataArray(seconds_per_month, dims= 'time', coords = {'time': months_included})

    gw_xr_monthly_tot = gw_xr_monthly * seconds_per_month
    gw_xr_yrly_tot = gw_xr_monthly_tot.resample(time = '1Y', keep_attrs = True).sum('time', skipna = False)
    
    days_per_year = xr.DataArray(months_included.daysinmonth, dims= 'time', coords = {'time': months_included}) \
                     .resample(time = '1Y', keep_attrs = True) \
                     .sum('time')
                     
    gw_xr_yrly_per_day = (gw_xr_yrly_tot / days_per_year) \
                         .sel(time = slice('1971-01-01', '2010-12-31'))
                             
    gw_xr_yrly_per_day_anom = (gw_xr_yrly_per_day - gw_xr_yrly_per_day.mean('time')) / gw_xr_yrly_per_day.std('time')
    
    gw_np = gw_xr_yrly_per_day.sel(time = slice('1982-01-01', '2010-12-31')).transpose('lat','lon','time').values
    
    gw_anom_np = gw_xr_yrly_per_day_anom.sel(time = slice('1982-01-01', '2010-12-31')).transpose('lat','lon','time').values
    
    return gw_np, gw_anom_np


def export_data(data, crop, years_nc, export_note):
# Writes raster yield data into netcdf.
    lats_nc = np.linspace(89.75,-89.75,360)
    lons_nc = np.linspace(-179.75,179.75,720)
    
    data_xarray = xr.DataArray(data,
                               dims = ('latitude','longitude','time'),
                               coords={'latitude': lats_nc,
                                       'longitude': lons_nc,
                                       'time': years_nc
                                       }).to_dataset(name = export_note)
    
   
    # Modify data to align with Esha's fishnet:    
    data_xarray = data_xarray.sel(latitude = slice(83.5, -90))
    
    if 'rainfed_yield' in export_note:
        data_xarray.attrs['unit'] = 't ha-1'
    elif 'anomaly' in export_note:
        data_xarray.attrs['unit'] = 'z-score'
    else:
        data_xarray.attrs['unit'] = 'kg m-2 day-1'

    os.chdir(r'D:\work\research\greenwater\iizumi_yield_analysis\results\sep2020')

    data_xarray.to_netcdf('raster_'+crop+'_'+export_note+'.nc')

    data_xarray[export_note].to_dataframe() \
        .to_csv('table_'+crop+'_'+export_note+'.csv')



if __name__ == "__main__":    

# Define variables
    crops = ['maize','rice','soy','wheat']
    years = np.linspace(1982,2010,29).astype(int)
    
# Import data
    irrig_area_factor = import_irrig_area_mat()
    mirca_ha_irc, mirca_ha_rfc = import_mirca_total_harvested_area()
    crop_yield_data = import_iizumi_crop_data_all_crops()
    gw, gw_anom = import_greenwater_data()
    
    gw_data_yield = {}
    gw_anom_data_yield = {}

    
    for crop in crops:

        gw_data_yield[crop] = np.copy(gw)

        gw_anom_data_yield[crop] = np.copy(gw_anom)
        
# Mask out irrigated areas from crop yield data
        mirca_ha_tot = mirca_ha_rfc[crop] + mirca_ha_irc[crop]
# Change zero to nan to avoid dividing with zero.
        mirca_ha_tot[mirca_ha_tot == 0] = np.nan
        mirca_ha_irc_perc = mirca_ha_irc[crop] / mirca_ha_tot
# Create a mask of rainfed areas (irrigated areas < 10%)
        rfc_boolean = (mirca_ha_irc_perc[:,:,np.newaxis] * irrig_area_factor) < 0.1

# Change cells that are not considered rainfed to nan
        crop_yield_data[crop][rfc_boolean == False] = np.nan
        
        gw_data_yield[crop][np.isnan(crop_yield_data[crop])] = np.nan
        gw_anom_data_yield[crop][np.isnan(crop_yield_data[crop])] = np.nan
        
# Export data
        export_data(crop_yield_data[crop], crop, years, 'rainfed_yield')
        
        export_data(gw_data_yield[crop], crop, years, 'green_water_yield_mask')
        export_data(gw_anom_data_yield[crop], crop, years, 'green_water_anomaly_yield_mask')
    
    export_data(gw, 'all_crops_plus', years, 'green_water_no_mask')
    export_data(gw_anom, 'all_crops_plus', years, 'green_water_anomaly_no_mask')









