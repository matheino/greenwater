import os
import gzip
import numpy as np
import h5py
import rasterio
import netCDF4
import geopandas as gdp
import rasterstats as rs
import affine
import pandas as pd
import copy


def import_irrig_area_mat(): 
# Import data for irrigated areas.
    irrig_area = np.swapaxes(h5py.File(r'D:\work\research\greenwater\data\irrig\hyde_final_hyde_05_annual.mat')['hyde_final_hyde_05_int'][()],0,2)
    years = np.arange(1900,2006,1)

# Since irrig area data is only until 2005, repeat irrig area values of 2005 six times, so that the data expands to 2011.
    irrig_area = np.dstack((irrig_area[:,:,years >= 1981],np.repeat(irrig_area[:,:,-1][:,:,np.newaxis], 6, axis = 2)))
# Calculate and irrigated area factor to scale the harvested irrigated area data relative to irrigated areas in year 2000.
    years_updt = np.hstack((years[years >= 1981],np.array((2006,2007,2008,2009,2010,2011))))
    irrig_area[irrig_area == 0.0] = np.nan
    irrig_area_factor = irrig_area / irrig_area[:,:,years_updt == 2000]
    irrig_area[np.isnan(irrig_area)] = 1.0
    
    return irrig_area_factor


def import_mirca_total_harvested_area():
# Import harvested areas data (MIRCA2000).
    os.chdir('D:\work\data\MIRCA2000\harvested_area_grids')
    
    file_list_irc = [file for file in os.listdir('D:\work\data\MIRCA2000\harvested_area_grids') if '30mn.asc.gz' in file.lower() and 'irc' in file.lower()]
    file_list_rfc = [file.replace('irc','rfc') for file in file_list_irc]
    
    mirca_ha_allcrops_irc = np.zeros((int(180/0.5),int(360/0.5)))
    mirca_ha_allcrops_rfc = np.zeros((int(180/0.5),int(360/0.5)))
    
    for file_name_irc, file_name_rfc in zip(file_list_irc, file_list_rfc):
        
        with gzip.open(file_name_irc, 'rb') as file_in:            
            with rasterio.open(file_in) as raster_in:
                ha_per_crop = np.squeeze(raster_in.read())
                ha_per_crop[np.isnan(ha_per_crop)] = 0.0
                mirca_ha_allcrops_irc += ha_per_crop
                raster_in.close()
            file_in.close()
       
        with gzip.open(file_name_rfc, 'rb') as file_in:            
            with rasterio.open(file_in) as raster_in:
                ha_per_crop = np.squeeze(raster_in.read())
                ha_per_crop[np.isnan(ha_per_crop)] = 0.0
                mirca_ha_allcrops_rfc += ha_per_crop
                raster_in.close()
            file_in.close()
              
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
        

    return mirca_ha_allcrops_irc, mirca_ha_allcrops_rfc, mirca_ha_irc, mirca_ha_rfc


def import_iizumi_crop_data(crop):
# Import crop yield raster data.
    path = os.path.join(r'D:\work\data\iizumis_data\Iizumi_2019\dias\data\GDHYv1.2',crop)
    os.chdir(path)    
    
    
    file_list = []
    yield_data = np.zeros((360,720,31))

    for i, file_name in enumerate(os.listdir(path), 0):
        
        nc = netCDF4.Dataset(file_name, mode = 'r', format = 'NETCDF4')
        lon = nc['lon'][:]
        yield_annual = np.flip(nc['var'][:],axis=0)
        yield_annual = np.roll(yield_annual, np.sum(lon < 180),axis = 1)
        yield_data[:,:,i] = yield_annual
        file_list.append(file_name)
    yield_data[yield_data < 0] = np.nan
    
    yield_data = yield_data[:,:,np.argsort(file_list)]
    
    return yield_data


def import_iizumi_crop_data_all_crops():
# Combine crop yield data into a single dictionary.
    crop_yield_data = {'maize':import_iizumi_crop_data('maize_major'),
                       'rice':import_iizumi_crop_data('rice_major'),
                       'soy':import_iizumi_crop_data('soybean'),
                       'wheat':import_iizumi_crop_data('wheat')}
    return crop_yield_data


def import_land_area():
# Import rasterized information about land area (2.5 arc-min resolution).
    os.chdir(r'D:\work\data\spatial_mask_data\2_5_arcmin_land_mask\gpw-v4-land-water-area-rev11_landareakm_2pt5_min_tif')
    with rasterio.open(r'gpw_v4_land_water_area_rev11_landareakm_2pt5_min.tif') as raster:
        land_area = np.squeeze(raster.read())
        raster.close()
    land_area[land_area<=0] = 0.0
#    plt.imshow(land_area);plt.colorbar()
    return land_area    


def combine_borders(land_cntry, cntry_borders, state_borders):
# Replace country polygons with state polygons, if land area is more than 10^6 km2.
    cntry_large = land_cntry.loc[land_cntry['area'] > 10**6]['area_name'].tolist()
    cntry_small = land_cntry.loc[land_cntry['area'] <= 10**6]['area_name'].tolist()

    long_cntry_names_for_combined_data = cntry_borders[['ADM0_A3','NAME_EN']].rename(columns={'ADM0_A3':'cntry_name','NAME_EN':'cntry_name_long'})

    cntry_borders = cntry_borders.loc[cntry_borders['ADM0_A3'].isin(cntry_small)].rename(columns={'ADM0_A3':'cntry_name','NAME_EN':'area_name'}) 
    state_borders = state_borders.loc[state_borders['adm0_a3'].isin(cntry_large)].rename(columns={'adm0_a3':'cntry_name','name_en':'area_name'}) 

    combined_borders = cntry_borders.append(state_borders)
    combined_borders = combined_borders.merge(long_cntry_names_for_combined_data, on = 'cntry_name', how = 'left')
    
    combined_names_area = combined_borders['area_name'].tolist()
    combined_names_cntry = combined_borders['cntry_name'].tolist()
    combined_names_cntry_long = combined_borders['cntry_name_long'].tolist()
    
    return combined_borders, combined_names_area, combined_names_cntry, combined_names_cntry_long


def import_shape():
# Imports data for the spatial units used.
    os.chdir(r'D:\work\data\map_files\administrative_areas\Ne\ne_10m_admin_0_countries')
    cntry_borders = gdp.read_file('ne_10m_admin_0_countries.shp',encoding='utf-8')
    os.chdir(r'D:\work\data\map_files\administrative_areas\Ne\ne_10m_admin_1_states_provinces')
    state_borders = gdp.read_file('ne_10m_admin_1_states_provinces.shp',encoding='utf-8')
    
    cntry_borders = cntry_borders[['geometry','NAME_EN','ADM0_A3']]
    state_borders = state_borders[['geometry','name_en','adm0_a3']]    
    
    cntry_names = cntry_borders['ADM0_A3'].tolist()
    cntry_names_long = cntry_borders['NAME_EN'].tolist()
    
    return cntry_borders, state_borders, cntry_names, cntry_names_long


if __name__ == "__main__":    

# Define variables
    crops = ['maize','rice','soy','wheat']
    years = np.linspace(1982,2010,29).astype(int)
# Affine from_gdal parameters:
# 1st: x-coord of upper left corner of upper left pixel
# 2nd: pixel width
# 3rd: row rotation
# 4th:y_coord of upper left corner of upper left pixel
# 5th: column rotation
# 6th: pixel height (typically negative)
    affine_land = affine.Affine.from_gdal(-180, 0.041666666666666664, 0.0,  90, 0.0, -0.041666666666666664)
    affine_info = affine.Affine.from_gdal(-180, 0.5, 0.0,  90, 0.0, -0.5)
    
# Import data
    irrig_area = import_irrig_area_mat()
    mirca_ha_allcrops_irc, mirca_ha_allcrops_rfc, mirca_ha_irc, mirca_ha_rfc = import_mirca_total_harvested_area()
    crop_yield_data = import_iizumi_crop_data_all_crops()
    cntry_borders, state_borders, cntry_names, cntry_names_long = import_shape()
    land_area = import_land_area()
    
# Define new shape based on land area
    land_cntry = pd.DataFrame(rs.zonal_stats(vectors=cntry_borders['geometry'], raster = land_area, affine = affine_land, stats = 'sum', nodata = -999))['sum']
    land_cntry = pd.DataFrame({'area_name': cntry_names, 'area': land_cntry}, columns = ['area_name','area'])            
    combined_borders, combined_names_area, combined_names_cntry, combined_names_cntry_long = combine_borders(land_cntry, cntry_borders, state_borders)
    
    for crop in crops:
# Mask out irrigated areas from crop yield data
        
        mirca_ha_tot = mirca_ha_rfc[crop] + mirca_ha_irc[crop]
# Change zero to nan to avoid dividing with zero.
        mirca_ha_tot[mirca_ha_tot == 0] = np.nan
        mirca_ha_irc_perc = mirca_ha_irc[crop] / mirca_ha_tot
# Create a mask of rainfed areas (irrigated areas < 10%)
        rfc_boolean = (mirca_ha_irc_perc[:,:,np.newaxis] * irrig_area) < 0.1
# Change cells that are not considered rainfed to zero
        crop_yield_data[crop][rfc_boolean == False] = np.nan
# For some areas first and last year of crop yield data is missing, remove the first and last year
        crop_yield_data[crop] = crop_yield_data[crop][:,:,1:-1]
                
        yield_cntry_df = pd.DataFrame()
        yield_state_df = pd.DataFrame()
        for index, year in enumerate(years, 0):
# Calculate production, harvested areas and yield for each spatial unit:     
            mirca_ha_rfc_temp = copy.deepcopy(mirca_ha_rfc[crop])
            mirca_ha_rfc_temp[rfc_boolean[:,:,index] == False] = np.nan
            
            prod_data_temp = mirca_ha_rfc_temp*crop_yield_data[crop][:,:,index]
            
            mirca_ha_rfc_temp[np.isnan(prod_data_temp)] = 0.0
            prod_data_temp[np.isnan(prod_data_temp)] = 0.0
# Aggregate harvested areas to the spatial units
            ha_cntry_rfc = pd.DataFrame(rs.zonal_stats(vectors=cntry_borders['geometry'], raster = mirca_ha_rfc_temp, affine = affine_info, stats = 'sum', nodata = -999))['sum']
            ha_state_rfc = pd.DataFrame(rs.zonal_stats(vectors=combined_borders['geometry'], raster = mirca_ha_rfc_temp, affine = affine_info, stats = 'sum', nodata = -999))['sum']
# Change harvested areas zeros to nan to avoid dividing by 0 later
            ha_cntry_rfc[ha_cntry_rfc == 0] = np.nan
            ha_state_rfc[ha_state_rfc == 0] = np.nan
# Aggregate "production" (harvested areas (ha) * yield (t/ha) to the spatial units
            prod_cntry = pd.DataFrame(rs.zonal_stats(vectors=cntry_borders['geometry'], raster = prod_data_temp, affine = affine_info, stats = 'sum', nodata = -999))['sum']
            prod_state = pd.DataFrame(rs.zonal_stats(vectors=combined_borders['geometry'], raster = prod_data_temp, affine = affine_info, stats = 'sum', nodata = -999))['sum']
# Calculate yield for each spatial unit
            yield_cntry = prod_cntry/ha_cntry_rfc
            yield_state = prod_state/ha_state_rfc            
# Combine data to a pandas data frame and save data
            yield_cntry_df_temp = pd.DataFrame({'cntry_a3': cntry_names, 'area_name': cntry_names_long  , 'year': np.repeat(year,yield_cntry.shape[0]), 'yield': yield_cntry}, columns = ['cntry_a3','area_name','year','yield'])
            yield_cntry_df = yield_cntry_df.append(yield_cntry_df_temp)
            yield_state_df_temp = pd.DataFrame({'cntry_a3': combined_names_cntry,'cntry_name_long': combined_names_cntry_long, 'area_name': combined_names_area, 'year': np.repeat(year,yield_state.shape[0]), 'yield': yield_state}, columns = ['cntry_a3','cntry_name_long','area_name','year','yield'])
            yield_state_df = yield_state_df.append(yield_state_df_temp)           
        
        os.chdir(r'D:\work\research\greenwater\iizumi_yield_analysis\results\yield_data')
        yield_cntry_df.to_csv('cntry_'+crop+'_rainfed_yield.csv', encoding = 'utf-8', na_rep = 'nan')
        yield_state_df.to_csv('combined_'+crop+'_rainfed_yield.csv', encoding = 'utf-8', na_rep = 'nan')
        
# Calculate percentage of rainfed harvested areas in each spatial unit
        
        mirca_ha_tot = mirca_ha_irc[crop] + mirca_ha_rfc[crop]
        
        ha_cntry_tot = pd.DataFrame(rs.zonal_stats(vectors=cntry_borders['geometry'], raster = mirca_ha_tot, affine = affine_info, stats = 'sum', nodata = -999))['sum']
        ha_state_tot = pd.DataFrame(rs.zonal_stats(vectors=combined_borders['geometry'], raster = mirca_ha_tot, affine = affine_info, stats = 'sum', nodata = -999))['sum']        
        
        ha_cntry_rfc = pd.DataFrame(rs.zonal_stats(vectors=cntry_borders['geometry'], raster = mirca_ha_rfc[crop], affine = affine_info, stats = 'sum', nodata = -999))['sum']
        ha_state_rfc = pd.DataFrame(rs.zonal_stats(vectors=combined_borders['geometry'], raster = mirca_ha_rfc[crop], affine = affine_info, stats = 'sum', nodata = -999))['sum']                
        
        ha_cntry_tot[ha_cntry_tot == 0] = np.nan
        ha_state_tot[ha_state_tot == 0] = np.nan
        
        perc_cntry_rfc = ha_cntry_rfc / ha_cntry_tot
        perc_state_rfc = ha_state_rfc / ha_state_tot
        
        perc_cntry_rfc_df = pd.DataFrame({'cntry_a3': cntry_names, 'area_name': cntry_names_long , 'rfc_perc': perc_cntry_rfc}, columns = ['cntry_a3','area_name','rfc_perc'])
        perc_state_rfc_df = pd.DataFrame({'cntry_a3': combined_names_cntry,'cntry_name_long': combined_names_cntry_long, 'area_name': combined_names_area, 'rfc_perc': perc_state_rfc}, columns = ['cntry_a3','cntry_name_long','area_name','rfc_perc'])
        
        os.chdir(r'D:\work\research\greenwater\iizumi_yield_analysis\results\percentage_of_rainfed_harvested_area')
        perc_cntry_rfc_df.to_csv('cntry_'+crop+'_rfc_perc.csv', encoding = 'utf-8', na_rep = 'nan')
        perc_state_rfc_df.to_csv('combined_'+crop+'_rfc_perc.csv', encoding = 'utf-8', na_rep = 'nan')
        
# Save the shape files about the spatial units used.
    os.chdir(r'D:\work\research\greenwater\iizumi_yield_analysis\results\country_borders')
    cntry_borders.to_file('country_borders.shp')
    os.chdir(r'D:\work\research\greenwater\iizumi_yield_analysis\results\combined_borders')
    combined_borders.to_file('combined_borders.shp')
    





