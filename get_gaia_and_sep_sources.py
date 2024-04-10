
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import sep
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.units import Quantity
import os
from shutil import copyfile
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
        
class get_sources:

    '''
    Class designed to get SEP and Gaia sources for a given drizzled image.

    Parameters
    ----------
    drz_file : str
        File name of drizzled image
    sci_ext : int
        Extension containing the science image
    origin : 0 or 1
        Whether to return 0 or 1-based pixel coordinates. See documentation for the skycoord_to_pixel function:
        https://docs.astropy.org/en/stable/api/astropy.wcs.utils.skycoord_to_pixel.html
    '''
    
    def __init__(self, drz_file, sci_ext = 0, origin = 0):
        
        # print('Initializing ....')
        hdu = fits.open(drz_file) # Read drizzled fits image
        self.hdu = hdu
        self.file = drz_file

        # print('Reading science array...')
        sciarr = hdu[sci_ext].data # Get science data from the first exentions
        self.sciarr = sciarr

        # Get usefule information from headers
        self.drc_ra = hdu[sci_ext].header['CRVAL1'] # Reference RA
        self.drc_dec = hdu[sci_ext].header['CRVAL2'] # Reference DEC
        # self.drc_time = Time(date_obs, format = date_fmt).to_value('jyear') # Get reference epoch 
        # self.drc_time = drc_time

        # print('Reading WCS...')
        ### Get RA and DEC of Objects founnd ###
        drc_wcs = WCS(hdu[sci_ext].header)
        self.drc_wcs = drc_wcs
        
        self.origin = origin
        # print('!!! Origin = {}'.format(origin))
        
        hdu.close()
        
        # print('Initialization complete.')
        # print('\n')

    def mask_borders(self, data_og, border = 10):
        '''
        Function to mask borders of images. Probably doesn't work that well.
        '''

        data = np.copy(data_og)
        
        # Add border that you will remove later
        data = np.pad(data, pad_width=border, mode='constant', constant_values=np.nan)

        # Make array where Nan = False and Number = True
        bool_data = (data == data)

        # Make array that shows where NaN becomes Number or Number becomes NaN
        # True --> nan to num or num to nan
        col_diff = np.diff(bool_data, axis = 0) #cols
        row_diff = np.diff(bool_data, axis = 1) #rows
        
        # Find indices where NaN becomes number using col_diff and  row_diff
        w_col_transition = np.where(col_diff == True)
        w_row_transition = np.where(row_diff == True)
        
        for i in range(1, border+1):
            
            ### Right and Bottom borders ###
            w_col1 = (w_col_transition[0]-(i-1), w_col_transition[1])
            w_row1 = (w_row_transition[0], w_row_transition[1]-(i-1))
            # Set all places where NaN becomes a number to NaN to mask the borders
            # However, this will only mask the bottom or right-most borders becomes w_col and w_row
            # indicate the index BEFORE the transition happens
            data[w_col1] = np.nan
            data[w_row1] = np.nan

            ### Left and top borders ###
            # To get the indices after the transition happens (for masking top and right-most section):
            w_col2 = (w_col_transition[0]+i, w_col_transition[1])
            w_row2 = (w_row_transition[0], w_row_transition[1]+i)

            data[w_col2] = np.nan
            data[w_row2] = np.nan

        # Get rid of border you added at the beginning 
        data = data[border:-border, border:-border]

        # Need to do this so this data works with SEP
        data=np.ascontiguousarray(data)

        return data
        
    def run_sep(self, save_sep, thresh, deblend_cont, flux_limits, mask_border = 10):
        
        '''
        Function to run SEP on drizzled image that the get_sources class is based off of.
        SEP Documentation: https://sep.readthedocs.io/en/stable/api/sep.extract.html#sep.extract

        Parameters
        ----------
        save_sep - str
            Where to save csv containing SEP sources
        thresh - float
            Threshold for SEP (see SEP documentation)
        deblend_cont - float
            deblen_cont for SEP (see SEP documentation)
        flux_limits - tuple
            (Min, Max) values to save objects based on their flux

        Outputs
        -------
        Object dataframe containing SEP sources from drizzled image

        '''

        drc_wcs = self.drc_wcs

        # print('Masking borders...')
        # Mask 10 pixel boreders
        sciarr = self.mask_borders(self.sciarr, border = mask_border)

        # print('Changing values so SEP likes the array...')
        # Need to do this for sextractor to work
        sciarr = sciarr.byteswap().newbyteorder()
        sciarr[sciarr != sciarr] = -99.999
        
        # print('Estimating background...')
        bkg = sep.Background(sciarr)

        # Need to do this so sextractor works
        sep.set_extract_pixstack(10000000)
        sep.set_sub_object_limit(10000)

        # print('Extracting sources...')
        objects = sep.extract(sciarr, thresh = thresh, err=bkg.globalrms, deblend_cont = deblend_cont)
        
        # print('Reading sources into dataframe...')
        object_df = pd.DataFrame(objects)

        # print('Getting RA & DEC of sources...')
        # IMPORTANT: SEP assumes the origin to be (0,0), WHICH DIFFERS FROM SEXTRACTOR!!
        # https://sep.readthedocs.io/en/v1.0.x/reference.html
        wcs_coords = pixel_to_skycoord(object_df['x'], object_df['y'], drc_wcs, origin = self.origin, mode = 'all')
        object_df['ra'] = wcs_coords.ra.deg
        object_df['dec'] = wcs_coords.dec.deg
  
        # print('Doing flux cuts...')
        object_df = object_df.loc[object_df['flux'] > flux_limits[0]] # Only want to brightest sources (so it's easier to pick out the stars)
        object_df = object_df.loc[object_df['flux'] < flux_limits[1]] # SExtractor probably cant correctly odentofy the centroids of oversaturated pixels
        
        
        print('Saving dataframe to {}'.format(save_sep))
        ### SAVE DF ###
        object_df.to_csv(save_sep)

        return object_df


    def run_gaia_query(self, drc_time, save_gaia, gaia_flux_limits, match_drc = False, object_df = None, drop_coord = (0,0), new_ra_col = 'new_ra', new_dec_col = 'new_dec'):

        '''
        Function to query Gaia objects within boundaries of drizzled image that the get_sources class is based off of.

        Paramters
        ---------
        save_gaia - str
            Where to save csv containing gaia sources
        gaia_flux_limits - tuple
            (Min, Max) values to save objects based on their phot_g_mean_flux flux (see Gaia.query_object_async documentation)
        match_drc - bool
            Whether or not to save x and y positions of drc sources in Gaia dataframe and also base the Gaia objects off the bounds of the SEP objects
        object_df - pandas DataFrame
            Dataframe containing sources found in image already using SEP. Only used in match_drc = True
        drop_coord - tuple
            (x,y) coordinates at center of 200 pixel boundary that you want to ignore for source finding
        new_ra_col, new_dec_col - str
            Column names to assign to corrected RA and DEC columns
        '''
        
        drc_ra = self.drc_ra
        drc_dec = self.drc_dec
        # drc_time = self.drc_time
        drc_wcs = self.drc_wcs

        coord = SkyCoord(ra=drc_ra, dec=drc_dec, unit=(u.deg, u.deg))
        radius = Quantity(3, u.arcmin)
        gaia_query = Gaia.query_object_async(coordinate=coord, radius=radius)
        gaia_query = gaia_query.to_pandas()

        # Drop NaNs
        gaia_query = gaia_query.drop(index = np.where(gaia_query['pmra'] != gaia_query['pmra'])[0])
        gaia_query = gaia_query.drop(index = np.where(gaia_query['pmdec'] != gaia_query['pmdec'])[0])

        
        ### Account for proper motions ###

        # Get time of Gaia coords
        gaia_time = Time(gaia_query['ref_epoch'], 
                         format = 'jyear')
        # print('Gaia reference epoch: {}'.format(gaia_time))

        # Difference between drc image and gaia coords
        time_diff_yr = (drc_time-gaia_time.value)

        ra_arr = gaia_query['ra'].to_numpy()
        pmra_arr = gaia_query['pmra'].to_numpy()
        dec_arr = gaia_query['dec'].to_numpy()
        pmdec_arr = gaia_query['pmdec'].to_numpy()

        gaia_query[new_dec_col] = dec_arr*u.deg+pmdec_arr*u.mas*time_diff_yr
        gaia_query[new_ra_col] = ra_arr*u.deg+pmra_arr*u.mas*time_diff_yr/np.cos(np.array(gaia_query[new_dec_col])*u.deg.to(u.rad))


        # Remove sources outside of the field of view
        if match_drc == True:

            ### Get x and y positions of Gaia sources ###
            gaia_coord = SkyCoord(ra=gaia_query[new_ra_col], dec=gaia_query[new_dec_col], unit=(u.deg, u.deg))
            gaia_query['x'] = skycoord_to_pixel(gaia_coord, drc_wcs, mode = 'all', origin = self.origin)[0]
            gaia_query['y'] = skycoord_to_pixel(gaia_coord, drc_wcs, mode = 'all', origin = self.origin)[1]

            print('Only including sources within the field of view...')
            x_cond = (gaia_query['x'] > np.min(object_df['x'])) & (gaia_query['x'] < np.max(object_df['x']))
            y_cond = (gaia_query['y'] > np.min(object_df['y'])) & (gaia_query['y'] < np.max(object_df['y']))
            len_before = len(gaia_query)
            gaia_query = gaia_query.loc[x_cond & y_cond]
            len_after = len(gaia_query)
            print('{} objects removed.'.format(len_before - len_after))

            # Drop x and y positions since we don't know if they correspond to the aligned or unaligned frame
            gaia_query = gaia_query.drop(['x', 'y'], axis = 1)

        gaia_query = gaia_query.loc[gaia_query['phot_g_mean_flux'] > gaia_flux_limits[0]]
        gaia_query = gaia_query.loc[gaia_query['phot_g_mean_flux'] < gaia_flux_limits[1]]
        
        # Drop objects that likely didnt fit well
        if drop_coord != (0,0):
            drop_x_max = drop_coord[0]+200
            drop_x_min = drop_coord[0]-200
            drop_y_max = drop_coord[1]+200
            drop_y_min = drop_coord[1]-200
            drop_gaia_object = gaia_query.loc[(gaia_query['x'] < drop_x_max) & (gaia_query['x'] > drop_x_min) & (gaia_query['y'] < drop_y_max) & (gaia_query['y'] > drop_y_min)]
            gaia_query = gaia_query.drop(index = drop_gaia_object.index)

        gaia_query['ref_epoch'] = drc_time
        gaia_query = gaia_query[['ra', 'dec', 'ref_epoch', 'phot_g_mean_flux', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error']]
        gaia_query.to_csv(save_gaia)

        return gaia_query

    def read_gaia_objects(self, gaia_ra, gaia_dec, match_drc = False, object_df = None, drop_coord = (0,0)):

        '''
        Function to query Gaia objects within boundaries of drizzled image that the get_sources class is based off of.

        Paramters
        ---------
        save_gaia - str
            Where to save csv containing gaia sources
        gaia_flux_limits - tuple
            (Min, Max) values to save objects based on their phot_g_mean_flux flux (see Gaia.query_object_async documentation)
        match_drc - bool
            Whether or not to save x and y positions of drc sources in Gaia dataframe and also base the Gaia objects off the bounds of the SEP objects
        object_df - pandas DataFrame
            Dataframe containing sources found in image already using SEP. Only used in match_drc = True
        drop_coord - tuple
            (x,y) coordinates at center of 200 pixel boundary that you want to ignore for source finding
        '''
        
        drc_ra = self.drc_ra
        drc_dec = self.drc_dec
#         drc_time = self.drc_time
        drc_wcs = self.drc_wcs


        gaia_query = pd.DataFrame({'new_ra': gaia_ra, 'new_dec': gaia_dec})
        
        ### Get x and y positions of Gaia sources ###

        gaia_coord = SkyCoord(ra=gaia_query['new_ra'], dec=gaia_query['new_dec'], unit=(u.deg, u.deg))
#         gaia_query['x'] = drc_wcs.world_to_pixel(gaia_coord)[0]
#         gaia_query['y'] = drc_wcs.world_to_pixel(gaia_coord)[1]
#         refcat_coord = SkyCoord(ra=new_refcat_df[ra_col], dec=new_refcat_df[dec_col], unit=(u.deg, u.deg))
        pixel_coords = skycoord_to_pixel(gaia_coord, drc_wcs, origin = self.origin, mode = 'all')

        gaia_query['x'] = pixel_coords[0]
        gaia_query['y'] = pixel_coords[1]

        if match_drc == True:
            x_cond = (gaia_query['x'] > np.min(object_df['x'])) & (gaia_query['x'] < np.max(object_df['x']))
            y_cond = (gaia_query['y'] > np.min(object_df['y'])) & (gaia_query['y'] < np.max(object_df['y']))
            gaia_query = gaia_query.loc[x_cond & y_cond]
        
        # Drop objects that likely didnt fit well
        if drop_coord != (0,0):
            drop_x_max = drop_coord[0]+200
            drop_x_min = drop_coord[0]-200
            drop_y_max = drop_coord[1]+200
            drop_y_min = drop_coord[1]-200
            drop_gaia_object = gaia_query.loc[(gaia_query['x'] < drop_x_max) & (gaia_query['x'] > drop_x_min) & (gaia_query['y'] < drop_y_max) & (gaia_query['y'] > drop_y_min)]
            gaia_query = gaia_query.drop(index = drop_gaia_object.index)
            
        if match_drc == False:
            gaia_query = gaia_query.drop(columns = ['x', 'y'])

        return gaia_query
        
    def plot_sources(self, object_x, object_y, gaia_x, gaia_y, save = False, savepath = None, show = True, mask_border = 10, show_gaia = False):
        
        sciarr = self.mask_borders(self.sciarr, border = mask_border)
        # sciarr = self.sciarr
        
        fig, ax = plt.subplots(1, figsize = (20,20))

        clipped_data_lower = sigma_clip(sciarr, sigma = 2)
        clipped_data_upper = sigma_clip(sciarr, sigma = 2)
        cb = ax.imshow(sciarr, interpolation='nearest', cmap='Greys', origin='lower', 
                       vmin = -0.006, vmax = 0.009)

        ax.scatter(object_x, object_y, s = 500, facecolors = 'None', edgecolor = 'blue')
        
        if show_gaia == True:
            ax.scatter(gaia_x, gaia_y, color = 'red', s = 200, marker = '*')

        # ax.set_title(os.path.basename(file), fontsize = 20)

        if save == True:
            plt.savefig(savepath)
        
        if show == True:
            plt.show()