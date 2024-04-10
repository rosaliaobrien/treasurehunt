import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.stats import sigma_clip
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import match_coordinates_sky
from scipy import stats

from rotate_frame import offset, vect_diff_sum
from scipy.optimize import minimize

# Function to add tick marks to plots
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
def fix_ticks(ax, xmajor, ymajor, xminor, yminor, logscale = False):
    ax.tick_params(right = True, top=True, direction = 'in', which = 'major', length = 7)
    ax.tick_params(right = True, top=True, direction = 'in', which = 'minor', length = 4)
    ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins = ymajor))
    if logscale == False:
        ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins = xmajor))

# Function to convert RA/ DEC to pixel positions
def get_pixel_pos_refcat(file, refcat_df, ext,
                         ra_col = 'ra', dec_col = 'dec', 
                         distort_corr = False, origin = 0):
    
    ### Get x and y positions of Gaia sources ###
    hdu = fits.open(file)
    imagewcs = WCS(hdu[ext].header, hdu)
    
    new_refcat_df = refcat_df.copy()
    
    refcat_coord = SkyCoord(ra=new_refcat_df[ra_col], dec=new_refcat_df[dec_col], unit=(u.deg, u.deg))

    if distort_corr == False:
        pixel_coords = skycoord_to_pixel(refcat_coord, imagewcs, origin = origin, mode = 'wcs')
    if distort_corr == True:
        pixel_coords = skycoord_to_pixel(refcat_coord, imagewcs, origin = origin, mode = 'all')
    
    new_refcat_df['x'] = pixel_coords[0]
    new_refcat_df['y'] = pixel_coords[1]
    
    new_refcat_df = new_refcat_df.reset_index()
    
    return new_refcat_df

class get_offsets:
    
    def __init__(self, file, sciext, refcat_df, image_src_df, plot = True, 
                 refcat_ra_column = 'ra', refcat_dec_column = 'dec', pixel_scale = 0.03, 
                 num_it = 5, distort_corr = False, origin = 1):

        '''
        PARAMETERS
        ----------
        file - str
            File path to fits file
        sciext - int
            Science extension in fits file
        refcat_df - pandas dataframe
            Dataframe containing RA, DEC, X pixel and Y pixel positions of reference catalog
        image_src_df - pandas dataframe
            Dataframe containing RA, DEC, X pixel and Y pixel positions of sources detected in image
        plot - bool
            Whether or not to plot offsets
        refcat_ra_column - str
            Column name to RA coordinates in refcat_df
        refcat_dec_column - str
            Column name to DEC coordinates in refcat_df
        pixel_scale - float
            Pixel scale of image
        num_it - int
            Number of iterations to perform when removing outliers during fitting
        distort_corr - bool
            Whether or not image has been distortion corrected
        origin - int (1 or 0)
            Whether coordinates assume the origin to be (1,1) or (0,0)
        '''
        
        self.num_it = num_it
        
        self.file = file
        self.sciext = sciext
        self.plot = plot
        # self.bad_match_dist = bad_match_dist
        # self.refcat_ra_column = refcat_ra_column
        # self.refcat_dec_column = refcat_dec_column
        self.pixel_scale = pixel_scale
        
        # Read in image data
        hdu = fits.open(file)
        self.hdu = hdu
        sciarr = hdu[sciext].data
        sciarr = sciarr.byteswap().newbyteorder()
        self.sciarr = sciarr

        # Get WCS from image
        # Only do first science extension because the shifts and rotations should be the same for both chips
        file_wcs = WCS(hdu[sciext].header, hdu)
        self.file_wcs = file_wcs
        
        # Make SkyCoord object from reference catalog and get X and Y positions
        refcat_coord = SkyCoord(ra=refcat_df[refcat_ra_column].to_numpy(), dec=refcat_df[refcat_dec_column].to_numpy(), unit=(u.deg, u.deg))
        self.refcat_coord = refcat_coord
        self.refcat = get_pixel_pos_refcat(file, refcat_df, sciext,
                                           ra_col = refcat_ra_column, dec_col = refcat_dec_column, 
                                           distort_corr = distort_corr, origin = origin)
        
        # Get image source catalog that was generated on the image that isn't aligned to anything yet
        # Also use RA and DEC from this catalog to get X and Y positions
#         self.object_df = get_pixel_pos_refcat(file, image_src_df, sciext,
#                                               ra_col = 'ra', dec_col = 'dec', 
#                                               distort_corr = distort_corr, origin = origin)
        self.object_df = image_src_df

        # Whether or not image is distortion corrected
        self.distort_corr = distort_corr
        
        self.origin = origin
        print('!!! Origin = {}'.format(origin))
        
    # Find matches between the reference catalog and sources detected in the image
    def find_matches(self):

        object_df = self.object_df
        file_wcs = self.file_wcs

        # Get all reference catalog coordinates
        refcat_coord = self.refcat_coord

        # Coordinates of SEP objects based on image
        im_obj_coord = SkyCoord(ra=object_df['ra'].to_numpy()*u.degree, dec=object_df['dec'].to_numpy()*u.degree)        
        # Find likely matches
        # match_coordinates_sky(refcat_coord, im_obj_coord)
        # idx = indices into im_obj_coord to get matched points for refcat_coord
        idx, d2d, d3d = match_coordinates_sky(refcat_coord, im_obj_coord)

        # Get the pixel coordinates of the matches
        if self.distort_corr == False:
            matches = skycoord_to_pixel(im_obj_coord[idx], wcs = file_wcs, origin = self.origin, mode = 'wcs')
        if self.distort_corr == True:
            matches = skycoord_to_pixel(im_obj_coord[idx], wcs = file_wcs, origin = self.origin, mode = 'all')
        self.matches = matches

        print('{} matches found.'.format(len(matches[0])))
    
    # Drop a single bad object based on it's index (aka row number)
    # This function is read intto drop_bad_object_all
    def drop_bad_object(self, refcat_x, refcat_y, image_x, image_y, bad_idx):

        # Bad x and y positions
        bad_x_pos = image_x[bad_idx]
        bad_y_pos = image_y[bad_idx]

        # Drop x and y positions of bad object from image source list
        deleted_x_arr = np.delete(image_x, bad_idx)
        deleted_y_arr = np.delete(image_y, bad_idx)

        # Drop row of bad object from reference catalog (x and y index should be the same, so it doesnt matter which one you use)
        self.refcat = self.refcat.drop(index = bad_idx)
        self.matches = np.append([np.delete(self.matches[0], bad_idx)],[np.delete(self.matches[1], bad_idx)], axis = 0)

        return deleted_x_arr, deleted_y_arr
    
    # Drop all bad objects, where bad objects are defined to be those where the offset is greater than 100 pixels
    def drop_bad_object_all(self, refcat_x, refcat_y, image_x, image_y, bad_match_dist):

        # Offsets compared to reference catalog
        x_offsets = np.array(refcat_x - image_x)
        y_offsets = np.array(refcat_y - image_y)
        offsets = np.sqrt(x_offsets**2 + y_offsets**2)

        # Identify bad matches based on offsets that are greater than X pixels apart
        bad_idx = np.where(np.abs(offsets)>bad_match_dist)[0]

        self.dropped_object = False

        # If there aree bad objects detecteed, drop them
        if (len(bad_idx) > 0):
            
            print('{} bad objects detected! (Offset by more than {} pixels)'.format(len(bad_idx), bad_match_dist))
            
            self.dropped_object = True

            dropped_x, dropped_y = self.drop_bad_object(refcat_x, refcat_y, image_x, image_y, bad_idx)

            return dropped_x, dropped_y

        else:

            return image_x, image_y

            
    # Use scripts you made plus scipy.minimize to minimize the sum of the distances and find the shifts
    def find_shifts(self):
        
        # refcat = self.refcat
        matches = self.matches
        
        hdr = self.hdu[self.sciext].header
        crpix1 = hdr['CRPIX1']
        crpix2 = hdr['CRPIX2']

        ### NEW ###
        print('Iterating minimization...')

        # Find an initial solution using reference catalog positions and positions measured from image
        sol = minimize(vect_diff_sum, x0 = [0,0,0], args = (self.refcat['x'], self.refcat['y'], 
                                                            matches[0], matches[1],
                                                            crpix1,crpix2))
        dx,dy,theta = sol.x

        # Calculate the new x and y positions after correction has been applied
        final_x_1, final_y_1 = offset(matches[0],matches[1],dx,dy,theta,crpix1,crpix2)

        # Iterate to get rid of outliers
        # This iteration is only used to best constrain the sample! The final dx, dy and theta are calculated after
        # using the updated sample
        bad_match_dist_list = [10,7,5,2,1,0.7,0.5] # list of pixel distances to get rid of outliers in each iteration
        for i in range(0,self.num_it):

            sol = minimize(vect_diff_sum, x0 = [0,0,0], args = (self.refcat['x'], self.refcat['y'], 
                                                                                  final_x_1, final_y_1, crpix1,crpix2))
            dx_i,dy_i,theta_i = sol.x
            final_x_2, final_y_2 = offset(final_x_1,final_y_1,dx_i,dy_i,theta_i,crpix1,crpix2)
            
            # Calculate the median absolute deviation of this iteration
            temp_mad_x = stats.median_abs_deviation(final_x_2 - self.refcat['x'])
            temp_mad_y = stats.median_abs_deviation(final_y_2 - self.refcat['y'])

            # Drop objects that are offset by some amount (defined by bad_match_dist)
            # Reset index so that when you drop indices later, it corresponds the correct objects
            # NOTE: drop_bad_object_all also updates the self.matches object, but you decided it would be easier to also
            # return the updated image coordinates
            self.refcat = self.refcat.reset_index(drop = True)
            dropped_final_matches = self.drop_bad_object_all(self.refcat['x'], self.refcat['y'], 
                                                             final_x_2, final_y_2,
                                                             bad_match_dist = bad_match_dist_list[i])

            # The x and y coordinates to be used foor the next iteration 
            final_x_1 = dropped_final_matches[0]
            final_y_1 = dropped_final_matches[1]

            # Update dx, dy and theta values so that final value is the total shift + rotation
            # dx += dx_i 
            # dy += dy_i 
            # theta += theta_i

        # Find FINAL ALIGNMENT USING CONSTRIANED SAMPLE!!
        sol = minimize(vect_diff_sum, x0 = [0,0,0], args = (self.refcat['x'], self.refcat['y'], 
                                                            self.matches[0], self.matches[1],
                                                            crpix1,crpix2))
        dx,dy,theta = sol.x
        final_x, final_y = offset(self.matches[0],self.matches[1],dx,dy,theta,crpix1,crpix2)
            
        # final_x = final_x_1
        # final_y = final_y_1
        ######
        
        self.final_x = final_x
        self.final_y = final_y
        self.dx = dx
        self.dy = dy
        self.theta = theta

    # Get offsets of objects in image to their corresponding reference catalog positions
    def get_offsets_each(self, refcat_x, refcat_y, image_x, image_y):

        # Image objects (usually identified with sextractor or sep) offsets from reference catalog positions
        norm_sext_x = image_x-refcat_x
        norm_sext_y = image_y-refcat_y

        # print('x median not clippeed:', np.median(norm_sext_x))

        # Get the median offsets, mad of the offset, and the std of the offset
        clipped_x = sigma_clip(norm_sext_x, sigma = 5)
        clipped_y = sigma_clip(norm_sext_y, sigma = 5)
        
        x_median = np.ma.median(clipped_x)
        y_median = np.ma.median(clipped_y)
        
        x_mad = stats.median_abs_deviation(clipped_x)
        y_mad = stats.median_abs_deviation(clipped_y)
        
        x_std = np.ma.std(clipped_x)
        y_std = np.ma.std(clipped_y)

        # print('x median clipped:', x_median)
        
        return norm_sext_x, norm_sext_y, x_median, y_median, x_mad, y_mad, x_std, y_std
                
    # Function to help with plotting results
    def plot_offsets_subplot(self, ax, refcat_x, refcat_y, image_x, image_y, 
                          tick_params = [5,5,2,2], title = 'Offets from Reference Catalog', 
                          limits = [-10,10,-10,10], text_pos = [4,1]):

        norm_sext_x, norm_sext_y, x_median, y_median, x_mad, y_mad, x_std, y_std = self.get_offsets_each(refcat_x, refcat_y, image_x, image_y)

        ax.scatter(norm_sext_x, norm_sext_y, s = 100, color = 'blue', marker = '+')

        ax.set_xlabel('Relative X offset (arcsec)', fontsize = 15)
        ax.set_ylabel('Relative Y offset (arcsec)', fontsize = 15)
#         ax.legend(fontsize = 15)
        ax.tick_params('both', labelsize = 13)
        ax.axvline(x = 0, color = 'black', alpha = 0.2)
        ax.axhline(y = 0, color = 'black', alpha = 0.2)

        ax.axvline(x = x_median, color = 'red', alpha = 0.5)
        ax.axhline(y = y_median, color = 'red', alpha = 0.5)
        
        ax.axvline(x = x_median+x_mad, color = 'black', alpha = 0.2, linestyle = '--')
        ax.axhline(y = y_median+y_mad, color = 'black', alpha = 0.2, linestyle = '--')
        ax.axvline(x = x_median-x_mad, color = 'black', alpha = 0.2, linestyle = '--')
        ax.axhline(y = y_median-y_mad, color = 'black', alpha = 0.2, linestyle = '--')

        ax.text(text_pos[0],text_pos[1],'3$\sigma$-clipped Median X: {:.3f}'.format(x_median), 
                fontsize = 10, color = 'red')
        ax.text(text_pos[0],text_pos[1]-0.05,'3$\sigma$-clipped Median Y: {:.3f}'.format(y_median), 
                fontsize = 10, color = 'red')
        ax.text(text_pos[0],text_pos[1]-0.10,'3$\sigma$-clipped Med Abs Dev X: {:.3f}'.format(x_mad), 
                fontsize = 10, color = 'red')
        ax.text(text_pos[0],text_pos[1]-0.15,'3$\sigma$-clipped Med Abs Dev Y: {:.3f}'.format(y_mad), 
                fontsize = 10, color = 'red')

        ax.set_title(title, fontsize = 18)

        ax.set_xlim([limits[0], limits[1]])
        ax.set_ylim([limits[2], limits[3]])

        fix_ticks(ax,tick_params[0],tick_params[1],tick_params[2],tick_params[3])
        
        # return norm_sext_x, norm_sext_y, x_median, y_median, x_mad, y_mad, x_std, y_std
    
    # Plot offsets before and after correction
    def plot_offsets(self):
    
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,7))

#         limits = [-0.2,0.2,-0.2,0.2] # axis limits of plots
        tick_params = [6,6,2,2] # tick params for fix_ticks
        text_pos = [0.03,-0.15] # position of text showing the median and mad
        
        refcat_x_arcsec = self.refcat['x']*self.pixel_scale
        refcat_y_arcsec = self.refcat['y']*self.pixel_scale
        image_x_arcsec_og = self.matches[0]*self.pixel_scale
        image_y_arcsec_og = self.matches[1]*self.pixel_scale
        final_x_arcsec = self.final_x*self.pixel_scale
        final_y_arcsec = self.final_y*self.pixel_scale
        
        # Make plot before shift is made
        print('Making unaligned plot...')
        # print('Average x diff:', np.median(image_x_arcsec_og-refcat_x_arcsec))
        self.plot_offsets_subplot(ax1,
                                  refcat_x_arcsec,refcat_y_arcsec,image_x_arcsec_og,image_y_arcsec_og,
                                  tick_params = tick_params, title = 'Not Corrected', 
                                  limits = [-0.5,0.5,-0.5,0.5], text_pos = text_pos)
        
        print('Making aligned plot...')
        # Make plot showing offsets after shift is made
        self.plot_offsets_subplot(ax2,
                                  refcat_x_arcsec,refcat_y_arcsec,final_x_arcsec,final_y_arcsec,
                                  tick_params = tick_params, title = 'Corrected', 
                                  limits = [-0.5,0.5,-0.5,0.5], text_pos = text_pos)

        fig.suptitle(self.file, fontsize = 22, y = 1.03)
        
        plt.show()
    
    # Plot results where the x or y axis shows the true x or y position
    def plot_offsets_subplot_xy(self, ax, refcat_x, refcat_y, image_x, image_y, 
                          tick_params = [5,5,2,2], title = 'Offets from Reference Catalog', 
                          limits = [0,4096,0,2048], main_axis = 'x'):
        
        # Image objects (usually identified with sextractor or sep) offsets from ref cat positions
        norm_sext_x = image_x-refcat_x
        norm_sext_y = image_y-refcat_y

        if main_axis == 'x':
            ax.scatter(image_x, norm_sext_y*self.pixel_scale, s = 100, color = 'blue', marker = '+')
            ax.set_xlabel('X offset (pixels)', fontsize = 15)
            ax.set_ylabel('Relative Y offset (arcsec)', fontsize = 15)
            
        if main_axis == 'y':
            ax.scatter(norm_sext_x*self.pixel_scale, image_y, s = 100, color = 'blue', marker = '+')
            ax.set_xlabel('Relative X offset (arcsec)', fontsize = 15)
            ax.set_ylabel('Y offset (pixels)', fontsize = 15)

#         ax.legend(fontsize = 15)
        ax.tick_params('both', labelsize = 13)
        ax.axvline(x = 0, color = 'black', alpha = 0.2)
        ax.axhline(y = 0, color = 'black', alpha = 0.2)

        ax.set_title(title, fontsize = 18)

        ax.set_xlim([limits[0], limits[1]])
        ax.set_ylim([limits[2], limits[3]])

        fix_ticks(ax,tick_params[0],tick_params[1],tick_params[2],tick_params[3])
        
    # Plot offsets where x (or y) axis shows true position
    def plot_offsets_xy(self, main_axis = 'x'):
        
        refcat = self.refcat
        matches = self.matches
        final_x = self.final_x
        final_y = self.final_y
    
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,7))

        size_y, size_x = np.shape(self.sciarr)

        if main_axis == 'x':
            limits = [0,size_x,-0.5,0.5]
        if main_axis == 'y':
            limits = [-0.5,0.5,0,size_y]

        tick_params = [5,5,2,2]
        self.plot_offsets_subplot_xy(ax1, refcat['x'],refcat['y'],matches[0], matches[1],
                          tick_params = tick_params, title = 'Not Corrected', limits = limits, main_axis = main_axis)
        self.plot_offsets_subplot_xy(ax2, refcat['x'],refcat['y'],final_x,final_y,
                          tick_params = tick_params, title = 'Corrected', limits = limits, main_axis = main_axis)

        fig.suptitle(self.file, fontsize = 22, y = 1.03)
        
        plt.show()
        
    def main(self):
        
        self.find_matches()
        self.drop_bad_object_all(self.refcat['x'], self.refcat['y'],
                                 self.matches[0], self.matches[1], 
                                 bad_match_dist = 100)
        print('Number of matches = {}'.format(len(self.matches[0])))

        self.find_shifts()

        # Get the final offsets after correction will be applied
        self.offset_outputs = self.get_offsets_each(self.refcat['x']*self.pixel_scale, self.refcat['y']*self.pixel_scale, 
                                                    self.final_x*self.pixel_scale, self.final_y*self.pixel_scale)
        self.norm_sext_x, self.norm_sext_y, self.final_x_median, self.final_y_median, self.final_x_mad, self.final_y_mad, self.final_x_std, self.final_y_std = self.offset_outputs
        
        print('DX (arcsec): ', self.dx*self.pixel_scale)
        print('DY (arcsec): ', self.dy*self.pixel_scale)
        
        if self.plot == True:
            self.plot_offsets()
            self.plot_offsets_xy(main_axis = 'x')
            self.plot_offsets_xy(main_axis = 'y')
            
        outputs = {'dx': self.dx, 'dy': self.dy, 'theta': self.theta,
                   'corr_median_x': self.final_x_median, 'corr_median_y': self.final_y_median, 
                   'corr_std_x': self.final_x_std, 'corr_std_y': self.final_y_std,
                   'corr_mad_x': self.final_x_mad, 'corr_mad_y': self.final_y_mad,
                   'dropped_object': self.dropped_object, 
                   'num_match': len(self.matches[0])}
        
        # Return 1) information on fit, 2) position of objects in mosaic, 3) reference catalog positions, 4) X offsets (image x - refcat x) and 5) y offsets (image y - refcat y)
        return outputs, self.matches, self.refcat
    