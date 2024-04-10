from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import numpy as np

def calc_new_wcs(file, sciext, dx, dy, theta):
    
    '''
    Calculate new WCS header information of fits file.
    
    Parameters
    -----------
    file - str
        File that you need to update wcs of
    sciext - int
        1 or 2 (which science extension are you updating)
    dx - float
        Shift in x pixels
    dy - float
        Shift in y pixels
    theta - float
        Rotation along center of chip (CRPIX1, CRPIX2) in units of radians
        
    Outputs
    -------
    new_refra - float
        New reference RA in units of deg
    new_refdec - float
        New reference DEC in units of deg
    new_pa_deg - float
        New position angle in units of deg
    new_cd1_1 - float
        New CD1_1 value
    new_cd1_2 - float
        New CD1_1 value
    new_cd2_1 - float
        New CD2_1 value
    new_cd2_2 - float
        New CD2_2 value
    '''

    hdu = fits.open(file)

    # Get original header and original WCS information
    hdr = hdu[sciext].header
    print('Original CD values:', hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2'])
    og_cd1_1 = hdr['CD1_1'] # RA per x
    og_cd1_2 = hdr['CD1_2'] # RA per y
    og_cd2_1 = hdr['CD2_1'] # DEC per x
    og_cd2_2 = hdr['CD2_2'] # DEC per y
    og_refra = hdr['CRVAL1'] # Reference RA
    og_refdec = hdr['CRVAL2'] # Reference DEC
    crpix1 = hdr['CRPIX1'] # X reference pixel
    crpix2 = hdr['CRPIX2'] # Y reference pixel

    # Calculate new reference RA and DEC
    wcs = WCS(hdr, hdu) # WCS from image
    # IMPORTANT: Need to specify that the first pixel has coordinates (1,1) and that there is NO
    # distortion correction (mode = 'wcs')
    new_skycoord = pixel_to_skycoord(hdr['CRPIX1']+dx,hdr['CRPIX2']+dy, wcs, origin = 1, mode = 'wcs')
    new_refra = new_skycoord.ra.deg
    new_refdec = new_skycoord.dec.deg
    
#     og_aper_ra = hdr['RA_APER']
#     og_aper_dec = hdr['DEC_APER']
#     ra_diff = new_refra-og_refra
#     dec_diff = new_refdec-og_refdec
#     new_aper_ra = og_aper_ra+ra_diff
#     new_aper_dec = og_aper_dec+dec_diff
    
    # Calculate new position angle
    og_pa_deg = hdr['ORIENTAT']*u.deg # Original position angle
    og_pa_rad = og_pa_deg.to(u.rad)
    # For some reason, one visit has the PA listed as a string??
    if type(hdr['ORIENTAT']) == str:
        og_pa_deg = hdr['ORIENTAT'].replace('deg', '')
        og_pa_deg = float(og_pa_deg)*u.deg
    dpa_rad = theta
    dpa_deg = (theta*u.rad).to(u.deg)
    new_pa_deg = og_pa_deg+dpa_deg
    new_pa_rad = new_pa_deg.to(u.rad)
    print('Old PA (deg, rad):', og_pa_deg, og_pa_rad)
    print('New PA (deg, rad):', new_pa_deg, new_pa_rad)
    
#     og_aper_pa = hdr['PA_APER']
#     new_aper_pa = og_aper_pa*u.deg+dpa_deg

    # Calculate new CD matrix (THETA MUST BE IN RADIANS)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                           [np.sin(theta), np.cos(theta)]])
    (new_cd1_1, new_cd1_2), (new_cd2_1, new_cd2_2) = np.dot(wcs.wcs.cd, rot_matrix)
    
    hdu.close()
    
    return new_refra, new_refdec, new_pa_deg, new_cd1_1, new_cd1_2, new_cd2_1, new_cd2_2

def update_wcs(file, sciext, new_refra, new_refdec, new_pa, new_cd1_1, new_cd1_2, new_cd2_1, new_cd2_2):
    
    '''
    Updates the WCS information of a drc file
    
    '''
    
    print('Updating WCS for {}, science extension = {}'.format(file, sciext))
    
    with fits.open(file, 'update') as f:

        hdu = f[sciext]

        hdr = hdu.header
        
        print('Updating with CD matrix:', new_cd1_1, new_cd1_2, new_cd2_1, new_cd2_2)

        hdr['CRVAL1'] = new_refra
        hdr['CRVAL2'] = new_refdec

        hdr['CD1_1'] = new_cd1_1
        hdr['CD1_2'] = new_cd1_2
        hdr['CD2_1'] = new_cd2_1
        hdr['CD2_2'] = new_cd2_2

        hdr['ORIENTAT'] = new_pa
        
#         hdr['RA_APER'] = new_aper_ra
#         hdr['DEC_APER'] = new_aper_dec
#         hdr['PA_APER'] = new_aper_pa

        # print(hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2'])

        print('Done')