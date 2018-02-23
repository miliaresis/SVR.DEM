# -*- coding: utf-8 -*-
"""
Created on  17th of Jan., 2018

@author: gmiliar (George Ch. Miliaresis)
Analyze/model clusters  (SVR.CLASSES) by G.Ch. Miliaresis
Ver. 2017.02 winpython implementation, (https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
----------------------------------------------------------------
TO LOAD your data, define a header in the file svr_data_headers.py.
-------------------------------------------------------------------------
"""


def phead(xy, ML, x2, x3, Lmn, Lmx, LDIR, T):
    """PRINT DATA HEADER.
        DATA files stored in a subdir named  data within the dir where the
        3 scripts are stored.
       The tif image filenames (in the data dir) are fixed :
         MASK [0 for no data and >1 for cluster classes], & 01, 02, 03 etc. for
         the feature images (eg. residual eleavations ALOS, SRTM, ASTER, NED)
         THE NAMES ARE CASE SENSITIVE and they are
         determined automatically from the script (as well as the dimension of
         the feature space -> length of tics list), so you should preserve them
         in your data dir.
    """
    Headers_ALL = ['dataCL1', 'dataCL4']
    print('Labels for x-axis, y-axis of images/histograms:\n        ', ML)
    print('Geographic extent of data: ', xy)
    print('AXES legends & Tables headers for rows  & columns',
          '\n   ', x2, '\n   ', x3)
    print('Domain of histograms, data: ', Lmn, Lmx, ' m')
    print('Subdir with images or vector files= ', LDIR)
    print('Method for TIF file import: ', T)
    print('\nData headers available: ', Headers_ALL)


def dataCL1(tiff_import_options):
    """ALOS, SRTM, ASTER DEMs: residual elevations """
    print('\n--> ALOS, SRTM, ASTER GDEMs residual H, 1s lat lon WGS84,EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.2362, 54.6810, 27.1107, 27.5555]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G']
    x3 = ['ALOS', 'SRTM', 'ASTER']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = -25
    Lmax = 25
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'dataCL1'
    phead(xy, ML, x2, x3, Lmin, Lmax, LDIR, T)
    return (xy, ML, x2, x3, Lmin, Lmax, LDIR, T)


def dataCL4(tiff_import_options):
    """ALOS, SRTM, ASTER DEMs: residual elevations """
    print('\n--> ALOS, SRTM, ASTER GDEMs residual H, 1s lat lon WGS84,EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [-117.2068, -116.7065, 36.0866, 36.5869]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G']
    x3 = ['ALOS', 'SRTM', 'ASTER']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = -25
    Lmax = 25
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'dataCL4'
    phead(xy, ML, x2, x3, Lmin, Lmax, LDIR, T)
    return (xy, ML, x2, x3, Lmin, Lmax, LDIR, T)
