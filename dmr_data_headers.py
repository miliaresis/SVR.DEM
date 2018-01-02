# -*- coding: utf-8 -*-
"""
Created on  20th of December, 2017

@author: gmiliar (George Ch. Miliaresis)
Selective Variance Reduction for DEMs (dim reduction) by G. Ch.Miliaresis
Ver. 2017.02 (winpython implementation, https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
----------------------------------------------------------------
TO LOAD your data, define a header in the file svr_data_headers.py.
-------------------------------------------------------------------------
"""


def phead(xy, ML, row, col, x, x2, x3, Lmn, Lmx, Rmn, Rmx, vfile, LDIR, T, cm):
    """PRINT DATA HEADER.
        DATA files are in a subdir named  data, data2, in the dir where the
        3 scripts are stored.
       The tif image filenames in the data dir, are fixed :
         MASK, and 01, 02, 03 for the 3 DEMs (ALOS, SRTM, ASTER)
         THE NAMES ARE CASE SENSITIVE and they are
         determined automatically from the script (as well as the dimension of
         the feature space -> length of tics list), so you should preserve them
         in your data dir.
    """
    Headers_ALL = ['dataDEM2']
    print('Labels for x-axis, y-axis of images/histograms:\n        ', ML)
    print('Geographic extent of data: ', xy)
    print('AXES legends & Tables headers for rows  & columns',
          '\n   ', x2, '\n   ', x, '\n   ', x3)
    print('Domain of histograms,     LST: ', Lmn, Lmx, ' RLST: ', Rmn, Rmx)
    print('Vectors file: ', vfile)
    print('Subdir with images or vector files= ', LDIR)
    print('Clustering method: ', cm)
    print('Method for TIF file import: ', T)
    print('row=', row, ' col=', col, '    valid with vectors-else overwritten')
    print('\nData headers available: ', Headers_ALL)


def dataDEM2(clustering_options, tiff_import_options):
    """ALOS, SRTM, ASTER GDEMs """
    print('\n---> ALOS, SRTM, ASTER GDEMs, 1 arc sec, Lat/Lon, WGS84, EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.2361, 54.6008, 27.1108, 27.5555]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G']
    x = x2
    x3 = ['ALOS', 'SRTM', 'ASTER']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = 301
    Lmax = 2210
    Rmin = 301
    Rmax = 2210
# clustering method: Kmeans refined by NBG
    clustering_method = clustering_options[1]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# csv vector filename (if used instead of tif images)
    vfile = 'd.csv'
# Sub-directory for image files or vector matrix
    LDIR = 'data'
# Rows & Cols for image reconstruction from vectors (if vector csv is used)
    row = 1601
    col = 1601
    phead(xy, ML, row, col, x, x2, x3, Lmin, Lmax, Rmin, Rmax, vfile, LDIR,
          T, clustering_method)
    return (xy, ML, row, col, x, x2, x3, Lmin, Lmax, Rmin, Rmax, vfile, LDIR,
            T, clustering_method)
