# -*- coding: utf-8 -*-
"""
Created on  17th of Jan., 2018

@author: gmiliar (George Ch. Miliaresis)
Dimensonality reduction for DEMs (SVR.DEM) by G.Ch. Miliaresis
Ver. 2017.02 winpython implementation, (https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
----------------------------------------------------------------
TO LOAD your data, define a header in the file svr_data_headers.py.
-------------------------------------------------------------------------
"""


def phead(xy, ML, x2, x3, Lmn, Lmx, Rmn, Rmx, LDIR, T, cm):
    """PRINT DATA HEADER.
        DATA files stored in a subdir named  data within the dir where the
        3 scripts are stored.
       The tif image filenames (in the data dir) are fixed :
         MASK [0, 1 for data], & 01, 02, 03 for the 3 DEMs (ALOS, SRTM, ASTER)
         THE NAMES ARE CASE SENSITIVE and they are
         determined automatically from the script (as well as the dimension of
         the feature space -> length of tics list), so you should preserve them
         in your data dir.
    """
    Headers_ALL = ['dataDEM2', 'dataDEM1', 'dataDEM3', 'dataDEM4', 'dataDEM5',
                   'dataDEM6']
    print('Labels for x-axis, y-axis of images/histograms:\n        ', ML)
    print('Geographic extent of data: ', xy)
    print('AXES legends & Tables headers for rows  & columns',
          '\n   ', x2, '\n   ', x3)
    print('Domain of histograms, data: ', Lmn, Lmx, ' Rdata: ', Rmn, Rmx, ' m')
    print('Subdir with images or vector files= ', LDIR)
    print('Clustering method: ', cm)
    print('Method for TIF file import: ', T)
    print('\nData headers available: ', Headers_ALL)


def dataDEM1(clustering_options, tiff_import_options):
    """ALOS, SRTM, ASTER GDEMs (3 DEMS), SE Zagros Ranges"""
    print('\n---> ALOS, SRTM, ASTER GDEMs, 1 arc sec, Lat/Lon, WGS84, EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.2362, 54.6810, 27.1107, 27.5555]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G']
    x3 = ['ALOS', 'SRTM', 'ASTER']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = 301
    Lmax = 2210
    Rmin = -25
    Rmax = 25
# clustering method: Kmeans
    clustermethod = clustering_options[0]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'data'
    phead(xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
    return (xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)


def dataDEM2(clustering_options, tiff_import_options):
    """ALOS, SRTM, ASTER, NED GDEMs (4 DEMs) """
    print('\n---> ALOS, SRTM, ASTER, NED GDEMs, 1 arcsec, Lat/Lon,WGS84,EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [-117.2068, -116.7065, 36.0866, 36.5869]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G', 'N']
    x3 = ['ALOS', 'SRTM', 'ASTER', 'NED']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = -156
    Lmax = 2500
    Rmin = -25
    Rmax = 25
# clustering method: Kmeans refined by NBG
    clustermethod = clustering_options[1]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'data2'
    phead(xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
    return (xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)


def dataDEM3(clustering_options, tiff_import_options):
    """ALOS(median,average), SRTM,ASTER DEMS, SE Zagros Ranges"""
    print('\n---> ALOS, SRTM, ASTER GDEMs, 1 arc sec, Lat/Lon, WGS84, EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.16158, 54.69491, 27.03999, 27.57332]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G']
    x3 = ['ALOS', 'SRTM', 'ASTER']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = 205
    Lmax = 2208
    Rmin = -25
    Rmax = 25
# clustering method: Kmeans refined by NBG
    clustermethod = clustering_options[1]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'data3'
    phead(xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
    return (xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)


def dataDEM4(clustering_options, tiff_import_options):
    """ALOS(median,average), SRTM,ASTER DEMS, SE Zagros Ranges, great area"""
    print('\n---> ALOS, SRTM, ASTER GDEMs, 1 arc sec, Lat/Lon, WGS84, EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.17698, 54.95448, 27.02163, 27.71580]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S', 'G']
    x3 = ['ALOS', 'SRTM', 'ASTER']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = 132
    Lmax = 2209
    Rmin = -25
    Rmax = 25
# clustering method: Kmeans refined by NBG
    clustermethod = clustering_options[1]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'data4'
    phead(xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
    return (xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)


def dataDEM5(clustering_options, tiff_import_options):
    """ALOS(median,average), SRTM,ASTER DEMS, SE Zagros Ranges, great area
    only ALOS-median & SRTM are considered <- 2 dimensional test for DATA4"""
    print('\n---> ALOS, SRTM, GDEMs, 1 arc sec, Lat/Lon, WGS84, EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.17698, 54.95448, 27.02163, 27.71580]
# tics for axes of figures and cross-correlation matrix
    x2 = ['A', 'S']
    x3 = ['ALOS', 'SRTM']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = 132
    Lmax = 2209
    Rmin = -25
    Rmax = 25
# clustering method: 0 for Kmeans or  1 for Kmeans refined by NBG
    clustermethod = clustering_options[0]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'data4'
    phead(xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
    return (xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)


def dataDEM6(clustering_options, tiff_import_options):
    """ALOS(median,average),SE Zagros Ranges, great area, only ALOS-median &
       ALOS-average are considered <- 2 dimensional test for DATA4"""
    print('\n---> ALOS median & mean GDEMs, 1 arc sec, Lat/Lon, WGS84, EGM96')
# Main figure labels (title, x-axis, y-axis)
    ML = ['H, m', 'Longitude,DD', 'Latitude, DD']
    # Geograhic extent (X-LON-min, X-LON-max, Y-LAT-min, Y-LAT-max)
    xy = [54.17698, 54.95448, 27.02163, 27.71580]
# tics for axes of figures and cross-correlation matrix
    x2 = ['M', 'A']
    x3 = ['median', 'average']
# Histograms domain for data (eg. DEM) & reconstructed data (eg. DEM)
    Lmin = 132
    Lmax = 2209
    Rmin = -25
    Rmax = 25
# clustering method: 0 for Kmeans or  1 for Kmeans refined by NBG
    clustermethod = clustering_options[1]
# PIL Library is used for TIF file import
    T = tiff_import_options[0]
# Sub-directory for image files or vector matrix
    LDIR = 'data5'
    phead(xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
    return (xy, ML, x2, x3, Lmin, Lmax, Rmin, Rmax, LDIR, T, clustermethod)
