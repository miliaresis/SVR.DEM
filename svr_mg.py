# -*- coding: utf-8 -*-
"""
Created on  20th of December, 2017

@author: gmiliar (George Ch. Miliaresis)
Selective Variance Reduction for DEMs (dim reduction) by G.Ch. Miliaresis
Ver. 2017.02 (winpython implementation, https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
"""
from svrmg_myf import Processing_constants
from svrmg_myf import data_imv_read
from svrmg_myf import MainRun
import svr_data_headers


#  1st FUNCTION CALL -------------- Defines clustering & tiff import options...
clustering_options, tiff_import_options = Processing_constants()
#  2nd FUNCTION CALL ---------------Selects the data file (header) to work with
[GeoExtent, FigureLabels, row, col, LabelHLatLonLST, LabelLST, LabelLSTxls,
 Lmin, Lmax, Rmin, Rmax, vectorsfile, LDatadir, Tiffimporttype, cluster_method
 ] = svr_data_headers.dataDEM2(clustering_options, tiff_import_options)
#  3rd FUNCTION CALL -------------- IMPORTS the data files, creates the vectors
data, row, col, continue1 = data_imv_read(row, col, vectorsfile, LDatadir,
                                          len(LabelLST), Tiffimporttype)
#  4th FUNCTION call -------------- Starts the processing of vectors ---------
if continue1 == 'yes':
    MainRun(data, row, col, GeoExtent, FigureLabels, LabelHLatLonLST, LabelLST,
            LabelLSTxls, Lmin, Lmax, Rmin, Rmax, cluster_method,
            clustering_options)
else:
    print('----> Check the data files')
