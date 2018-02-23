# -*- coding: utf-8 -*-qw
"""
Created on  17th of Jan., 2018vd
a
@author: gmiliar (George Ch. Miliaresis)
Cluster analysis/modelling (SVR.CLUSTER) by G.Ch. Miliaresis
Ver. 2017.02 winpython implementation, (https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
"""
from clu_myf import Processing_constants
from clu_myf import data_imv_read
from clu_myf import MainRun
import clu_data_headers


#  1st FUNCTION CALL -------------- Defines tiff import options...
tiff_import_options = Processing_constants()
#  2nd FUNCTION CALL ---------------Selects the data file (header) to work with
[GeoExtent, FigureLabels, LabelLST, LabelLSTxls, Lmin, Lmax,
 LDatadir, Tiffimporttype] = clu_data_headers.dataCL1(tiff_import_options)
#  3rd FUNCTION CALL -------------- IMPORTS the data files, creates the vectors
data, row, col, continue1 = data_imv_read(LDatadir, len(LabelLST),
                                          Tiffimporttype)
#  4th FUNCTION call -------------- Starts the processing of vectors ---------
if continue1 == 'yes':
    MainRun(data, row, col, GeoExtent, FigureLabels, LabelLST, LabelLSTxls,
            Lmin, Lmax)
else:
    print('----> Check the data files')
