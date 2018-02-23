# -*- coding: utf-8 -*-
"""
Created on  17th of Jan., 2018

@author: gmiliar (George Ch. Miliaresis)
Analyze/model clusters(SVR.CLASSES) by G.Ch. Miliaresis
Ver. 2017.02 winpython implementation, (https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
"""
import numpy as np


def Processing_constants():
    """ TIF import options (in function tiff_to_np in dim_myf)
              if PIL  then Image from PIL is used
              if SKITimage  then skimage.io is used """
    print('__________________________________________________________________')
    print('\n --- Cluster visualization & analysis by G. Ch. Miliaresis ---\n')
    print('   Vector data model differs from SVR & SVR.DEM')
    print('      MASK [0=no data, >1 for cluster classes], 01, 02,... for the')
    print('      feature images (eg. residual H for ALOS, SRTM, ASTER)')
    tiff_import_options = ['PIL', 'SKITimage']
    print('   TIFF import options', tiff_import_options)
    print('__________________________________________________________________')
    print('\nDISPLAY ACTIVE DATA HEADER')
    return tiff_import_options


def filenames_of_images(k):
    """ Defines the filenames of images  MASK, 01, 02, 03 .... """
    a = '0'
    Lfiles = ['MASK']
    for i in range(k):
        if i < 9:
            d = a + str(i+1)
        else:
            d = str(i+1)
        Lfiles.append(d)
    return Lfiles


def findcreatenewpath():
    """ Creates a new (non exisiting) path within the data/script-path where
    the output files are stored. The path name is ...\outX where X is
    a number determined automatically by this script
    """
    import os
    oldpath = os.getcwd()
    newpath = oldpath+'\out0'
    i = 0
    while os.path.isdir(newpath) is True:
        i = i + 1
        newpath = oldpath+'\out'+str(i)
    os.makedirs(newpath)
    print('\n Output files path: ', newpath)
    return newpath


def historyfile():
    """ Track (save to file) the user inputs and the file outputs """
    from time import time
    from datetime import date
    f = open('_history.txt', 'w')
    f.write('\n date: ' + str(date.today()) + ' time = ' + str(time()))
    f.write('\n _history.txt tracks user selections & output files')
    return f


def input_screen_int(xstring, xmin, xmax):
    """ input an integer X from screen in the range min<=X<=xmax """
    yy = xstring + ' in [' + str(xmin) + ', ' + str(xmax) + ']: '
    X = xmin-1
    while (X < xmin) or (X > xmax):
        X = int(input(yy))
    return X


def input_screen_str_yn(xstring):
    """ input a string X from screen y, Y, n, N """
    yy = xstring + '(y, Y, n, N) : '
    X = 'y '
    while (X != 'y') and (X != 'Y') and (X != 'n') and (X != 'N'):
        X = input(yy)
    return X


def dummyvar_fcheck():
    """ assign dummy variables if file donot exist (to exit from return var """
    imarray = np.zeros(shape=(3, 3))
    rows = 3
    cols = 3
    continue1 = 'no'
    return imarray, rows, cols, continue1


def data_imv_read(LfilesDIR, featuredimension, T):
    """Main Data FILE (individual images read) """
    print('__________________________________________________________________')
    print('\nIMPORT/READ DATA FILES')
    Lfiles = filenames_of_images(featuredimension)
    LfilesEXTENSION = '.tif'
    print('\nFiles EXTENSION= ', LfilesEXTENSION, 'DIR: ', LfilesDIR, '\n')
    print('FILENAMES: ', Lfiles, ' (names are case sensitive)\n')
    for i in range(len(Lfiles)):
        Lfiles[i] = LfilesDIR + "\\" + Lfiles[i] + LfilesEXTENSION
    data, row, col, continue1 = readimagetiff(Lfiles, T)
    return data, row, col, continue1


def tiff_to_np(filename, T):
    """Read/Import tiff file """
    if T == 'PIL':
        from PIL import Image
        img = Image.open(filename)
        im2 = np.array(img)
        img.close()
    if T == 'SKITimage':
        from skimage.io import imread
        im2 = imread(filename)
    return im2


def readdatafiles0(filename, continue1, T):
    """Read image 2-d tif file &  convert it 1-d to numpy array """
    import os.path
    if continue1 == 'yes':
        if os.path.isfile(filename):
            im2 = tiff_to_np(filename, T)
            imarray = im2.reshape(im2.shape[0] * im2.shape[1])
            print(filename, im2.shape)
            rows = im2.shape[0]
            cols = im2.shape[1]
        else:
            print(filename, ' do not exist')
            imarray, rows, cols, continue1 = dummyvar_fcheck()
    return imarray, rows, cols, continue1


def readdatafiles(filename, rows1, cols1, continue1, T):
    """Read SVR 2-d tif file &  convert it 1-dto numpy array """
    import os.path
    if continue1 == 'yes':
        if os.path.isfile(filename):
            im2 = tiff_to_np(filename, T)
            imarray = im2.reshape(im2.shape[0] * im2.shape[1])
            print(filename, im2.shape)
            if filename == '    ':
                print(' ')
            else:
                if rows1 == im2.shape[0] and cols1 == im2.shape[1]:
                    rows = im2.shape[0]
                    cols = im2.shape[1]
                else:
                    imarray, rows, cols, continue1 = dummyvar_fcheck()
                    print(filename, 'rows, cols differ from others')
        else:
            print(filename, ' do not exist')
            imarray, rows, cols, continue1 = dummyvar_fcheck()
    else:
        imarray, rows, cols, continue1 = dummyvar_fcheck()
    return imarray, rows, cols, continue1


def readimagetiff(Ldatafiles, T):
    """Read individual tiff images - convert data"""
    c1 = 'yes'
    img0, rows, cols, c1 = readdatafiles0(Ldatafiles[0], c1, T)
    img = np.zeros(shape=(img0.shape[0], len(Ldatafiles)))
    img[:, 0] = img0[:]
    rows1 = rows
    cols1 = cols
    for k in range(1, len(Ldatafiles)):
        img1, rows, cols, c1 = readdatafiles(Ldatafiles[k], rows1, cols1,
                                             c1, T)
        img[:, k] = img1
    if c1 == 'yes':
        all_data_elements = img0.sum()
        data = np.zeros(shape=(all_data_elements, len(Ldatafiles)))
        print('\n      Vector data dimensions : ', data.shape)
        m = -1
        for i in range(img0.shape[0]):
            if img0[i] >= 0:
                m = m + 1
                data[m, 0] = img0[i]
                for k in range(1, len(Ldatafiles)):
                    data[m, k] = img[i, k]
    else:
        data = np.zeros(shape=(3, 3))
        rows1 = 0
        cols1 = 0
    return data, rows1, cols1, c1


def findpaths_data2csv(data):
    """find newpath to store outputs, change to newpath data dir """
    newpath = findcreatenewpath()
    import os
    oldpath = os.getcwd()
    os.chdir(newpath)
    f = historyfile()
    f.write("""\n\nDimensionality reduction-DEM Selective Variance Reduction by
                George Ch. Miliaresis (https://about.me/miliaresis)
                Details in https://github.com/miliaresis [Repository SVR.DEM]
                https://sites.google.com/site/miliaresisg/ \n""")
    f.write('\n      Output data files are stored to : ' + newpath + '\n')
    return f, oldpath


def create_data_files(data):
    """ Read data file, create sub-matrices"""
    rows, cols = data.shape
    # Create sub-matrices: IDs, H, LAT, LON & LST
    Ids = np.zeros(shape=(rows, 1))
    Ids[:, 0] = data[:, 0]
    LST = np.zeros(shape=(rows, data.shape[1]-1))
    LST = data[:, 1:data.shape[1]]
    return Ids, LST


def define_cluster_matrices(data, k, f):
    """create cluster sub-matrices, k= the specific cluster id """
    cluster_elements = 0
    for i in range(data.shape[0]):
        if data[i, 0] == k:
            cluster_elements = cluster_elements + 1
    file_xxx = '_descriptive' + str(k) + '.xlsx'
    print('   Cluster: ', k)
    f.write('\n' + file_xxx + ' pixels: ' + str(cluster_elements))
    cluster_matrix = np.zeros(shape=(cluster_elements+1, data.shape[1]))
    m = -1
    for i in range(data.shape[0]):
        if data[i, 0] == k:
            m = m + 1
            for l in range(1, data.shape[1]):
                cluster_matrix[m, l] = data[i, l]
    return cluster_matrix


def test_call(data, LABELmonths3, f):
    """Compute & save to xlsx descriptive statistics for Rdata """
    print('\n Main call that defines cluster matrices for further processing')
    f.write('\n Define cluster matrices for further processing')
    No_of_clusters = data[:, 0].max(axis=0)
    for cluster_id in range(1, int(No_of_clusters)+1):
        datacluster = define_cluster_matrices(data, cluster_id, f)
        data2 = datacluster[:, 1:datacluster.shape[1]]
        print('           ', data2.shape)


def MainRun(data, rows, cols, GeoExtent, FigureLabels, LabelLST, LabelLSTxls,
            Hmin, Hmax):
    """ Main run module of SVR_CLU.py"""
    f, oldpath = findpaths_data2csv(data)
    xyxstr = 'Define cluster matrices ? '
    Display_yesno2 = input_screen_str_yn(xyxstr)
    if Display_yesno2 == 'Y' or Display_yesno2 == 'y':
        f.write('\n DISPLAY:descriptive stats of input data')
        test_call(data, LabelLSTxls, f)
    f.close()
    from os import chdir
    chdir(oldpath)
