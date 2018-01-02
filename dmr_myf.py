# -*- coding: utf-8 -*-
"""
Created on  20th of December, 2017

@author: gmiliar (George Ch. Miliaresis)
Selective Variance Reduction for DEMs (DEM dimentional reduction)
       by G.Ch. Miliaresis
Ver. 2017.02 (winpython implementation, https://winpython.github.io/)
Details in https://github.com/miliaresis
           https://sites.google.com/site/miliaresisg/
"""
import numpy as np


def program_constants():
    """ program constants (you might increase them according to your needs)
    I = Maximum possible iterations (Clusters) for inertia computation as well
        as for BIC score for full covariance (GMM) computation
    maxC = Maximum number of clusters
    maxNBG =  Maximum number of NBG refinements """
    I = 29
    maxC = 100
    maxNBG = 5
    return I, maxC, maxNBG


def Processing_constants():
    """ Alternative clustering & tif import options.
            There are various options for TIFF import. The methods included are
        available in the default library available in WinPython. See function
        tiff_to_np in svrmg_myf for the specific calls.
        If    PIL  then Image from PIL is used
              SKITimage  then skimage.io is used
        The problems encountered has to do with the files format. For example
             Some libraries "do not like"" the 1 bit or even 1/2 bytes
             integers MASK image. Some others library "do not like" many
             bytes per pixel, or even signed real values (FLOAT).
        THE PROBLEM is solved with SKIimage.io that allows float tif matrix
        import but all the files should include matrices (pixels) that are of
        FLOAT type. This is valid even for the mask image that is actually a
        0/1 matrix. If your mask image pixel depth is 1-bit, or 1 byte or 2
        byte integers instead of float, data files will not be imported if you
        use SKIimage.io.
        PIL is used for LST images since due to data value range (LAT,LON, LST,
        H) are handled ok by PIL. In this case, you do not have to convert Mask
        image to float. PIL might not be used for DayMET data due to the value
        range of X and Y [in a newer version of these libraries, this situation
        might be changed].
        CLUSTERING - CLASSIFICATION OPTIONS:
        These are the clustering options:
            Kmeans -> K-means clastering
            Kmeans clustering refined by Naive Bayes Gaussian classification
            etc., etc.
    """
    print('__________________________________________________________________')
    print('\n --- DEM SVR by G. Ch. Miliaresis ---\n')
    tiff_import_options = ['PIL', 'SKITimage']
    clustering_options = ['Kmeans', 'Kmeans refined by NBG']
    cluster_assessment = ['Inertia', 'BIC (GMM) scores']
    print('Processing options: \n  TIFF import options', tiff_import_options,
          '\n  Clustering options', clustering_options, '\n  Cluster assess ',
          cluster_assessment)
    print('__________________________________________________________________')
    print('\nDISPLAY ACTIVE DATA HEADER')
    return clustering_options, tiff_import_options


def filenames_of_images(k):
    """ Defines the filenames of images  MASK, 01, 02, 03    """
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
    the output files are stored. The path name is .......\outX where X is
    a number determined automatically by the this script
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


def data_imv_read(row, col, vectordfile, LfilesDIR, featuredimension, T):
    """Main Data FILE (individual images or vector file read) """
    print('__________________________________________________________________')
    print('\nIMPORT/READ DATA FILES')
    Lfiles = filenames_of_images(featuredimension)
    LfilesEXTENSION = '.tif'
    print('\nFiles EXTENSION= ', LfilesEXTENSION, 'DIR: ', LfilesDIR, '\n')
    print('FILENAMES: ', Lfiles, ' (names are case sensitive)\n')
#   print('            xxxxx: ', len(Lfiles))
    for i in range(len(Lfiles)):
        Lfiles[i] = LfilesDIR + "\\" + Lfiles[i] + LfilesEXTENSION
    data, row, col, continue1 = readimagetiff(Lfiles, T)
    return data, row, col, continue1


def tiff_to_np(filename, T):
    """Read/Import tiff file - various options are tested """
    # option 1
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
    """Read SVR 2-d tif file &  convert it 1-d to numpy array """
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
    """"Read individual tiff images - convert data"""
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
            if img0[i] > 0:
                m = m + 1
                data[m, 0] = i+1
                for k in range(1, len(Ldatafiles)):
                    data[m, k] = img[i, k]
    else:
        data = np.zeros(shape=(3, 3))
        rows1 = 0
        cols1 = 0
    return data, rows1, cols1, c1


def findpaths_data2csv(data):
    """find-define newpath to store the outputs, change to newpath data dir &
       Write vector data matrix to a csv file within the newpath dir """
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
    wcsv = input_screen_str_yn('      Write vector data to csv, 4 decimals ? ')
    if wcsv == 'Y' or wcsv == 'y':
        newf = 'data.csv'
        print('\n               Saved vector data (4 decimals) to:', newf)
        f.write('\n WRITE/SAVE data (vector matrix 4 decimals) to:' + newf)
        np.savetxt(newf, data, fmt='%.4f', delimiter=',')
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


def standardize_matrix2(A):
    """standardize a 2-d matrix per columns"""
    B = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    return B


def crosscorrelate(LST):
    """ compute the crosscorrelation matrix"""
    LST2 = standardize_matrix2(LST)
    crosscorrelation = LST2.T.dot(LST2)/(LST2.shape[0]-1)
    return crosscorrelation


def translatebymean(LST):
    """ Translate a matrix by mean (per columns)"""
    LSTMEAN = LST.mean(axis=0)
    LST2 = LST - LSTMEAN.T
    return LST2


def retranslatebymean(LST, RLST):
    """ RETranslate a matrix by mean vector (per columns)"""
    LSTMEAN = LST.mean(axis=0)
    RLST = RLST + LSTMEAN.T
    return RLST


def covariance_matrix(LST2):
    """ Compoute variance-covariance matrix"""
    covmat = LST2.T.dot(LST2)/(LST2.shape[0]-1)
    return covmat


def savepcamatrices_csv(evs_per, crosscorrelation, covmat, evs, evmat):
    """ save PCA matrices to CSV files"""
    print('         PC,    %   , eigenvalue')
    for i in range(covmat.shape[1]):
        print('        %2d : %6.3f , %7.3f ' % (i+1, evs_per[i], evs[i]))
    np.savetxt('PCAcrosscor.csv', crosscorrelation, fmt='%.2f', delimiter=',')
    np.savetxt('PCAeigenvalue_percent.csv', evs_per, fmt='%.3f', delimiter=',')
    np.savetxt('PCAcovariance.csv', covmat, fmt='%.1f', delimiter=',')
    np.savetxt('PCAeigenvalues.csv', evs, fmt='%.4f', delimiter=',')
    np.savetxt('PCAeigenvector.csv', evmat, fmt='%.5f', delimiter=',')


def sortdescent(evs, evmat):
    """sort eigenvalues-eigenvectors in descenting eigenvalue magnitude """
    i = np.argsort(evs)[::-1]
    evs = evs[i]
    evmat = evmat[:, i]
    evs_percent = np.zeros(shape=(evs.shape[0]))
    evs_percent = (100 * evs / np.sum(evs))
    return evs, evmat, evs_percent


def pcanew(LST):
    """ compute eigevalues, & eigenvectors"""
    from scipy import linalg
    LST2 = translatebymean(LST)
    covmat = covariance_matrix(LST2)
    evs, evmat = linalg.eig(covmat)
    evs = np.real(evs)
    evmat = np.real(evmat)
    evs, evmat, evs_percent = sortdescent(evs, evmat)
    return evs_percent, covmat, evs, evmat


def Reconstruct_matrix(evmat, LST):
    """ Inverse transform keep pc-1 only """
    X = np.zeros(shape=(evmat.shape[0], 1))
    X[:, 0] = evmat[:, 0]
    Y = X.T
    Z = np.dot(X, Y)
    Reconstruct = np.dot(LST, Z)
    return Reconstruct


def Reconstruct_matrix2(evmat, LST):
    """ Inverse transform keep pc2 & pc3 only """
    X = np.zeros(shape=(evmat.shape[0], 2))
    X[:, :] = evmat[:, 1:2]
    Y = X.T
    Z = np.dot(X, Y)
    Reconstruct = np.dot(LST, Z)
    return Reconstruct


def xlspca(data, data1, data2, data3, x):
    """ write correlation matrix, eigen-vectors/values to xls file"""
    import xlsxwriter
    print('Create pca.xlsx')
    workbook = xlsxwriter.Workbook('_pca.xlsx')
    worksheet1 = workbook.add_worksheet()
    print('   write cross correlation matrix')
    worksheet1.write(1, 0, 'Cross Correlation')
    worksheet1.name = 'Cross_correlation'
    for i in range(0, data.shape[0]):
        worksheet1.write(1, i+2, x[i])
        worksheet1.write(i+2, 1, x[i])
        for j in range(0, data.shape[1]):
            worksheet1.write(i+2, j+2, str(round(data[i, j], 4)))
    worksheet2 = workbook.add_worksheet()
    worksheet2.name = 'Eigenvectors'
    print('   write eigenvalues & eigenvectors')
    for i in range(0, data1.shape[0]):
        worksheet2.write(1, i+2, 'PC'+str(i+1))
        worksheet2.write(i+2, 1, 'Eigenvector '+str(i+1))
        for j in range(0, data1.shape[1]):
            worksheet2.write(i+2, j+2, data1[i, j])
        worksheet2.write(data1.shape[0]+2, i+2, data2[i])
        worksheet2.write(data1.shape[0]+3, i+2, data3[i])
    worksheet2.write(data1.shape[0]+2, 1, 'EIGENVALUE')
    worksheet2.write(data1.shape[0]+3, 1, 'Variance %')
    workbook.close()


def ImplementSVR_MG(data, Labelmonth1, f):
    """main calls to SVR_MG """
    print('__________________________________________________________________')
    Ids, LST = create_data_files(data)
    print('\nSVR IMPLEMENTATION')
    f.write('\nSVR IMPLEMENTATION')
    data2 = data[:, 1:data.shape[1]]
    crosscorrelation = crosscorrelate(data2)
    f.write('\n    Compute cross correlation matrix')
    evs_percent, covmat, evs, evmat = pcanew(LST)
    f.write('\n    Compute eigenvalues & eigenvectors')
    xlspca(crosscorrelation, evmat, evs, evs_percent, Labelmonth1)
    f.write('\n    Write xlsx file: pca.xlsx')
    xyxstr = 'reconstruct from PC1 (yes) else from PC2 & PC3 (no)? '
    Display_yesno2 = input_screen_str_yn(xyxstr)
    if Display_yesno2 == 'Y' or Display_yesno2 == 'y':
        Reconstruct = Reconstruct_matrix(evmat, LST)
    else:
        Reconstruct = Reconstruct_matrix2(evmat, LST)
    return Reconstruct


def prnxls_confuse(workbook, data2):
    """Add confusion matrix to an xls sheet within a workbook """
    all_elements = data2.sum()
    all_correct = sum(data2[i][i] for i in range(0, data2.shape[1]))
    reclassified = (1 - all_correct / all_elements) * 100
    worksheet3 = workbook.add_worksheet()
    worksheet3.name = 'Confusion_matrix'
    worksheet3.write(0, 0, 'Confusion Matrix')
    worksheet3.write(data2.shape[1]+2, 0, 'Correct')
    worksheet3.write(data2.shape[1]+2, 1, all_correct)
    worksheet3.write(data2.shape[1]+3, 0, 'out of')
    worksheet3.write(data2.shape[1]+3, 1, all_elements)
    worksheet3.write(data2.shape[1]+4, 0, '% reclassified')
    worksheet3.write(data2.shape[1]+4, 1, reclassified)
    for i in range(0, data2.shape[1]):
        worksheet3.write(1, i+2, 'B-' + str(i+1))
        for j in range(0, data2.shape[0]):
            worksheet3.write(j+2, i+2, data2[j, i])
    for i in range(0, data2.shape[0]):
        worksheet3.write(i+2, 1, 'A-' + str(i+1))
    return all_elements, all_correct, reclassified


def prn_xls_centroids(workbook, Centroids, LabelLST):
    """ write Centroids matrix to a sheet of an excel workbook"""
    worksheet1 = workbook.add_worksheet()
    worksheet1.name = 'Centroids'
    worksheet1.write(0, 0, 'Cluster centers')
    for i in range(0, Centroids.shape[1]):
        worksheet1.write(1, i+2, LabelLST[i])
        for j in range(0, Centroids.shape[0]):
            worksheet1.write(j+2, i+2, Centroids[j, i])
    for i in range(0, Centroids.shape[0]):
        worksheet1.write(i+2, 1, 'cluster ' + str(i+1))


def prn_xls_sigma(workbook, sigma, LabelLST):
    """ write Sigma matrix to a sheet of an excel workbook"""
    worksheet2 = workbook.add_worksheet()
    worksheet2.name = 'Centroid_variance'
    worksheet2.write(0, 0, 'Centroids variance')
    for i in range(0, sigma.shape[1]):
        worksheet2.write(1, i+2, LabelLST[i])
        for j in range(0, sigma.shape[0]):
            worksheet2.write(j+2, i+2, sigma[j, i])
    for i in range(0, sigma.shape[0]):
        worksheet2.write(i+2, 1, 'cluster ' + str(i+1))


def prn_xls_divergence(workbook, Diverg):
    """ write Divergence matrix to a sheet of an excel workbook"""
    worksheet4 = workbook.add_worksheet()
    worksheet4.name = 'Divergence'
    worksheet4.write(0, 0, 'Divergence of cluster centroids')
    divcell = (((Diverg.shape[0])*(Diverg.shape[0]))-(Diverg.shape[0])) / 2
    divsum = Diverg.sum() / divcell
    worksheet4.write(0, 2, 'Mean divergence')
    worksheet4.write(0, 3, divsum)
    for i in range(0, Diverg.shape[1]):
        worksheet4.write(1, i+2, 'cluster' + str(i+1))
        for j in range(0, Diverg.shape[0]):
            worksheet4.write(j+2, i+2, Diverg[j, i])
    for i in range(0, Diverg.shape[0]):
        worksheet4.write(i+2, 1, 'cluster' + str(i+1))


def prn_xls_cluster_membership(workbook, CLlabels):
    """compute & write cluster membership to excel file """
    worksheet5 = workbook.add_worksheet()
    worksheet5.name = 'Cluster_membership'
    worksheet5.write(0, 0, 'Count cluster members')
    worksheet5.write(1, 1, 'Cluster ID')
    worksheet5.write(1, 2, 'membership')
    worksheet5.write(1, 3, '%')
    rows = CLlabels.shape[0]
    i = CLlabels.max(axis=0)+1
    data5 = np.zeros(shape=(i))
    for l in range(rows):
        data5[CLlabels[l]] = data5[CLlabels[l]]+1
    for i in range(0, data5.shape[0]):
        worksheet5.write(i+2, 1, str(i+1))
        worksheet5.write(i+2, 2, data5[i])
        worksheet5.write(i+2, 3, 100 * data5[i] / rows)


def ListdefineforaxisX(k):
    """ define list for X axis labels (inertia graph) """
    for i in range(k):
        if i == 0:
            Lx = ['2']
        else:
            Lx.append(str(i+2))
    return Lx


def Kmeans_init(number_of_clusters):
    """Kmeans initialization """
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=10,
                 max_iter=500, tol=0.00001, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1)
    return clf


def centroids_visualize(data, figuretitle, Lx, MDLabel):
    """Visualize centroids"""
    import matplotlib.pyplot as plt
    print('\nVisualize & SAVE: ', figuretitle+'.png')
    x = np.arange(0, len(Lx), 1)
    plt.figure(1)
    plt.xticks(x, Lx)
    plt.ylabel(MDLabel[0], fontsize=12, color='b')
    plt.title(figuretitle, fontsize=15, color='r')
    a = np.zeros(shape=(data.shape[1]))
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            a[j] = data[i, j]
        plt.plot(a, label=str(i+1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('__'+figuretitle+'.png', dpi=300)
    plt.show(1)
    plt.close("all")


def write2classconvergece(a, iteration):
    """ Save mean inertia convergence to xlsx file """
    import xlsxwriter
    print('\nSave mean inertia convergence to file: convergence_NBG.xlsx')
    workbook = xlsxwriter.Workbook('_convergence_NBG.xlsx')
    worksheet5 = workbook.add_worksheet()
    worksheet5.name = 'NBG_convergence'
    worksheet5.write(0, 0, 'Convergence of NBG classification (by mean div)')
    worksheet5.write(1, 1, 'Iterations')
    worksheet5.write(1, 2, 'Percent reclassified')
    worksheet5.write(1, 3, 'Number of reclassified')
    worksheet5.write(1, 4, 'Mean divergence')
    for i in range(1, iteration+1):
        worksheet5.write(i+2, 1, str(a[i, 0]))
        worksheet5.write(i+2, 2, str(a[i, 1]))
        worksheet5.write(i+2, 3, a[i, 2])
        worksheet5.write(i+2, 4, a[i, 3])
    workbook.close()


def clusterRefineNBG(CM, centroid, iteration, centroid_variance, bb):
    """ Clustering refinements by NBG,
        display mean standardized divergence (n*n)-n, n=clusters"""
    from sklearn.metrics import pairwise_distances
    all_elements = CM.sum()
    all_correct = sum(CM[i][i] for i in range(0, CM.shape[1]))
    reclassified = (1 - all_correct / all_elements) * 100
    reclassified2 = all_elements - all_correct
    xxyy = (centroid - centroid_variance) / centroid_variance
    unifor = pairwise_distances(xxyy, metric='euclidean')
    xyz = (unifor.shape[0] * unifor.shape[0]) - unifor.shape[0]
    divsum = unifor.sum() / xyz
    print(' %3.0f    %0.4f          ( %5.0f )           %.6f' % (iteration,
                                                                 reclassified,
                                                                 reclassified2,
                                                                 divsum))
    bb[iteration, 0] = iteration
    bb[iteration, 1] = reclassified
    bb[iteration, 2] = reclassified2
    bb[iteration, 3] = divsum
    return bb, reclassified2


def clustering_Kmeans_by_NBG(data, ML2, maxC, maxNBG, f, MDLabel,
                             Clustering_method):
    """ Kmeans clustering refined by NBG -density, display mean divergence"""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import pairwise_distances
    print('\nClustering refined by NBG (display standardized mean divergence)')
    print('\n   1st: K-means clustering ')
    Nofclusters = input_screen_int('       Number of clusters', 2, maxC)
    clf = Kmeans_init(Nofclusters)
    X = clf.fit(data)
    Nofrefine = maxNBG
    print('\n   2nd:refine by NBG classification, MAX iterations: ', Nofrefine)
    print('\n   no     %              vectors       mean(st.divergence)')
    train = X.labels_
    iteration = 0
    reclassified = 1
    bb = np.zeros(shape=(Nofrefine+1, 4))
    while (iteration < Nofrefine) and (reclassified > 0):
        iteration = iteration + 1
        clf = GaussianNB()
        Y = clf.fit(data, train).predict(data)
        centroids = clf.fit(data, train).theta_
        centroid_variance = clf.fit(data, train).sigma_
        CM = confusion_matrix(train, Y)
        Diverg = pairwise_distances(centroids, metric='euclidean')
        bb, reclassified = clusterRefineNBG(CM, centroids, iteration,
                                            centroid_variance, bb)
        train = Y
    write2classconvergece(bb, iteration)
    CM = confusion_matrix(X.labels_, Y)
    import xlsxwriter
    file2write = '_clustering_output_tables'+'.xlsx'
    f.write('\n    Save clustering outputs to ' + file2write)
    workbook = xlsxwriter.Workbook(file2write)
    prn_xls_centroids(workbook, centroids, ML2)
    prn_xls_sigma(workbook, centroid_variance, ML2)
    [all_element, all_correct, reclassified] = prnxls_confuse(workbook, CM)
    prn_xls_divergence(workbook, Diverg)
    prn_xls_cluster_membership(workbook, Y)
    workbook.close()
    print(' NBG iterations: ', iteration, 'output file:', file2write)
    print('   Centroids, Sigma, Divergence, Occurence, Confusion Matrix')
    print('     Confusion of KMEANS versus F I N A L  NBG')
    xyz = all_element - all_correct
    print('         Reclassified %.4f percent ( %.0f ) ' % (reclassified, xyz))
    centroids_visualize(centroids, 'Centroids', ML2, MDLabel)
    f.write('\n        Save centroids to centroids.png')
    centroids_visualize(centroid_variance, 'Sigma', ML2, MDLabel)
    f.write('\n        Save sigma to Sigma.png')
    return Y


def clustering_Kmeans(data, LabelLST, maxC, maxNBG, f, FigureLabels,
                      Clustering_method):
    """ Kmeans clustering """
    from sklearn.metrics import pairwise_distances
    print('   K-means clustering ')
    Nofclusters = input_screen_int('   Number of clusters', 2, maxC)
    clf = Kmeans_init(Nofclusters)
    X = clf.fit(data)
    CLlabels = X.labels_
    centroids = X.cluster_centers_
    Diverg = pairwise_distances(centroids, metric='euclidean')
    import xlsxwriter
    file2write = '_clustering_Kmeans'+'.xlsx'
    f.write('\n    Save clustering outputs to ' + file2write)
    workbook = xlsxwriter.Workbook(file2write)
    prn_xls_centroids(workbook, centroids, LabelLST)
    prn_xls_divergence(workbook, Diverg)
    prn_xls_cluster_membership(workbook, CLlabels)
    workbook.close()
    centroids_visualize(centroids, 'Centroids', LabelLST, FigureLabels)
    f.write('\n        Save centroids to centroids.png')
    return CLlabels


def creatematrix(rows, cols, ids, labels):
    """ vector to image matrix"""
    total = (rows * cols)
    labels2 = np.zeros(shape=(total))
    for i in range(0, ids.shape[0]):
        k = int(ids[i]-1)
        labels2[k] = labels[i]+1
    b = np.reshape(labels2, (rows, cols))
    return b


def CreateMask_fromCluster(c):
    """Create mask matrix from cluster image matrix """
    mask = np.zeros(shape=(c.shape[0], c.shape[1]))
    for i in range(0, c.shape[0]):
        for j in range(0, c.shape[1]):
            if c[i, j] > 0:
                mask[i, j] = 1
    return mask


def plotmatrix(c, xyrange, lut, name1, yesno, MDLabel):
    """plot a matrix """
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(c, cmap=lut, extent=xyrange)
    if yesno == 'y':
        plt.colorbar(label=MDLabel[0])
    plt.xlabel(MDLabel[1])
    plt.ylabel(MDLabel[2])
    plt.title(name1)
    plt.savefig(name1+'.png', dpi=300)
    plt.show(1)
    plt.close("all")


def savematrix2image(c, name1):
    """save image to matlab, tif & csv files """
    import scipy.io as sio
    import scipy.misc
    print('SAVE CLUSTER IMAGE to:')
    sio.savemat(name1+'.mat', {'labels': c})
    print('   ', name1+'.mat')
    scipy.misc.toimage(c, high=np.max(c), low=np.min(c),
                       mode='I').save(name1 + '.tif')
    print('   ', name1 + '.tif', '(16 bit, in true [min, max])')
    np.savetxt(name1+'_image.csv', c, fmt='%.0f', delimiter=',')
    print('   ', name1 + '_image.csv')


def display_save_maskimage(xyrange, c, MDLabel):
    """covert vector cluster labels to image, plot &  save as csv, mat, tif """
    mask = CreateMask_fromCluster(c)
    print('\nDisplay mask image')
    plotmatrix(mask, xyrange, 'hot', 'Mask', 'n', MDLabel)


def display_save_clusterimage(rows, cols, xyrange, data, labels, f, w, MDLabe):
    """covert vector cluster labels to image, plot &  save as csv, mat, tif """
    ids = np.zeros(shape=(data.shape[0], 1))
    ids[:, 0] = data[:, 0]
    c = creatematrix(rows, cols, ids, labels)
    print('\nVisualize cluster image')
    f.write('\n   VISUALIZE cluster image & save to Clusters.png')
    plotmatrix(c, xyrange, 'spectral', w, 'y', MDLabe)
    savematrix2image(c, 'Clustermap')
    f.write('\n        Save to Clustermap.tif, & Clustermap.mat')
    display_save_maskimage(xyrange, c, MDLabe)


def display_RLST(rows, cols, xyrange, data, RLST, x, f, MDLabel):
    """ display RLST images and save to png/tif files """
    import scipy.misc
    print('\nVisualize the R(data) images')
    f.write('\n VISUALIZE & SAVE (png/tif/csv) the RLST images')
    ids = np.zeros(shape=(data.shape[0], 1))
    ids[:, 0] = data[:, 0]
    labels = np.zeros(shape=(data.shape[0], 1))
    Display_yesno3 = input_screen_str_yn('Save RLST images to TIF & CSV ? ')
    for i in range(0, RLST.shape[1]):
        labels[:, 0] = RLST[:, i]
        c = creatematrix(rows,  cols, ids, labels)
        RLSTname = 'R' + str(i+1) + '_' + x[i]
        f.write('\n    ' + RLSTname)
        plotmatrix(c, xyrange, 'Greys', RLSTname, 'y', MDLabel)
        if Display_yesno3 == 'Y' or Display_yesno3 == 'y':
            scipy.misc.toimage(c, high=np.max(c), low=np.min(c),
                               mode='I').save(RLSTname + '.tif')
            np.savetxt(RLSTname + '.csv', c, fmt='%.4f', delimiter=',')


def display_LST(rows, cols, xyrange, data, x, f, MDLabel):
    """ display LST images and save to png/tiff files """
    print('VISUALIZE & SAVE (png) the LST images')
    f.write('\n   VISUALIZE & SAVE (png) the LST images')
    ids, LST = create_data_files(data)
    labels = np.zeros(shape=(data.shape[0], 1))
    for i in range(0, LST.shape[1]):
        labels[:, 0] = LST[:, i]
        c = creatematrix(rows,  cols, ids, labels)
        RLSTname = 'L' + str(i+1) + '_' + x[i]
        f.write('\n      ' + RLSTname)
        plotmatrix(c, xyrange, 'Greys', RLSTname, 'y', MDLabel)


def compute_descriptive_stats(RLST, x, lst_or_rlst):
    """compute mean, st.dev, kurtosis, skew"""
    from scipy.stats import kurtosis
    from scipy.stats import skew
    import xlsxwriter
    a = np.zeros(shape=(RLST.shape[1], 6))
    a[:, 0] = RLST.min(axis=0)
    a[:, 1] = RLST.max(axis=0)
    a[:, 2] = RLST.mean(axis=0)
    a[:, 3] = RLST.std(axis=0)
    a[:, 4] = skew(RLST, axis=0)
    a[:, 5] = kurtosis(RLST, axis=0)
    y = ['Minimum', 'Maximum', 'Mean', 'St.Dev.', 'Skew', 'Kurtosis']
    if lst_or_rlst == 'RLST':
        print('SAVE descriptive RLST stats to file: descriptives_RLST.xlsx')
        workbook = xlsxwriter.Workbook('_descriptives_RLST.xlsx')
    else:
        print('SAVE descriptive LST stats to file: descriptives_LST.xlsx')
        workbook = xlsxwriter.Workbook('_descriptives_LST.xlsx')
    worksheet5 = workbook.add_worksheet()
    worksheet5.name = 'descriptives'
    worksheet5.write(0, 0, 'descriptive stats')
    for i in range(6):
        worksheet5.write(1, i+1, y[i])
    for i in range(len(x)):
        worksheet5.write(i+2, 0, x[i])
    for i in range(a.shape[1]):
        for j in range(a.shape[0]):
            worksheet5.write(j+2, i+1, str(a[j, i]))
    workbook.close()


def descriptive_stats_RLST(data, LABELmonths3, Lx, f, lst_or_rlst):
    """Compute, display & save to xlsx descriptive statistics for RLST """
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis
    from scipy.stats import skew
    print('\nCompute, display & save (to xlsx) descriptive statistics')
    f.write('\n Compute, display descriptive statistics')
    compute_descriptive_stats(data, LABELmonths3, lst_or_rlst)
    x = np.arange(0, len(Lx), 1)
    plt.figure(1)
    plt.xticks(x, Lx)
    plt.title('Absolute skew, kurtosis')
    c = abs(kurtosis(data, axis=0))
    b = abs(skew(data, axis=0))
    plt.plot(c, marker='D', markersize=4, linestyle='-',
             color='r', label='|Kurtosis|')
    plt.plot(b, marker='o', markersize=4, linestyle='--',
             color='b', label='|Skew|')
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if lst_or_rlst == 'RLST':
        plt.savefig('RLST_abs_kurtosis_skew.png', dpi=300)
        f.write('\n    Write RLST stats to descriptives_RLST.xlsx')
    else:
        plt.savefig('LST_abs_kurtosis_skew.png', dpi=300)
        f.write('\n    Write RLST stats to descriptives_LST.xlsx')
    plt.show(1)
    plt.close("all")
    f.write('\n    Save absolute kurtosis & skew to abs_kurtosis_skew.png')


def printNPP(RLST, x, f, lst_or_rlst):
    """print normal propability plot """
    from scipy import stats
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale
    f.write('\n Display & write NPP files')
    for X in range(RLST.shape[1]):
        plt.figure(X)
        standardized_X = scale(RLST[:, X], axis=0)
        stats.probplot(standardized_X, plot=plt)
        plt.title(x[X])
        if lst_or_rlst == 'RLST':
            plt.savefig('NPP_RH' + str(X+1) + '.png', dpi=300)
            f.write('\n    NPP_RH' + str(X+1) + '.png')
        else:
            plt.savefig('NPP_H' + str(X+1) + '.png', dpi=300)
            f.write('\n    NPP_H' + str(X+1) + '.png')
        plt.show(X)
        plt.close("all")


def printHST(RLST, Fstring, xmin, xmax, x, f, MDLabel):
    """ print histogram of LST/RLST"""
    import matplotlib.pyplot as plt
    print('DISPLAY & PRINT histograms for', Fstring, ' data')
    f.write('\n   DISPLAY & PRINT histograms for ' + Fstring + ' data')
    if Fstring == 'LST':
        ids, LST = create_data_files(RLST)
        RLST = LST
    for X in range(RLST.shape[1]):
        plt.figure(1)
        plt.hist(RLST[:, X], bins=200, range=[xmin, xmax], normed=True,
                 edgecolor='white')
        plt.title(x[X])
        plt.xlabel(MDLabel[0])
        plt.ylabel("Frequency")
        plt.savefig('H_' + Fstring + str(X+1) + '.png',  dpi=300)
        f.write('\n       H_' + Fstring + str(X+1) + '.png')
        plt.show(1)
    plt.close("all")


def printRLST_correlation(data, x):
    """ write RLST cross correlation matrix  to xls file"""
    import xlsxwriter
    print('Create RLST_correlation.xlsx')
    workbook = xlsxwriter.Workbook('_RLST_correlation.xlsx')
    worksheet1 = workbook.add_worksheet()
    worksheet1.write(1, 0, 'Cross Correlation')
    worksheet1.name = 'Cross_correlation'
    for i in range(0, data.shape[0]):
        worksheet1.write(1, i+2, x[i])
        worksheet1.write(i+2, 1, x[i])
        for j in range(0, data.shape[1]):
            worksheet1.write(i+2, j+2, str(round(data[i, j], 4)))
    workbook.close()


def saveClusterLabels_to_vectors(f, Labels):
    """Saves cluster membership as vectors to a csv file """
    LV_yes_no = input_screen_str_yn('\n    Save Cluster Labels as vectors ? ')
    if LV_yes_no == 'Y' or LV_yes_no == 'y':
        print('\n    Saved clusters vector map to Labels.csv \n')
        f.write('\n    Save cluster vector map to Labels.csv')
        np.savetxt('Labels.csv', Labels, fmt='%.0f', delimiter=',')


def MainRun(data, rows, cols, GeoExtent, FigureLabels, LabelHLatLonLST,
            LabelLST, LabelLSTxls, Hmin, Hmax, HRmin, HRmax, Clustering_method,
            clustering_options):
    """ Main run module of SVR-mg.py"""
    f, oldpath = findpaths_data2csv(data)
    In, maxC, mNBG = program_constants()
    xyxstr = 'LST:Stats, Correlation, NPPS, Images, Histograms ? '
    Display_yesno2 = input_screen_str_yn(xyxstr)
    if Display_yesno2 == 'Y' or Display_yesno2 == 'y':
        f.write('\n DISPLAY:descriptives, NPPs, LST images & histograms')
        data2 = data[:, 1:data.shape[1]]
        descriptive_stats_RLST(data2, LabelLSTxls, LabelLST, f, 'LST')
        display_LST(rows, cols, GeoExtent, data, LabelLSTxls, f, FigureLabels)
        printNPP(data2, LabelLSTxls, f, 'LST')
        printHST(data, 'LST', Hmin, Hmax, LabelLSTxls, f, FigureLabels)
    Reconstruct = ImplementSVR_MG(data, LabelHLatLonLST, f)
    Display_yesno3 = input_screen_str_yn(
        'R(data):Stats, Correlation, NPPS, Images, Histograms ? ')
    if Display_yesno3 == 'Y' or Display_yesno3 == 'y':
        descriptive_stats_RLST(Reconstruct, LabelLSTxls, LabelLST, f, 'RLST')
        printNPP(Reconstruct, LabelLSTxls, f, 'RLST')
        display_RLST(rows, cols, GeoExtent, data, Reconstruct, LabelLSTxls, f,
                     FigureLabels)
        printHST(Reconstruct, 'RLST', HRmin, HRmax, LabelLSTxls, f,
                 FigureLabels)
    Cluster_yesno = input_screen_str_yn('Cluster R(data) ? ')
    if Cluster_yesno == 'Y' or Cluster_yesno == 'y':
        if Clustering_method in clustering_options:
            if Clustering_method == clustering_options[1]:
                Labels = clustering_Kmeans_by_NBG(Reconstruct, LabelLST, maxC,
                                                  mNBG, f, FigureLabels,
                                                  Clustering_method)
            if Clustering_method == clustering_options[0]:
                Labels = clustering_Kmeans(Reconstruct, LabelLST, maxC, mNBG,
                                           f, FigureLabels, Clustering_method)
            saveClusterLabels_to_vectors(f, Labels)
            display_save_clusterimage(rows, cols, GeoExtent, data, Labels, f,
                                      'Cluster', FigureLabels)
    f.close()
    from os import chdir
    chdir(oldpath)
