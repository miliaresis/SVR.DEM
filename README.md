# SVR.DEM
**Dimension reduction of multi-dimensional elevation data for DEMs optimization & evaluation**
* A win python program (https://winpython.github.io/) consiting of 3 modules **dmr_mg.py** and the 2 library MODULEs **dmr_data_headers**, & **dmr_myf.py**.
* _2-d DEM optimization method presents an alternative to DEM comparison by elevation differences modeling. In it's general version (3.d, 4.d etc., etc.) is not limited to the comparison/evaluation of only 2 DEMs but can handle 3 DEMs, 4 DEMs at the same time, etc. etc._
* A YouTube video is available at https://vimeo.com/253018987
* Table 1: Cluster centroids 
See SVR.Clusters https://github.com/miliaresis/SVR.CLUSTERS for min/max statistics and feature space visualization before you derive any conclusions. K-means clustering followed by exhausted NBG clasification (177 iterations) are used for centroids/cluster definition.

| Clusters  	| Mean 	|   (m)	|       	| st.dev. 	|    (m)  	|         	| Occurrence 	|            	|
|-----------	|------	|------	|-------	|---------	|---------	|---------	|------------	|------------	|
| Centroids 	| ALOS 	| SRTM 	| ASTER 	| ALOS    	| SRTM    	| ASTER   	| pixels     	| %          	|
| 1         	| 1.9  	| 1.7  	| -3.5  	| 0.5     	| 0.5     	| 1       	| 1102976    	| 15.8       	|
| 2         	| 3.5  	| 3.2  	| -6.7  	| 0.5     	| 0.4     	| 0.9     	| 1167599    	| 16.7       	|
| 3         	| 1.7  	| 1.6  	| -3.3  	| 22.2    	| 20.2    	| 42      	| 522388     	| 7.5        	|
| 4         	| 5.3  	| 4.8  	| -10.1 	| 0.6     	| 0.5     	| 1.1     	| 1132675    	| 16.2       	|
| 5         	| -4.6 	| -4.2 	| 8.7   	| 2       	| 1.8     	| 3.8     	| 1006213    	| 14.4       	|
| 6         	| -0.3 	| -0.3 	| 0.7   	| 0.8     	| 0.7     	| 1.5     	| 1076842    	| 15.4       	|
| 7         	| 8.4  	| 7.6  	| -15.9 	| 1.5     	| 1.3     	| 2.8     	| 986008     	| 14.1       	|

* Data
  * Multi-(2-d) dimensional (ALOS median & average) DEMs of SE Zagros Ranges. Mendeley Data,  v.4, _**http://dx.doi.org/10.17632/z4nxdjdyys.4**_ [_It is applied to ALOS median & average DEMS, aiming to stretch (in accuracy terms) the method to it's limits_] 
 * Multi-(2-d) dimensional (ALOS {median}, SRTM) DEM of SE Zagros Ranges. Mendeley Data, v.5, _**http://dx.doi.org/10.17632/k9zpyh8c9k.5**_
  * Multi-(3-d) dimensional (ALOS {median}, SRTM, ASTER) DEM of SE Zagros Ranges, Mendeley Data,  v.15, _**http://dx.doi.org/10.17632/bswsr3gpy2.15**_
  * Multi-(4-d) dimensional (ALOS, SRTM, ASTER, NED)  DEM of Death Valley (CA). Mendeley Data, v.10, _**http://dx.doi.org/10.17632/fbd9pd6hnx.10**_ [_it tests the software performance with 4-d data only, (since the vertical datums among  a) ALOS, SRTM, ASTER GDEM and b) NED DTM differ)_]
* Publications
* **Used in publications**
  1. Dimension reduction of multi-dimensional elevation data for DEMs optimization & evaluation (in review)
![Example of output images](https://github.com/miliaresis/SVR.DEM/blob/master/mapping.jpg)
* **Background publications**: *Quantification & evaluation of digital elevation models*
  1. Miliaresis G., Paraschou Ch.V., 2011. An evaluation of the accuracy of the ASTER GDEM and the role of stack number: A case study of   Nisiros Island, Greece. *Remote Sensing Letters*  2(2):127-135. DOI:10.1080/01431161.2010.503667 
  1. Miliaresis G., Delikaraoglou D., 2009. Effects of Percent Tree Canopy Density and DEM Mis-registration to SRTM/NED Vegetation Height Estimates. *Remote Sensing* 1(2):36-49, DOI:10.3390/rs1020036 
  1. Miliaresis G., 2008. The Landcover Impact on the Aspect/Slope Accuracy Dependence of the SRTM-1 Elevation Data for the Humboldt Range. *Sensors* 8(5):3134-3149. DOI: 10.3390/s8053134. 
  1. Miliaresis G., 2007. An upland object based modeling of the vertical accuracy of the SRTM-1 elevation dataset. *Journal of Spatial Sciences* 52(1):13-29. DOI: 10.1080/14498596.2007.9635097 
  1. Miliaresis G., Paraschou Ch., 2005. Vertical accuracy of the SRTM DTED Level 1 of Crete. *Int. J. of Applied Earth Observation & GeoInformation* 7(1):49-59. DOI: 10.1016/j.jag.2004.12.001 
