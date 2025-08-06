#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 06:38:53 2025

@author: zjpeters
"""
import os
import pandas as pd
import numpy as np
import json

'''
notes:
    pandas doesn't recognize hidden rows, so will perform exclusion criteria in code
    exclude E-MDD
    only include subjects with 3 in column F ['exclusion (1=excluded, 2=stopped/incomplete)']
'''

# begin by loading data from spreadsheet
mddSpreadsheetLoc = os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','rawdata','AllData_MDD_10April2025_fixM.csv')
mddData = pd.read_csv(mddSpreadsheetLoc)
columnHeadersAndFilenames= json.load(open(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','code','columnHeadersAndFileNames.json')))
# mddData.columns = list(columnHeadersAndFilenames.keys())
# x = list(columnHeadersAndFilenames.keys())
# converts the categorical columns from floating point numbers to integers
#%%
excludeEMDDList = mddData['Study'] != 'E-MDD'
mddDataEMDDRemoved = mddData.loc[excludeEMDDList,:]
excludeList = mddDataEMDDRemoved['exclusion'] == 3
mddDataExcludeRemoved = mddDataEMDDRemoved.loc[excludeList,:]
#%%

mddDataExcludeRemoved.astype({'control ID Doro': 'int32'})

#%%
'''
write a function to replace all nan with randomized value within upper and lower
limits of existing data

account for the need for lists in the previous infections column

various subjects have 'm' for unknown
'''

testIntColumn = mddDataExcludeRemoved['Depression (BDI) classification']
nullList = testIntColumn.isnull()
fullList = testIntColumn[~nullList]
fullList = np.array(fullList)
if any(fullList % 1):
    fullList = np.array(fullList, dtype='float64')
else:
    fullList = np.array(fullList, dtype='int64')
    
testFloatColumn = mddDataExcludeRemoved['Neutrophils per µl']
nullList = testFloatColumn.isnull()
fullList = testFloatColumn[~nullList]
fullList = np.array(fullList)
if any(fullList % 1):
    fullList = np.array(fullList, dtype='float64')
else:
    fullList = np.array(fullList, dtype='int64')
    

def fillInNaNs(columnToFill, studyList, replaceRandomFromRange=True):
    """

    Parameters
    ----------
    columnToFill : TYPE
        The column to have the nan values filled with randomized data.
    studyList : TYPE
        List saying which group each subject is in, assumes HC-MDD and MDD.
    replaceRandomFromRange : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    newList : TYPE
        pandas dataframe with nans replaced with randomized values.

    """
    rng = np.random.default_rng(12345)
    # generate list of locations with nan values
    nullList = columnToFill.isnull()
    # generate a column with only filled in values to use for sampling
    fullListDF = columnToFill[~nullList]
    lessThanZeroList = fullListDF < 0
    maskedList = fullListDF[~lessThanZeroList]
    fullList = maskedList.to_numpy()
    studyListNullMasked = studyList[~nullList]
    studyListZeromasked = studyListNullMasked[~lessThanZeroList]
    hcIDX = studyListZeromasked == 'HC-MDD'
    mddIDX = studyListZeromasked == 'MDD'
    hcColumn = maskedList.loc[hcIDX]
    mddColumn = maskedList.loc[mddIDX]
    columnNegRemoved = columnToFill.to_numpy()
    columnNegRemoved[columnNegRemoved < 0] = np.nan
    newList = list(columnToFill)
    # check whether numbers are integers or floating point numbers
    # if, when divided by 1 there is a remainder, value is floating point
    if any(fullList % 1):
        print("running for float")
        fullList = maskedList.to_numpy(dtype='float64')
        listMax = np.max(fullList)
        listMin = np.min(fullList)
        # generate a range of random numbers, 12345 is the seed value
        for actNull in enumerate(nullList):
            if actNull[1] == True:
                # if nan is found, replaces with randomly generated value between min and max 
                newList[actNull[0]] = rng.uniform(listMin, listMax)
        newList = np.array(newList, dtype='float64')
    # if when divided by 1 there is no remainder, value is an integer
    else:
        print("running for integer")
        fullList = maskedList.to_numpy(dtype='int64')
        for actNull in enumerate(nullList):
            if actNull[1] == True:
                # if nan is found, replaces with randomly sampled integer from original list
                if studyList[0] == 'HC-MDD':
                    
                    newList[actNull[0]] = hcColumn.sample().to_numpy(dtype='int64')[0]
                elif studyList[1] == 'MDD':
                    newList[actNull[0]] = mddColumn.sample().to_numpy(dtype='int64')[0]
        newList = np.array(newList, dtype='int64')# .to_numpy(dtype='int64')
    newList = pd.to_numeric(pd.Series(newList,name=columnToFill.name), errors='coerce')
    return newList   

studyList = mddDataExcludeRemoved['Study']

x = fillInNaNs(testIntColumn, studyList)

y = fillInNaNs(testFloatColumn, studyList)

#%% fix '<0,6' and '<0,3' in 'C-reaktives Protein mg/l' column
rng = np.random.default_rng(12345)
lessThanIdx = mddDataExcludeRemoved['C-reaktives Protein mg/l'] == '<0,6'
newList = list(mddDataExcludeRemoved['C-reaktives Protein mg/l'])
for actNull in enumerate(lessThanIdx):
    if actNull[1] == True:
        # replace '<0,6' with a float between 0 and 0.6
        # if looking for a certain number of points after decimal, use code below
        # newList[actNull[0]] = round(rng.uniform(0, 0.6), 2)
        newList[actNull[0]] = rng.uniform(0, 0.6)
newList = pd.DataFrame({'C-reaktives Protein mg/l':newList})

lessThanIdx = newList['C-reaktives Protein mg/l'] == '< 0,6'
newList = list(newList['C-reaktives Protein mg/l'])
for actNull in enumerate(lessThanIdx):
    if actNull[1] == True:
        # replace '<0,6' with a float between 0 and 0.6
        # if looking for a certain number of points after decimal, use code below
        # newList[actNull[0]] = round(rng.uniform(0, 0.6), 2)
        newList[actNull[0]] = rng.uniform(0, 0.3)
newList = pd.DataFrame({'C-reaktives Protein mg/l':newList})

lessThanIdx = newList['C-reaktives Protein mg/l'] == '<0,3'
newList = list(newList['C-reaktives Protein mg/l'])
for actNull in enumerate(lessThanIdx):
    if actNull[1] == True:
        # replace '<0,6' with a float between 0 and 0.6
        # if looking for a certain number of points after decimal, use code below
        # newList[actNull[0]] = round(rng.uniform(0, 0.6), 2)
        newList[actNull[0]] = rng.uniform(0, 0.3)
# newList = np.array(newList, dtype='float64')
newList = pd.to_numeric(pd.Series(newList, name='C-reaktives Protein mg/l'), errors='coerce')
# newList = np.array(newList, dtype='float64')
mddDataExcludeRemoved.loc[:,'C-reaktives Protein mg/l'] = newList

#%% fix '<0,25658' in 'NCBAIL-18Conc' column

lessThanIdx = mddDataExcludeRemoved['NCBAIL-18Conc'] == '<0,25658'
newList = list(mddDataExcludeRemoved['NCBAIL-18Conc'])
for actNull in enumerate(lessThanIdx):
    if actNull[1] == True:
        # replace '<0,6' with a float between 0 and 0.6
        # if looking for a certain number of points after decimal, use code below
        # newList[actNull[0]] = round(rng.uniform(0, 0.6), 2)
        newList[actNull[0]] = rng.uniform(0, 0.25658)
    newList = pd.to_numeric(pd.Series(newList, name='NCBAIL-18Conc'), errors='coerce')
mddDataExcludeRemoved.loc[:,'NCBAIL-18Conc'] = newList
### if needing to re-write output, can uncomment below
mddDataExcludeRemoved.to_csv(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','rawdata','preprocessed_MDD_data.csv'), index=False)

#%% read updated csv
preprocessedData = pd.read_csv(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','rawdata','preprocessed_MDD_data.csv'))
# preprocessedData.columns = columnHeaders
#%% replace nan values with randomized values
"""
columns to ignore when replacing nan values:
    Date of Analysis
    Study
    ID
    control ID Doro
    group (1=patient, 2=HC)
    exclusion (1=excluded, 2=stopped/incomplete)
    DATEOFBIRTH
    AGE
    SEX
    Smoking Status
    Smoking (1=yes (py), 2=no or formerly (> 2m abstinent))
    Alcohol (1=yes, 2=no or formerly (> 2m abstinent))
    Other drugs (1=yes, 2=no or formerly (> 2m abstinent))
    0 {not sure what this column is in general}
    Weight
    BMI
    BMI Group
"""
### updated the formatting in the csv in order to remove /n lines that threw off importing
#%% check MFICD15 column, keeps adding very large negative numbers
studyList = preprocessedData['Study']
columnToFill = preprocessedData['ClassicalMFICD15']
x = fillInNaNs(preprocessedData['ClassicalMFICD15'], studyList)
#%%
ignoreWhenFillingNan = ['Date of Analysis', 'Study', 'ID', 'control ID Doro', 'group', 'exclusion', 'DATEOFBIRTH', 'AGE', 'SEX', 'Smoking Status', 'Smoking', 'Alcohol', 'Other drugs', '0', 'Weight', 'BMI','BMI Group', 'Infektionskrankheiten']
updatedPreprocessedList = preprocessedData
studyList = preprocessedData['Study']

for i in preprocessedData.columns:
    if i not in ignoreWhenFillingNan:
        print(i)
        filledNanColumn = fillInNaNs(preprocessedData[i], studyList)
        updatedPreprocessedList.loc[:,i] = filledNanColumn

updatedPreprocessedList.to_csv(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','rawdata','preprocessed_MDD_data_nan_filled.csv'))
