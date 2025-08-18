#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:47:26 2025

@author: zjpeters
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import ListedColormap
from string import Template
# load data
columnHeadersAndFilenames= json.load(open(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','code','columnHeadersAndFileNames.json')))
plt.style.use('seaborn-v0_8-colorblind')

# colors to use in the output figures, with scatter color being slightly darker
colorHCHex = '#f3766e'
colorHCScatterHex = '#fe4d34'
colorMDDHex = '#1cbdc2'
colorMDDScatterHex = '#14a7c2'

# data locations
derivatives = os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives')
rawdata = os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','rawdata')
preprocessedData = pd.read_csv(os.path.join(rawdata,'preprocessed_MDD_data.csv'), index_col=0)

def hex_to_rgb(value):
    """"taken from: https://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa"""
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

#%% begin processing data
# set list of columns that won't be used when running t-test
ignoreColumns = ['Unnamed: 0', 'Date of Analysis', 'Study', 'ID', 'control ID Doro', 'group', \
                'exclusion', 'DATEOFBIRTH', 'AGE', 'SEX', 'Smoking Status', \
                'Smoking', 'Alcohol', 'Other drugs', '0', 'Weight', \
                'Infektionskrankheiten']

hcIdx = preprocessedData['group'] == 2
mddIdx = preprocessedData['group'] == 1
groups = [0,1]
desiredPval = 0.05
nChecked = 0

# figure preparation
plt.close('all')    # close open figures
w = 0.6             # width of bars in graph
figWidth = 6
figHeight = 8

#%% permutation testing the t-test
def diffOfMeans(x,y):
    diff = np.mean(x) - np.mean(y)
    return diff

def ttest(A,B, nResample=1000):
    mu_A = np.sum(A, axis=0)/nResample
    mu_B = np.sum(B, axis=0)/nResample
    
    sig_2_A = np.sum((A - mu_A)**2)/(nResample - 1)
    sig_2_B = np.sum((B - mu_B)**2)/(nResample - 1)
    se_A = np.sqrt(sig_2_A/nResample)
    se_B = np.sqrt(sig_2_B/nResample)
    # test null hypothesis
    tStat = (mu_A - mu_B)/np.sqrt(se_A**2 + se_B**2)
    return tStat
for actColumn in preprocessedData.columns:
    if actColumn not in ignoreColumns:
        truncDF = preprocessedData.loc[preprocessedData[actColumn].notnull()]
        nonTruncDF = preprocessedData.loc[preprocessedData[actColumn].notnull()]
        hcGroupNonTrunc = nonTruncDF[actColumn].loc[nonTruncDF['Study'] == 'HC-MDD']
        mddGroupNonTrunc = nonTruncDF[actColumn].loc[nonTruncDF['Study'] == 'MDD']
        upperLimit = truncDF[actColumn].quantile(0.9)
        lowerLimit = truncDF[actColumn].quantile(0.1)
        truncDF = truncDF.loc[truncDF[actColumn] > lowerLimit]
        truncDF = truncDF.loc[truncDF[actColumn] < upperLimit]
        hcGroup = truncDF[actColumn].loc[truncDF['Study'] == 'HC-MDD']
        mddGroup = truncDF[actColumn].loc[truncDF['Study'] == 'MDD']
        print(actColumn)
        print('N of HC before truncation: ', len(hcGroupNonTrunc), 'N of HC after truncation: ', len(hcGroup))
        print('N of MDD: ', len(mddGroupNonTrunc), 'N of HC after truncation: ', len(mddGroup))
        if len(hcGroup) > 1 and len(mddGroup) > 1:
            permResult = scipy_stats.permutation_test((hcGroup, mddGroup), ttest, n_resamples=10000, random_state=12345)
            tStat = permResult.statistic
            pVal = permResult.pvalue
            print(pVal)
            data = [hcGroup, mddGroup]
            nChecked += 1
            n_hc = len(hcGroup)
            n_mdd = len(mddGroup)
            n_outliers = len(preprocessedData[actColumn]) - (n_hc + n_mdd)
            # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
            fig, ax = plt.subplots()
            fig.set_figwidth(figWidth)
            fig.set_figheight(figHeight)
            # plot bar graph showing mean of HC and MDD with SEM error bars
            ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
            ax.set_ylabel(columnHeadersAndFilenames[actColumn]['axisLabel'])
            for j in range(len(groups)):
                # distribute scatter randomly across whole width of bar
                if j == 0:
                    ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorHCScatterHex)
                else:
                    ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorMDDScatterHex)
            plt.title(columnHeadersAndFilenames[actColumn]['figureTitle'])
            plt.legend()
            # below if statement adds brackets for significance and outputs into sig figures folder
            if pVal < desiredPval:
                x1, x2 = 0, 1
                y, h, col = max(map(max, [hcGroup, mddGroup])) + 2, 2, 'k'
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                plt.savefig(os.path.join(derivatives,'permutationTesting','sigFigures',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap.svg'))
            plt.show()
            plt.savefig(os.path.join(derivatives,'permutationTesting',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap.svg'))
            plt.close()

#%% permutation testing outliers included

for actColumn in preprocessedData.columns:
    if actColumn not in ignoreColumns:
        truncDF = preprocessedData.loc[preprocessedData[actColumn].notnull()]
        nonTruncDF = preprocessedData.loc[preprocessedData[actColumn].notnull()]
        hcGroupNonTrunc = nonTruncDF[actColumn].loc[nonTruncDF['Study'] == 'HC-MDD']
        mddGroupNonTrunc = nonTruncDF[actColumn].loc[nonTruncDF['Study'] == 'MDD']
        # upperLimit = truncDF[i].quantile(0.9)
        # lowerLimit = truncDF[i].quantile(0.1)
        truncDF = truncDF.loc[truncDF[actColumn] > lowerLimit]
        truncDF = truncDF.loc[truncDF[actColumn] < upperLimit]
        hcGroup = truncDF[actColumn].loc[truncDF['Study'] == 'HC-MDD']
        mddGroup = truncDF[actColumn].loc[truncDF['Study'] == 'MDD']
        print(actColumn)
        print('N of HC before truncation: ', len(hcGroupNonTrunc), 'N of HC after truncation: ', len(hcGroup))
        print('N of MDD: ', len(mddGroupNonTrunc), 'N of HC after truncation: ', len(mddGroup))
        if len(hcGroup) > 1 and len(mddGroup) > 1:
            permResult = scipy_stats.permutation_test((hcGroup, mddGroup), ttest, n_resamples=10000, random_state=12345)
            tStat = permResult.statistic
            pVal = permResult.pvalue
            print(pVal)
            data = [hcGroup, mddGroup]
            nChecked += 1
            n_hc = len(hcGroup)
            n_mdd = len(mddGroup)
            n_outliers = len(preprocessedData[actColumn]) - (n_hc + n_mdd)
            # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
            fig, ax = plt.subplots()
            fig.set_figwidth(figWidth)
            fig.set_figheight(figHeight)
            # plot bar graph showing mean of HC and MDD with SEM error bars
            ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
            ax.set_ylabel(columnHeadersAndFilenames[actColumn]['axisLabel'])
            for j in range(len(groups)):
                # distribute scatter randomly across whole width of bar
                if j == 0:
                    ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorHCScatterHex)
                else:
                    ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorMDDScatterHex)
            plt.title(columnHeadersAndFilenames[actColumn]['figureTitle'])
            plt.legend()
            if pVal < desiredPval:
                x1, x2 = 0, 1
                y, h, col = max(map(max, [hcGroup, mddGroup])) + 2, 2, 'k'
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                plt.savefig(os.path.join(derivatives,'permutationTestingOutliersIncluded','sigFigures',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap_outliers_included.svg'))
            plt.show()
            plt.savefig(os.path.join(derivatives,'permutationTestingOutliersIncluded',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap_outliers_included.svg'))
            plt.close()

#%% same calculation as above but using violin plots

# Also, could you please make them as thin violin plots, with colors #f3766e for HC and #1cbdc2 for MDD

for actColumn in preprocessedData.columns:
    if actColumn not in ignoreColumns:
        truncDF = preprocessedData.loc[preprocessedData[actColumn].notnull()]
        upperLimit = truncDF[actColumn].quantile(0.9)
        lowerLimit = truncDF[actColumn].quantile(0.1)
        truncDF = truncDF.loc[truncDF[actColumn] > lowerLimit]
        truncDF = truncDF.loc[truncDF[actColumn] < upperLimit]
        hcGroup = truncDF[actColumn].loc[truncDF['Study'] == 'HC-MDD']
        mddGroup = truncDF[actColumn].loc[truncDF['Study'] == 'MDD']
        if len(hcGroup) > 1 and len(mddGroup) > 1:
            permResult = scipy_stats.permutation_test((hcGroup, mddGroup), ttest, n_resamples=10000, random_state=12345)
            tStat = permResult.statistic
            pVal = permResult.pvalue
            print(pVal)
            data = [hcGroup, mddGroup]
            nChecked += 1
            n_hc = len(hcGroup)
            n_mdd = len(mddGroup)
            n_outliers = len(preprocessedData[actColumn]) - (n_hc + n_mdd)
            # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
            fig, ax = plt.subplots()
            fig.set_figwidth(figWidth)
            fig.set_figheight(figHeight)
            # plot bar graph showing mean of HC and MDD with SEM error bars
            # ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorBlindColorPalette[0],colorBlindColorPalette[3]])
            # violin plot
            violinInfo = ax.violinplot([hcGroup,mddGroup], showextrema=False, showmeans=False)
            violinInfo['bodies'][0].set_color(colorHCHex)
            violinInfo['bodies'][1].set_color(colorMDDHex)
            ax.set_xticks([1,2], ['HC','MDD'])
            ax.set_ylabel(columnHeadersAndFilenames[actColumn]['axisLabel'])
            plt.title(columnHeadersAndFilenames[actColumn]['figureTitle'])
            if pVal < desiredPval:
                plt.savefig(os.path.join(derivatives,'permutationTesting','sigFigures',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap_violin_plot.svg'))
            plt.show()
            plt.savefig(os.path.join(derivatives,'permutationTesting',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap_violin_plot.svg'))
            plt.close()

#%% violin plots without removing outliers

# Also, could you please make them as thin violin plots, with colors #f3766e for HC and #1cbdc2 for MDD

for actColumn in preprocessedData.columns:
    if actColumn not in ignoreColumns:
        truncDF = preprocessedData.loc[preprocessedData[actColumn].notnull()]
        upperLimit = truncDF[actColumn].quantile(0.9)
        lowerLimit = truncDF[actColumn].quantile(0.1)
        # truncDF = truncDF.loc[truncDF[i] > lowerLimit]
        # truncDF = truncDF.loc[truncDF[i] < upperLimit]
        hcGroup = truncDF[actColumn].loc[truncDF['Study'] == 'HC-MDD']
        mddGroup = truncDF[actColumn].loc[truncDF['Study'] == 'MDD']
        if len(hcGroup) > 1 and len(mddGroup) > 1:
            permResult = scipy_stats.permutation_test((hcGroup, mddGroup), ttest, n_resamples=10000, random_state=12345)
            tStat = permResult.statistic
            pVal = permResult.pvalue
            print(pVal)
            data = [hcGroup, mddGroup]
            nChecked += 1
            n_hc = len(hcGroup)
            n_mdd = len(mddGroup)
            n_outliers = len(preprocessedData[actColumn]) - (n_hc + n_mdd)
            # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
            fig, ax = plt.subplots()
            fig.set_figwidth(figWidth)
            fig.set_figheight(figHeight)
            # plot bar graph showing mean of HC and MDD with SEM error bars
            # ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorBlindColorPalette[0],colorBlindColorPalette[3]])
            # violin plot
            violinInfo = ax.violinplot([hcGroup,mddGroup], showextrema=False, showmeans=False)
            violinInfo['bodies'][0].set_color(colorHCHex)
            violinInfo['bodies'][1].set_color(colorMDDHex)
            ax.set_xticks([1,2], ['HC','MDD'])
            ax.set_ylabel(columnHeadersAndFilenames[actColumn]['axisLabel'])
            plt.title(columnHeadersAndFilenames[actColumn]['figureTitle'])
            if pVal < desiredPval:
                plt.savefig(os.path.join(derivatives,'permutationTestingOutliersIncluded','sigFigures',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap_violin_plot_outliers_included.svg'))
            plt.show()
            plt.savefig(os.path.join(derivatives,'permutationTestingOutliersIncluded',f'{columnHeadersAndFilenames[actColumn]["filename"]}_ttest_for_HCvMDD_bootstrap_violin_plot_outliers_included.svg'))
            plt.close()
#%% perform correlation between cytokine levels and BDI scores
def calculateLeastSquaresRegression(x, y, n=1):
    """
    Calculate the least squares regression line for a dataset of coordinates
    (x,y)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    n : TYPE, optional
        1 for linear, 2 for square, 3 for cubic. The default is 1.

    Returns
    -------
    None.

    """
    S = len(x)
    S_x = 0
    S_y = 0
    
    for i in range(S):
        S_x += x[i]**n
        S_y += y[i]
    
    t_i = x**n - (S_x/S)
    S_tt = 0
    S_ty = 0
    for i in range(S):
        S_tt += t_i[i]**2
        S_ty += t_i[i]*y[i]
    
    b_hat = S_ty/S_tt
    a_hat = (S_y - (S_x*b_hat))/S
    
    x_out = np.sort(x)
    y_out = a_hat + b_hat*x_out**n
    return x_out, y_out
plt.close('all')
bdiColumns = ['Depression (BDI) classification','Depression scale (BDI) total score','BdiII_SAF (BDI score)','BdiII_CF (BDI score)','BdiII Q9 (suicidality)']

"""
should update the below to use the cleaner nan removal of the above section
"""
def corrTest(A,B):
    rStat, pVal = scipy_stats.pearsonr(A,B)
    return rStat

for bdi in bdiColumns:
    for actColumn in preprocessedData.columns:
        if actColumn not in ignoreColumns:
            nChecked += 1
            if actColumn not in bdiColumns:
                upperLimit = preprocessedData[actColumn].quantile(0.9)# + iqr * 1.5
                lowerLimit = preprocessedData[actColumn].quantile(0.1)# - iqr * 1.5
                hcGroup = preprocessedData[actColumn].loc[hcIdx]
                preOutlierLength = len(hcGroup)
                # hcGroup = hcGroup[hcGroup < upperLimit]
                # print(preOutlierLength, len(hcGroup))
                # hcGroup = np.array(hcGroup > lowerLimit)
                mddGroup = preprocessedData[actColumn].loc[mddIdx]
                mddBDIGroup = preprocessedData[bdi].loc[mddIdx]
                # print(upperLimit, lowerLimit, np.mean(mddGroup))
                outlierUpperIdx = mddGroup < upperLimit
                mddGroup = mddGroup.loc[outlierUpperIdx]
                mddBDIGroup = mddBDIGroup.loc[outlierUpperIdx]
                outlierLowerIdx = mddGroup > lowerLimit
                mddGroup = mddGroup.loc[outlierLowerIdx]
                mddBDIGroup = mddBDIGroup.loc[outlierLowerIdx]
                mddGroup = mddGroup[mddGroup.notna()]
                mddBDIGroup = mddBDIGroup[mddGroup.notna()]
                mddGroup = mddGroup[mddBDIGroup.notna()]
                mddBDIGroup = mddBDIGroup[mddBDIGroup.notna()]
                if len(mddGroup) > 2:
                    nData = len(mddGroup)
                    # rStat, pVal = scipy_stats.pearsonr(mddBDIGroup, mddGroup)
                    permResult = scipy_stats.permutation_test((mddBDIGroup, mddGroup), corrTest, n_resamples=10000, permutation_type='pairings', random_state=12345)
                    rStat = permResult.statistic
                    pVal = permResult.pvalue
                    print(pVal)
                    data = [hcGroup, mddGroup]
                    # data = [hcGroup, mddGroup]
                    # if pVal < 0.005:
                    # print(scipy_stats.iqr(preprocessedData[i]), np.mean(preprocessedData[i]))
                    fig, ax = plt.subplots()
                    fig.set_figwidth(figWidth)
                    fig.set_figheight(figHeight)
                    ax.scatter(mddBDIGroup, mddGroup, label=f'r={rStat}')
                    # xRange, yRange = np.polyfit(mddBDIGroup.to_numpy(), mddGroup.to_numpy(), deg=1)
                    xRange, yRange = calculateLeastSquaresRegression(mddBDIGroup.to_numpy(), mddGroup.to_numpy())
                    # y_err = scipy_stats.sem(mddBDIGroup.to_numpy())
                    y_err = xRange.std() * np.sqrt(1/len(xRange) +
                          (xRange - xRange.mean())**2 / np.sum((xRange - xRange.mean())**2))
                    ax.plot(xRange, yRange, label=f'p={pVal}')
                    #plt.fill_between(xRange, yRange - y_err, yRange + y_err, alpha=0.5)
                    # plot bar graph showing mean of HC and MDD with SEM error bars
                    # ax.bar(['HC', 'MDD'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], width=w)
                    ax.set_xlabel(columnHeadersAndFilenames[bdi]['axisLabel'])
                    ax.set_ylabel(columnHeadersAndFilenames[actColumn]['axisLabel'])
                    plt.legend()
                    plt.title(f"{columnHeadersAndFilenames[actColumn]['figureTitle']}\ncorrelation with {bdi}\nIn MDD, N={nData}")
                    plt.show()
                    if pVal < desiredPval:
                        plt.savefig(os.path.join(derivatives,'permutationTesting','sigFigures',f'{columnHeadersAndFilenames[actColumn]["filename"]}_pearson_r_{columnHeadersAndFilenames[bdi]["filename"]}.svg'))
                    plt.savefig(os.path.join(derivatives,'permutationTesting','BDICorrelations',f'{columnHeadersAndFilenames[actColumn]["filename"]}_pearson_r_{columnHeadersAndFilenames[bdi]["filename"]}.svg'))
                    plt.close()

#%% run correlation between cytokins and cell count

cellCountColumns = ["Singlets per µl", "Granulocytes per µl", "Neutrophils per µl", "Monocytes per µl", "Classical per µl", "Intermediate per µl", "Nonclassical per µl","Tcells per µl", "Thelpersper µl", "NKT per µl", "NK per µl", "Bcell per µl", "Cytotoxic T per µl"]

for cellCount in cellCountColumns:
    for actColumn in preprocessedData.columns:
        if actColumn not in ignoreColumns:
            nChecked += 1
            if actColumn not in cellCountColumns:
                
                upperLimit = preprocessedData[actColumn].quantile(0.9)# + iqr * 1.5
                lowerLimit = preprocessedData[actColumn].quantile(0.1)# - iqr * 1.5
                hcGroup = preprocessedData[actColumn].loc[hcIdx]
                preOutlierLength = len(hcGroup)
                # hcGroup = hcGroup[hcGroup < upperLimit]
                # print(preOutlierLength, len(hcGroup))
                # hcGroup = np.array(hcGroup > lowerLimit)
                mddGroup = preprocessedData[actColumn].loc[mddIdx]
                mddCellCountGroup = preprocessedData[cellCount].loc[mddIdx]
                # print(upperLimit, lowerLimit, np.mean(mddGroup))
                outlierUpperIdx = mddGroup < upperLimit
                mddGroup = mddGroup.loc[outlierUpperIdx]
                mddCellCountGroup = mddCellCountGroup.loc[outlierUpperIdx]
                outlierLowerIdx = mddGroup > lowerLimit
                mddGroup = mddGroup.loc[outlierLowerIdx]
                mddCellCountGroup = mddCellCountGroup.loc[outlierLowerIdx]
                mddGroup = mddGroup[mddGroup.notna()]
                mddCellCountGroup = mddCellCountGroup[mddGroup.notna()]
                mddGroup = mddGroup[mddCellCountGroup.notna()]
                mddCellCountGroup = mddCellCountGroup[mddCellCountGroup.notna()]
                if len(mddGroup) > 2:
                    nData = len(mddGroup)
                    # rStat, pVal = scipy_stats.pearsonr(mddBDIGroup, mddGroup)
                    permResult = scipy_stats.permutation_test((mddCellCountGroup, mddGroup), corrTest, n_resamples=10000, permutation_type='pairings', random_state=12345)
                    rStat = permResult.statistic
                    pVal = permResult.pvalue
                    print(pVal)
                    data = [hcGroup, mddGroup]
                    # data = [hcGroup, mddGroup]
                    # if pVal < 0.005:
                    # print(scipy_stats.iqr(preprocessedData[i]), np.mean(preprocessedData[i]))
                    fig, ax = plt.subplots()
                    fig.set_figwidth(figWidth)
                    fig.set_figheight(figHeight)
                    ax.scatter(mddCellCountGroup, mddGroup, label=f'r={rStat}')
                    # xRange, yRange = np.polyfit(mddBDIGroup.to_numpy(), mddGroup.to_numpy(), deg=1)
                    xRange, yRange = calculateLeastSquaresRegression(mddCellCountGroup.to_numpy(), mddGroup.to_numpy())
                    # y_err = scipy_stats.sem(mddBDIGroup.to_numpy())
                    y_err = xRange.std() * np.sqrt(1/len(xRange) +
                          (xRange - xRange.mean())**2 / np.sum((xRange - xRange.mean())**2))
                    ax.plot(xRange, yRange, label=f'p={pVal}')
                    #plt.fill_between(xRange, yRange - y_err, yRange + y_err, alpha=0.5)
                    # plot bar graph showing mean of HC and MDD with SEM error bars
                    # ax.bar(['HC', 'MDD'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], width=w)
                    ax.set_xlabel(columnHeadersAndFilenames[cellCount]['axisLabel'])
                    ax.set_ylabel(columnHeadersAndFilenames[actColumn]['axisLabel'])
                    plt.legend()
                    plt.title(f"{columnHeadersAndFilenames[actColumn]['figureTitle']}\ncorrelation with {cellCount}\nIn MDD, N={nData}")
                    plt.show()
                    if pVal < desiredPval:
                        plt.savefig(os.path.join(derivatives,'permutationTesting','sigFigures',f'{columnHeadersAndFilenames[actColumn]["filename"]}_pearson_r_{columnHeadersAndFilenames[cellCount]["filename"]}.svg'))
                    plt.savefig(os.path.join(derivatives,'permutationTesting','cellCountCorrelations',f'{columnHeadersAndFilenames[actColumn]["filename"]}_pearson_r_{columnHeadersAndFilenames[cellCount]["filename"]}.svg'))
                    plt.close()
#%% create function for running correlation matrices
def calculateCorrelationMatrices(columnList):
    corrMatrixHC = np.zeros([len(columnList), len(columnList)])
    corrMatrixMDD = np.zeros([len(columnList), len(columnList)])
    pValMatrixHC = np.zeros([len(columnList), len(columnList)])
    pValMatrixMDD = np.zeros([len(columnList), len(columnList)])
    iIdx = 0
    for i in enumerate(columnList):
        jIdx = 0
        for j in enumerate(columnList):
            truncDF = preprocessedData.loc[preprocessedData[i[1]].notnull() & preprocessedData[j[1]].notnull()]
            hcGroup = truncDF.loc[truncDF['Study'] == 'HC-MDD']
            hcGroupColumnA = np.array(hcGroup[i[1]])
            mddGroup = truncDF.loc[truncDF['Study'] == 'MDD']
            mddGroupColumnA = np.array(mddGroup[i[1]])
            hcGroupColumnB = np.array(hcGroup[j[1]])
            mddGroupColumnB = np.array(np.array(mddGroup[j[1]]))
            if len(hcGroupColumnA) > 1 and len(hcGroupColumnB) > 1 and len(mddGroupColumnA) > 1 and len(mddGroupColumnB) > 1:
                rStatHC, pValHC = scipy_stats.pearsonr(hcGroupColumnA, hcGroupColumnB)
                rStatMDD, pValMDD = scipy_stats.pearsonr(mddGroupColumnA, mddGroupColumnB)
                corrMatrixHC[iIdx,jIdx] = rStatHC
                corrMatrixMDD[iIdx,jIdx] = rStatMDD
                pValMatrixHC[iIdx,jIdx] = pValHC
                pValMatrixMDD[iIdx,jIdx] = pValMDD
            jIdx += 1
        iIdx += 1
    plt.close('all')
    # fig, ax = plt.subplots()
    ax = sns.heatmap(corrMatrixHC, vmin=-1, vmax=1, cmap='vlag', annot=True)
    plt.title('Correlation in HC')
    ax.set_xticks(range(len(columnList)))
    ax.set_yticks(range(len(columnList)))
    ax.set_xticklabels(columnList, rotation=45)
    ax.set_yticklabels(columnList, rotation=0)
    # for i in range(len(group01)):
    #     for j in range(len(group01)):
    #         text = ax.text(j, i, corrMatrixHC[i, j], color="k")
    
    fig, ax = plt.subplots()
    sns.heatmap(corrMatrixMDD, vmin=-1, vmax=1, cmap='vlag', annot=True)
    plt.title('Correlation in MDD')
    ax.set_xticks(range(len(columnList)))
    ax.set_yticks(range(len(columnList)))
    ax.set_xticklabels(columnList, rotation=45)
    ax.set_yticklabels(columnList, rotation=0)
    
    diffMap = corrMatrixHC - corrMatrixMDD
    fig, ax = plt.subplots()
    sns.heatmap(diffMap, vmin=-1, vmax=1, cmap='vlag', annot=True)
    plt.title('Difference between correlation, HC - MDD')
    ax.set_xticks(range(len(columnList)))
    ax.set_yticks(range(len(columnList)))
    ax.set_xticklabels(columnList, rotation=45)
    ax.set_yticklabels(columnList, rotation=0)

#%% group 1
group01 = ['Singlets per µl',	'Granulocytes per µl',	'Neutrophils per µl'	,'Monocytes per µl',	'Classical per µl',	'Intermediate per µl',	'Nonclassical per µl']
calculateCorrelationMatrices(group01)
#%% group 2
group02 = ['Tcells per µl','Thelpersper µl',	'NKT per µl','NK per µl','Bcell per µl',	'Cytotoxic T per µl']
calculateCorrelationMatrices(group02)
#%% group 3
group03 = ['ClassicalMFICCR2',	'ClassicalMFICD15',	'ClassicalMFICD36',	'ClassicalMFICD62L',	'ClassicalMFICD86',	'ClassicalMFICD163',	'ClassicalMFICX3CR1',	'ClassicalMFIHLAABC',	'IntMFICCR2',	'IntMFICD15',	'IntMFICD36',	'IntMFICD62L',	'IntMFICD86',	'IntMFICD163',	'IntMFICX3CR1',	'IntMFIHLAABC',	'NonclassicalMFICCR2',	'NonclassicalMFICD15',	'NonclassicalMFICD36',	'NonclassicalMFICD62L',	'NonclassicalMFICD86',	'NonclassicalMFICD163',	'NonclassicalMFICX3CR1',	'NonclassicalMFIHLAABC',	'HLAABCCountNonclassical']
calculateCorrelationMatrices(group03)

#%% group 4
group04 = ['ICBAIL-1bConc',	'ICBAIFN-a2Conc',	'ICBAIFN-gConc',	'ICBATNF-aConc',	'ICBAMCP-1Conc',	'ICBAIL-6Conc',	'ICBAIL-8Conc',	'ICBAIL-10Conc',	'ICBAIL-12p70Conc',	'ICBAIL-17AConc',	'ICBAIL-18Conc',	'ICBAIL-23Conc',	'ICBAIL-33Conc',	'NCBAVILIP-1Conc',	'NCBAMCP-1Conc',	'NCBAsTREM-2Conc',	'NCBABDNFConc',	'NCBATGF-b1(FreeActive)Conc',	'NCBAVEGFConc',	'NCBAIL-6Conc',	'NCBAsTREM-1Conc',	'NCBAb-NGFConc',	'NCBAIL-18Conc',	'NCBATNF-aConc',	'NCBAsRAGEConc',	'NCBACX3CL1Conc']
calculateCorrelationMatrices(group04)