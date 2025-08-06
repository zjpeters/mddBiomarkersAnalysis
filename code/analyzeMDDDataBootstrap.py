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
preprocessedData = pd.read_csv(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','rawdata','preprocessed_MDD_data.csv'), index_col=0)
columnHeadersAndFilenames= json.load(open(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','code','columnHeadersAndFileNames.json')))
plt.style.use('seaborn-v0_8-colorblind')

# palette from https://www.color-hex.com/color-palette/49436
fiveColorPalette = []
fiveColorPalette.append(np.array([213 / 255, 94 / 255,0 / 255]))
fiveColorPalette.append(np.array([204 / 255,121 / 255,167 / 255]))
fiveColorPalette.append(np.array([0 / 255, 114 / 255, 178 / 255]))
fiveColorPalette.append(np.array([240 / 255, 228 / 255, 66 / 255]))
fiveColorPalette.append(np.array([0 / 255, 158 / 255, 115 / 255]))
fiveColorPalette= np.array(fiveColorPalette)

# palette from https://projects.susielu.com/viz-palette?colors=[%22#ffd700%22,%22#ffb14e%22,%22#fa8775%22,%22#ea5f94%22,%22#cd34b5%22,%22#9d02d7%22,%22#0000ff%22]&backgroundColor=%22white%22&fontColor=%22black%22&mode=%22normal%22
sevenColorPalette = []
sevenColorPalette.append("#0000ff") # indigo
sevenColorPalette.append("#9d02d7") # dark magenta
sevenColorPalette.append("#cd34b5") # light magenta
sevenColorPalette.append("#ea5f94") # pink
sevenColorPalette.append("#fa8775") # light orange
sevenColorPalette.append("#ffb14e") # orange
sevenColorPalette.append("#ffd700") # gold
sevenColorPalette = np.array(sevenColorPalette)

colorBlindColorPalette = []
colorBlindColorPalette.append("#f05039")
colorBlindColorPalette.append("#e57a77")
colorBlindColorPalette.append("#eebab4")
colorBlindColorPalette.append("#1f449c")
colorBlindColorPalette.append("#3d65a5")
colorBlindColorPalette.append("#7ca1cc")
colorBlindColorPalette.append("#a8b6cc")
colorBlindColorPalette = np.array(colorBlindColorPalette)
def displayColorPalettes(colorPalette):
    paletteSize = len(colorPalette)
    paletteImage = np.zeros([paletteSize, 1])
    for i in range(paletteSize):
        paletteImage[i,:] = i+1
    plt.figure()
    plt.imshow(paletteImage,cmap=ListedColormap(colorPalette))
    plt.show()
# displayColorPalettes(sevenColorPalette)
# displayColorPalettes(fiveColorPalette)
# displayColorPalettes(colorBlindColorPalette)

#%% begin processing data
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

"""
Notes:
for bootstrapping, will ignore outliers and instead treat them as potential true variation
"""
nResample = 1000

alphaFdr = desiredPval/(nResample)
for i in preprocessedData.columns:
    if i not in ignoreColumns:
        truncDF = preprocessedData.loc[preprocessedData[i].notnull()]
        upperLimit = truncDF[i].quantile(0.9)
        lowerLimit = truncDF[i].quantile(0.1)
        # truncDF = truncDF.loc[truncDF[i] > lowerLimit]
        # truncDF = truncDF.loc[truncDF[i] < upperLimit]
        hcGroup = truncDF[i].loc[truncDF['Study'] == 'HC-MDD']
        mddGroup = truncDF[i].loc[truncDF['Study'] == 'MDD']
        if any(hcGroup) and any(mddGroup):
            hcRandIdx = np.arange(0, len(hcGroup))
            mddRandIdx = np.arange(0, len(mddGroup))
            hcBootstrap = np.zeros(nResample)
            mddBootstrap = np.zeros(nResample)
            
            for n in range(nResample):
                hcBootstrapIdx = np.random.choice(hcRandIdx, size=len(hcGroup), replace=True)
                mddBootstrapIdx = np.random.choice(mddRandIdx, size=len(mddGroup), replace=True)
                hcBootstrap[n] = np.mean(np.array(hcGroup)[hcBootstrapIdx])
                mddBootstrap[n] = np.mean(np.array(mddGroup)[mddBootstrapIdx])
            # rStat, pVal = scipy_stats.pearsonr(preprocessedData['group'], preprocessedData[i])
            mu_A = np.sum(hcBootstrap, axis=0)/nResample
            mu_B = np.sum(mddBootstrap, axis=0)/nResample
            cov_A = np.cov(hcBootstrap)
            cov_B = np.cov(mddBootstrap)
            
            sig_2_A = np.sum((hcBootstrap - mu_A)**2)/(nResample - 1)
            sig_2_B = np.sum((mddBootstrap - mu_B)**2)/(nResample - 1)
            se_A = np.sqrt(sig_2_A/nResample)
            se_B = np.sqrt(sig_2_B/nResample)
            # test null hypothesis
            t_xy = (mu_A - mu_B)/np.sqrt(se_A**2 + se_B**2)
            # get p values
            dof = nResample-1
            p_xy = 2*(1 - scipy_stats.t.cdf(t_xy, dof))
            tStat, pVal = scipy_stats.ttest_ind(hcBootstrap, mddBootstrap)
            print(t_xy, tStat)
            data = [hcBootstrap, mddBootstrap]
            nChecked += 1
            # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
            fig, ax = plt.subplots()
            fig.set_figwidth(figWidth)
            fig.set_figheight(figHeight)
            n_hc = len(hcGroup)
            n_mdd = len(mddGroup)
            n_outliers = len(preprocessedData[i]) - (n_hc + n_mdd)
            # plot bar graph showing mean of HC and MDD with SEM error bars
            ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcBootstrap), np.mean(mddBootstrap)], yerr=[scipy_stats.sem(hcBootstrap), scipy_stats.sem(mddBootstrap)], capsize=3, width=w, label=f'p={p_xy}', color=[colorBlindColorPalette[0],colorBlindColorPalette[3]])
            ax.set_ylabel(columnHeadersAndFilenames[i]['axisLabel'])
            for j in range(len(groups)):
                # distribute scatter randomly across whole width of bar
                if j == 0:
                    colorSelect = 2
                else:
                    colorSelect = 5
                ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorBlindColorPalette[colorSelect])
            plt.title(columnHeadersAndFilenames[i]['figureTitle'])
            plt.legend()
            if p_xy < alphaFdr:
                x1, x2 = 0, 1
                y, h, col = max(map(max, [hcGroup, mddGroup])) + 2, 2, 'k'
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','bootstrap','sigFigures',f'{columnHeadersAndFilenames[i]["filename"]}_ttest_for_HCvMDD_bootstrap.svg'))
            plt.show()
            plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','bootstrap',f'{columnHeadersAndFilenames[i]["filename"]}_ttest_for_HCvMDD_bootstrap.svg'))
            plt.close()

#%% permutation testing
def diffOfMeans(x,y):
    diff = np.mean(x) - np.mean(y)
    return diff

def ttest(A,B):
    mu_A = np.sum(A, axis=0)/nResample
    mu_B = np.sum(B, axis=0)/nResample
    
    sig_2_A = np.sum((A - mu_A)**2)/(nResample - 1)
    sig_2_B = np.sum((B - mu_B)**2)/(nResample - 1)
    se_A = np.sqrt(sig_2_A/nResample)
    se_B = np.sqrt(sig_2_B/nResample)
    # test null hypothesis
    tStat = (mu_A - mu_B)/np.sqrt(se_A**2 + se_B**2)
    return tStat
for i in preprocessedData.columns:
    if i not in ignoreColumns:
        truncDF = preprocessedData.loc[preprocessedData[i].notnull()]
        upperLimit = truncDF[i].quantile(0.9)
        lowerLimit = truncDF[i].quantile(0.1)
        truncDF = truncDF.loc[truncDF[i] > lowerLimit]
        truncDF = truncDF.loc[truncDF[i] < upperLimit]
        hcGroup = truncDF[i].loc[truncDF['Study'] == 'HC-MDD']
        mddGroup = truncDF[i].loc[truncDF['Study'] == 'MDD']
        if len(hcGroup) > 1 and len(mddGroup) > 1:
            permResult = scipy_stats.permutation_test((hcGroup, mddGroup), ttest, n_resamples=10000, random_state=12345)
            tStat = permResult.statistic
            pVal = permResult.pvalue
            print(pVal)
            data = [hcGroup, mddGroup]
            nChecked += 1
            # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
            fig, ax = plt.subplots()
            fig.set_figwidth(figWidth)
            fig.set_figheight(figHeight)
            n_hc = len(hcGroup)
            n_mdd = len(mddGroup)
            n_outliers = len(preprocessedData[i]) - (n_hc + n_mdd)
            # plot bar graph showing mean of HC and MDD with SEM error bars
            ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorBlindColorPalette[0],colorBlindColorPalette[3]])
            ax.set_ylabel(columnHeadersAndFilenames[i]['axisLabel'])
            for j in range(len(groups)):
                # distribute scatter randomly across whole width of bar
                if j == 0:
                    colorSelect = 2
                else:
                    colorSelect = 5
                ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorBlindColorPalette[colorSelect])
            plt.title(columnHeadersAndFilenames[i]['figureTitle'])
            plt.legend()
            if pVal < desiredPval:
                x1, x2 = 0, 1
                y, h, col = max(map(max, [hcGroup, mddGroup])) + 2, 2, 'k'
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','permutationTesting','sigFigures',f'{columnHeadersAndFilenames[i]["filename"]}_ttest_for_HCvMDD_bootstrap.svg'))
            plt.show()
            plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','permutationTesting',f'{columnHeadersAndFilenames[i]["filename"]}_ttest_for_HCvMDD_bootstrap.svg'))
            plt.close()


# for i in preprocessedData.columns:
#     if i not in ignoreColumns:
#         dataDescription = preprocessedData[i].describe()
#         # iqr = preprocessedData[i].quantile(0.75) - preprocessedData[i].quantile(0.25)
#         # exclude values outside of the interquartile range
#         upperLimit = preprocessedData[i].quantile(0.75)# + iqr * 1.5
#         lowerLimit = preprocessedData[i].quantile(0.25)# - iqr * 1.5
#         hcGroup = preprocessedData[i].loc[hcIdx]
#         preOutlierLength = len(hcGroup)
#         hcGroup = hcGroup[hcGroup < upperLimit]
#         print(preOutlierLength, len(hcGroup))
#         hcGroup = np.array(hcGroup > lowerLimit)
#         mddGroup = preprocessedData[i].loc[mddIdx]
#         mddGroup = mddGroup[mddGroup < upperLimit]
#         mddGroup = mddGroup[mddGroup > lowerLimit]
#         if any(hcGroup) and any(mddGroup):
#             # rStat, pVal = scipy_stats.pearsonr(preprocessedData['group'], preprocessedData[i])
#             tStat, pVal = scipy_stats.ttest_ind(hcGroup, mddGroup)
#             data = [hcGroup, mddGroup]
#             nChecked += 1
#             # print(lowerLimit, upperLimit, np.max(preprocessedData[i]))
#             fig, ax = plt.subplots()
#             fig.set_figwidth(figWidth)
#             fig.set_figheight(figHeight)
#             n_hc = len(hcGroup)
#             n_mdd = len(mddGroup)
#             n_outliers = len(preprocessedData[i]) - (n_hc + n_mdd)
#             # plot bar graph showing mean of HC and MDD with SEM error bars
#             ax.bar([f'HC, N={n_hc}', f'MDD, N={n_mdd}'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[scipy_stats.sem(hcGroup), scipy_stats.sem(mddGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorBlindColorPalette[0],colorBlindColorPalette[3]])
#             ax.set_ylabel(columnHeadersAndFilenames[i]['axisLabel'])
#             for j in range(len(groups)):
#                 # distribute scatter randomly across whole width of bar
#                 if j == 0:
#                     colorSelect = 2
#                 else:
#                     colorSelect = 5
#                 ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j], c=colorBlindColorPalette[colorSelect])
#             plt.title(columnHeadersAndFilenames[i]['figureTitle'])
#             plt.legend()
#             if pVal < alphaFdr:
#                 x1, x2 = 0, 1
#                 y, h, col = max(map(max, [hcGroup, mddGroup])) + 2, 2, 'k'
#                 plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#                 plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','noNaN','sigFigures',f'{columnHeadersAndFilenames[i]["filename"]}_ttest_for_HCvMDD.png'))
#             plt.show()
#             plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','noNaN',f'{columnHeadersAndFilenames[i]["filename"]}_ttest_for_HCvMDD.png'))
#             plt.close()
        # else:
            # chiSquare, pval = scipy_stats.chisquare(hcGroup, mddGroup)
            # if pVal < alphaFdr:
            #     fig, ax = plt.subplots()
            #     ax.bar(['HC', 'MDD'], [np.mean(hcGroup), np.mean(mddGroup)], yerr=[np.std(hcGroup), np.std(mddGroup)])
            #     for j in range(len(groups)):
            #         # distribute scatter randomly across whole width of bar
            #         ax.scatter(groups[j] + np.random.random(data[j].size) * w - w / 2, data[j])
            #     plt.title(i)
            #     plt.show()

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
    for i in preprocessedData.columns:
        if i not in ignoreColumns:
            nChecked += 1
            if i not in bdiColumns:
                
                upperLimit = preprocessedData[i].quantile(0.9)# + iqr * 1.5
                lowerLimit = preprocessedData[i].quantile(0.1)# - iqr * 1.5
                hcGroup = preprocessedData[i].loc[hcIdx]
                preOutlierLength = len(hcGroup)
                # hcGroup = hcGroup[hcGroup < upperLimit]
                # print(preOutlierLength, len(hcGroup))
                # hcGroup = np.array(hcGroup > lowerLimit)
                mddGroup = preprocessedData[i].loc[mddIdx]
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
                    ax.set_ylabel(columnHeadersAndFilenames[i]['axisLabel'])
                    plt.legend()
                    plt.title(f"{columnHeadersAndFilenames[i]['figureTitle']}\ncorrelation with {bdi}\nIn MDD, N={nData}")
                    plt.show()
                    if pVal < desiredPval:
                        plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','permutationTesting','sigFigures',f'{columnHeadersAndFilenames[i]["filename"]}_pearson_r_{columnHeadersAndFilenames[bdi]["filename"]}.svg'))
                    plt.savefig(os.path.join('/','home','zjpeters','Documents','otherLabs','agarza','derivatives','permutationTesting','BDICorrelations',f'{columnHeadersAndFilenames[i]["filename"]}_pearson_r_{columnHeadersAndFilenames[bdi]["filename"]}.svg'))
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
#%% process the correlation of all columns
# nChecked = len(preprocessedData.columns) - len(ignoreColumns) + 1
# corrMatrixHC = np.zeros([nChecked + 1, nChecked + 1])
# corrMatrixMDD = np.zeros([nChecked + 1, nChecked + 1])
# pValMatrixHC = np.zeros([nChecked + 1, nChecked + 1])
# pValMatrixMDD = np.zeros([nChecked + 1, nChecked + 1])
# iIdx = 0
# for i in enumerate(preprocessedData.columns):
#     if i[1] not in ignoreColumns:
#         hcGroup = preprocessedData[i[1]].loc[hcIdx]
#         mddGroup = preprocessedData[i[1]].loc[mddIdx]
#         mddGroup = mddGroup[mddGroup.notna()]
#         hcGroup = hcGroup[hcGroup.notna()]
#         hcGroupColumnA = np.array(hcGroup)
#         mddGroupColumnA = np.array(mddGroup)
#         # data = [hcGroup, mddGroup]
#         jIdx = 0
#         for j in enumerate(preprocessedData.columns):
#             if j[1] not in ignoreColumns:
#                 hcGroup = preprocessedData[j[1]].loc[hcIdx]
#                 mddGroup = preprocessedData[j[1]].loc[mddIdx]
#                 mddGroup = mddGroup[mddGroup.notna()]
#                 hcGroup = hcGroup[hcGroup.notna()]
#                 hcGroupColumnB = np.array(hcGroup)
#                 mddGroupColumnB = np.array(mddGroup)
#                 if len(hcGroupColumnA) > 1 and len(hcGroupColumnB) > 1 and len(mddGroupColumnA) > 1 and len(mddGroupColumnB) > 1:
#                     rStatHC, pValHC = scipy_stats.pearsonr(hcGroupColumnA, hcGroupColumnB)
#                     rStatMDD, pValMDD = scipy_stats.pearsonr(mddGroupColumnA, mddGroupColumnB)
#                     corrMatrixHC[iIdx,jIdx] = rStatHC
#                     corrMatrixMDD[iIdx,jIdx] = rStatMDD
#                     pValMatrixHC[iIdx,jIdx] = pValHC
#                     pValMatrixMDD[iIdx,jIdx] = pValMDD
#                 jIdx += 1
#         iIdx += 1

# #%% plotting heatmaps of correlation
# plt.close('all')
# plt.figure()
# sns.heatmap(corrMatrixHC, vmin=-1, vmax=1, cmap='vlag')
# plt.title("Correlation in HC")
# plt.figure()
# sns.heatmap(corrMatrixMDD, vmin=-1, vmax=1, cmap='vlag')
# plt.title("Correlation in MDD")
# diffMap = corrMatrixHC - corrMatrixMDD
# plt.figure()
# sns.heatmap(diffMap, vmin=-1, vmax=1, cmap='vlag')
# plt.title("Difference between HC and MDD (HC > MDD)")
#%% setup groups to use for correlation matrices

# group01 = ['Singlets per µl',	'Granulocytes per µl',	'Neutrophils per µl'	,'Monocytes per µl',	'Classical per µl',	'Intermediate per µl',	'Nonclassical per µl']
# corrMatrixHC = np.zeros([len(group01), len(group01)])
# corrMatrixMDD = np.zeros([len(group01), len(group01)])
# pValMatrixHC = np.zeros([len(group01), len(group01)])
# pValMatrixMDD = np.zeros([len(group01), len(group01)])
# iIdx = 0
# for i in enumerate(group01):
#     hcGroup = preprocessedData[i[1]].loc[hcIdx]
#     mddGroup = preprocessedData[i[1]].loc[mddIdx]
#     mddGroup = mddGroup[mddGroup.notna()]
#     hcGroup = hcGroup[hcGroup.notna()]
#     hcGroupColumnA = np.array(hcGroup)
#     mddGroupColumnA = np.array(mddGroup)
#     # data = [hcGroup, mddGroup]
#     jIdx = 0
#     for j in enumerate(group01):
#         hcGroup = preprocessedData[j[1]].loc[hcIdx]
#         mddGroup = preprocessedData[j[1]].loc[mddIdx]
#         mddGroup = mddGroup[mddGroup.notna()]
#         hcGroup = hcGroup[hcGroup.notna()]
#         hcGroupColumnB = np.array(hcGroup)
#         mddGroupColumnB = np.array(mddGroup)
#         if len(hcGroupColumnA) > 1 and len(hcGroupColumnB) > 1 and len(mddGroupColumnA) > 1 and len(mddGroupColumnB) > 1:
#             rStatHC, pValHC = scipy_stats.pearsonr(hcGroupColumnA, hcGroupColumnB)
#             rStatMDD, pValMDD = scipy_stats.pearsonr(mddGroupColumnA, mddGroupColumnB)
#             corrMatrixHC[iIdx,jIdx] = rStatHC
#             corrMatrixMDD[iIdx,jIdx] = rStatMDD
#             pValMatrixHC[iIdx,jIdx] = pValHC
#             pValMatrixMDD[iIdx,jIdx] = pValMDD
#         jIdx += 1
#     iIdx += 1

# #%% plotting heatmaps of correlation
# plt.close('all')
# # fig, ax = plt.subplots()
# ax = sns.heatmap(corrMatrixHC, vmin=-1, vmax=1, cmap='vlag', annot=True)
# plt.title('Correlation in HC')
# ax.set_xticklabels(group01, rotation=45)
# ax.set_yticklabels(group01, rotation=45)
# # for i in range(len(group01)):
# #     for j in range(len(group01)):
# #         text = ax.text(j, i, corrMatrixHC[i, j], color="k")

# fig, ax = plt.subplots()
# sns.heatmap(corrMatrixMDD, vmin=-1, vmax=1, cmap='vlag')
# plt.title('Correlation in MDD')
# ax.set_xticklabels(group01, rotation=45)
# ax.set_yticklabels(group01, rotation=45)

# diffMap = corrMatrixHC - corrMatrixMDD
# fig, ax = plt.subplots()
# sns.heatmap(diffMap, vmin=-1, vmax=1, cmap='vlag')
# plt.title('Difference between correlation, HC - MDD')
# ax.set_xticklabels(group01, rotation=45)
# ax.set_yticklabels(group01, rotation=45)

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