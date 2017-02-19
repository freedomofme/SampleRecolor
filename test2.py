# -*- coding: utf-8 -*-

import brewer2mpl
from numpy import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch


# helper for cleaning up axes by removing ticks, tick labels, frame, etc.
def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# my_data = genfromtxt('./data.csv', delimiter=',')
# print my_data


# test data
testL = []
# 5 samples from one group
for i in range(5):
    # 20 measurements from normal with mean 10, stdev 2
    testL.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

# 8 samples from another group
for i in range(8):
    # 20 measurements from normal with mean 4, stdev 4
    testL.append(random.normal(4,4,20))

# testL = my_data

# permute test data and make dataframe not random
testA = array(testL)[random.permutation(range(len(testL)))]
# print testA
testDF = pd.DataFrame(testA)
# print testDF

print testDF.shape

#线宽
matplotlib.rcParams['lines.linewidth'] = 0.3

# testDF.index = row
# testDF.columns = columns
# x和y轴的标签名称
testDF.index = [ 'Sample ' + str(x) for x in testDF.index ]
testDF.columns = [ 'c' + str(x) for x in testDF.columns ]

# look at raw data ，原始图
axi = plt.imshow(testA,interpolation='nearest', cmap=plt.cm.RdBu)
plt.savefig('abc.png')


pairwise_dists = distance.squareform(distance.pdist(testDF))
print pairwise_dists

clusters = sch.linkage(pairwise_dists,method='complete')
print clusters

plt.clf()


den = sch.dendrogram(clusters)
print den['leaves']
# plt.show()

plt.clf()
axi = plt.imshow(testDF.ix[den['leaves']],interpolation='nearest',cmap=plt.cm.RdBu)
plt.savefig('abc2.png')

# ----------


fig = plt.figure()
heatmapGS = gridspec.GridSpec(1,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1])

### row dendrogram ###
denAX = fig.add_subplot(heatmapGS[0,0])
sch.set_link_color_palette(['black'])
denD = sch.dendrogram(clusters,color_threshold=inf,orientation='left')
clean_axis(denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[0,1])
axi = heatmapAX.imshow(testDF.ix[den['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=plt.cm.RdBu)
clean_axis(heatmapAX)
plt.savefig('abc3.png')

# ------
plt.clf()

# rename row clusters
row_clusters = clusters

# calculate pairwise distances for columns
col_pairwise_dists = distance.squareform(distance.pdist(testDF.T))
# cluster
col_clusters = sch.linkage(col_pairwise_dists,method='complete')

# plot the results
fig = plt.figure()
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ####
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=inf,orientation='left')
clean_axis(row_denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[den['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=plt.cm.RdBu)
clean_axis(heatmapAX)

fig.tight_layout()
plt.savefig('abc4.png')

# ----------


# heatmap with row names
fig = plt.figure(figsize=(5,8))
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.20,1],height_ratios=[0.20,1])

### col dendrogram ###
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=inf,orientation='left')
clean_axis(row_denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[1,1])

axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=plt.cm.coolwarm)

plt.grid(ls='solid')
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(arange(testDF.shape[0]) -.5)
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']], fontsize = 6)

## col labels ##
## -.5 使得grid line偏移到正确位置
heatmapAX.set_xticks(arange(testDF.shape[1]) -.5)
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']], fontsize = 8)
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines():
    l.set_markersize(0)


### scale colorbar ###
scale_cbGSSS = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=heatmapGS[0,0],wspace=0.0,hspace=0.0)
scale_cbAX = fig.add_subplot(scale_cbGSSS[0,1]) # colorbar for scale in upper left corner
cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
cb.set_label('Measurements')
cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
cb.outline.set_linewidth(0)
# make colorbar labels smaller
tickL = cb.ax.yaxis.get_ticklabels()
for t in tickL:
    t.set_fontsize(t.get_fontsize() - 3)


fig.tight_layout()

plt.savefig('abc5.png',dpi = 200)




