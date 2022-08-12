import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def plot_image_outlines(images, edgecolor='k', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[6,6])
        noax = True
    else:
        noax = False

    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]

    for im, dat in images.items():
        if im == 'Data':
            continue
        size = dat['Size']
        center = dat['Center']
        
        xy = center - size / 2
        rect = patches.Rectangle(xy, *size, linewidth=1, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)
        
        if xy[0] < xlim[0]:
            xlim[0] = xy[0]
        if xy[0] + size[0] > xlim[1]:
            xlim[1] = xy[0] + size[0]
        
        if xy[1] < ylim[0]:
            ylim[0] = xy[1]
        if xy[1] + size[1] > ylim[1]:
            ylim[1] = xy[1] + size[1]

    xlim = xlim + np.ptp(xlim) * np.array([-0.05, 0.05])
    ylim = ylim + np.ptp(ylim) * np.array([-0.05, 0.05])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel('$\mu m$')
    ax.set_ylabel('$\mu m$')
    
    ax.invert_yaxis()
    
    ax.set_aspect(1)
    
    if noax:
        fig.set_size_inches(np.ptp(xlim) * 1e-3 * 0.1, np.ptp(ylim) * 1e-3 * 0.1)
    
    return ax.get_figure(), ax


def plot_image(image, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[6,6])
    
    if 'extent' in image:
        ax.imshow(image['data'], extent=image['extent'])
    else:
        for k, im in image.items():
            if k == 'Data':
                continue
            ax.imshow(im['data'], extent=im['extent'])
    
    return ax.get_figure(), ax

def in_axes(x, y, ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    if min(x) >= xlim[0] and max(x) <= xlim[1] and min(y) >= ylim[0] and max(y) <= ylim[1]:
        return True
    
    return False 


def plot_vertices(scan, label=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[6,6])
    
    if 'VertexList' in scan:
        x,y,z = scan['VertexList'].T
        if in_axes(x, y, ax):
            ax.scatter(x, y, lw=1, edgecolor='k', zorder=2)
            ax.plot(x, y, zorder=1, ls='--', color=(0,0,0,0.6))
            if label:
                ax.text(x[0], y[0], scan['Description'], ha='left', va='top')
    else:
        for k, s in scan.items():    
            x,y,z = s['VertexList'].T
            if in_axes(x, y, ax):
                ax.scatter(x, y, lw=1, edgecolor='k', zorder=2)
                ax.plot(x, y, zorder=1, ls='--', color=(0,0,0,0.6))
                if label:
                    ax.text(x[0], y[0], s['Description'], ha='left', va='top')

    return ax.get_figure(), ax


def plot_traces(traces, analyte, vmin=None, vmax=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[6,6])
        
    dvmin = np.inf
    dvmax = -np.inf
    for name, trace in traces.items():
        if trace[analyte].min() < dvmin:
            dvmin = trace[analyte].min()
        if trace[analyte].max() > dvmax:
            dvmax = trace[analyte].max()
    
    if vmin is None:
        vmin = dvmin
    if vmax is None:
        vmax = dvmax
    
    for name, trace in traces.items():
        ma = ax.scatter(trace['x'], trace['y'], c=trace[analyte], vmin=vmin, vmax=vmax, **kwargs)
        
    return ma