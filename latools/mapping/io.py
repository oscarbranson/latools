import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg

from glob import glob

def parse_vertices(text):
    return np.array([entry.split(',') for entry in text.split(';')], dtype=float)

def parse_ablation_settings(text):
    data = {}
    for item in text.split(';'):
        k, v = item.split('=')
        data[k] = v
    return data

def read_scancsv(f):
    scancsv = pd.read_csv(f)
    
    scans = {}

    for i, item in scancsv.iterrows():
        scans[i] = {}
        
        for col in scancsv.columns:
            data = item.loc[col]
            
            match col:
                case 'Vertex List':
                    out = parse_vertices(data)
                case 'Preablation Settings':
                    out = parse_ablation_settings(data)
                case 'Ablation Settings':
                    out = parse_ablation_settings(data)
                case 'Data':
                    if isinstance(data, str):
                        out = data.split(';')
                    else:
                        out = data
                case 'Description':
                    out = data.replace(' ', '')
                case _:
                    out = data

            scans[i][col.replace(' ', '')] = out
    
    return scans
    
    
def read_ImageMap(f):
    tree = ET.parse(f)

    root = tree.getroot()

    images = {}
    
    ImagePath = os.path.dirname(os.path.abspath('scandata/TG1_lines.ImageMap'))

    for child in root:
        # fname = child[0].text
        # if fname == image_file:
        #     break
        images[child.tag] = {}
        for item in child:
            images[child.tag][item.tag] = item.text

    
    for im, v in images.items():
        if im == 'Data':
            continue
        for par in ['Size', 'Center']:
            images[im][par] = np.array(v[par].split(','), dtype=float)
        
        center = images[im]['Center']
        size = images[im]['Size']
        images[im]['extent'] = center[0] - size[0] / 2, center[0] + size[0] / 2 , center[1] + size[1] / 2, center[1] - size[1] / 2
        
        images[im]['Filepath'] = os.path.join(ImagePath, v['Filename'])

        images[im]['data'] = mpimg.imread(images[im]['Filepath'])
        
    return images

def get_scan_by_description(description, scans):
    for k, v in scans.items():
        if v['Description'] == description:
            return v
        
    return None

def load_trace(f):
    traces = pd.read_csv(f, comment='#', index_col='Time')
    traces.dropna(how='all', inplace=True)
    traces.index = traces.index - min(traces.index)
    return traces

def load_traces(dir, scans, exclude='NIST'):
    fs = [f for f in glob('ablation_export/trace_export/*.csv') if exclude not in f]
    
    out = {}
    for f in fs:
        name = os.path.basename(f).replace('_calibrated.csv', '')
        t = load_trace(f)
        x, y, d = time_to_xyd(t, get_scan_by_description(name, scans))
        t['x'] = x
        t['y'] = y
        t['distance'] = d
        
        out[name] = t
    return out

def  time_to_xyd(traces, scaninfo, start_offset_um=None):
    # TODO: only works for 2 vertices.
    x, y, _ = scaninfo['VertexList'].T

    dx = np.diff(x)
    dy = np.diff(y)

    vertex_distance_um = (dx**2 + dy**2)**0.5
    trace_distance_um = 1e2 * traces.index.values / float(scaninfo['AblationSettings']['ScanSpeed'])

    if start_offset_um is None:
        delta_difference_um = vertex_distance_um.max() - trace_distance_um.max()
        start_offset_um = delta_difference_um / 2

    trace_distance_offset_um = trace_distance_um + start_offset_um

    f_distance = trace_distance_offset_um / vertex_distance_um
    trace_x = x[0] + dx * f_distance
    trace_y = y[0] + dy * f_distance

    return trace_x, trace_y, trace_distance_offset_um
