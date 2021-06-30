def CAM_iCAP(lines):
    meta = {}
    for line in lines:
        category, content = line.strip().split(':')
        meta[category] = {}
        for section in content.split(';'):
            if '=' in section:
                key, value = section.split('=')
                meta[category][key] = value
    return meta

meta_parsers = {'CAM-iCap': CAM_iCAP}