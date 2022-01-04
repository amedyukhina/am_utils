import os

import numpy as np
from omero.gateway import BlitzGateway

from am_utils.utils import imsave


def connect(USER, PASSWD, HOST, PORT):
    conn = BlitzGateway(USER, PASSWD, host=HOST, port=PORT)
    conn.connect()
    conn.setSecure(True)
    return conn


def load_image(img, conn):
    image = conn.getObject('Image', img.getId())
    size_z = image.getSizeZ()
    size_c = image.getSizeC()
    size_t = image.getSizeT()
    pixels = image.getPrimaryPixels()
    zct_list = []
    for z in range(size_z):
        for c in range(size_c):
            for t in range(size_t):
                zct_list.append((z, c, t))
    planes = pixels.getPlanes(zct_list)
    img_pix = []
    for p in planes:
        img_pix.append(p)
    img_pix = np.array(img_pix)
    img_pix = img_pix.reshape((size_z, size_c, size_t,) + img_pix.shape[1:])
    img_pix = img_pix.transpose(2, 1, 0, 3, 4)
    return img_pix


def download_project(project_id, conn, outputdir):
    project = conn.getObject('Project', project_id)
    for ds in project.listChildren():
        print(ds.getName())
        for image in ds.listChildren():
            fnout = os.path.join(outputdir, project.getName(), ds.getName(), image.getName())
            if not os.path.exists(fnout):
                img = load_image(image, conn)
                imsave(fnout, img)
