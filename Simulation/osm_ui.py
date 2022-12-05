import numpy as np
import matplotlib.pyplot as plt

def print_(message, type_="info"):
    print(message)
    if type_ == "error":
        raise ValueError

def plot_map(map, longitude_bounds, latitude_bounds,
             title, nro_fig=None):
    if nro_fig is None:
        ff = plt.figure()
    else:
        ff = plt.figure(nro_fig)
    ff.clf()
    gg = ff.add_subplot(111)

    gg.imshow(map, extent=[longitude_bounds[0], longitude_bounds[1],
                           latitude_bounds[0], latitude_bounds[1]],
              aspect="equal")

    gg.set_xlabel("longitude (°)")
    gg.set_ylabel("latitude (°)")
    gg.set_xlim([-3.02163, -3.01211])
    gg.set_ylim([48.19429, 48.20777])
    gg.set_title(title)
    gg.grid(True)
    #plt.pause(0.01)
    return gg



def plot_xy(data_x, data_y, x_label, y_label, title, symbol=None, nro_fig=None):
    if nro_fig is None:
        ff = plt.figure()
    else:
        ff = plt.figure(nro_fig)
    ff.clf()
    gg = ff.add_subplot(111)
    if symbol is None:
        gg.plot(data_x, data_y)
    else:
        gg.plot(data_x, data_y, symbol)
    gg.set_xlabel(x_label)
    gg.set_ylabel(y_label)
    gg.set_title(title)
    gg.grid(True)
    # plt.pause(0.00001)
    return gg

def plot_xy_add(gg, data_x, data_y, symbol=None):
    if symbol is None:
        gg.plot(data_x, data_y)
    else:
        gg.plot(data_x, data_y, symbol)
    #plt.pause(0.01)

def plot_logxlogy(data_x, data_y, x_label, y_label, title, symbol=None, nro_fig=None):
    if nro_fig is None:
        ff = plt.figure()
    else:
        ff = plt.figure(nro_fig)
    ff.clf()
    gg = ff.add_subplot(111)
    if symbol is None:
        gg.loglog(data_x, data_y)
    else:
        gg.loglog(data_x, data_y, symbol)
    gg.set_xlabel(x_label)
    gg.set_ylabel(y_label)
    gg.set_title(title)
    gg.grid(True)
    #plt.pause(0.01)
    return gg

def plot_logxlogy_add(gg, data_x, data_y, symbol=None):
    if symbol is None:
        gg.loglog(data_x, data_y)
    else:
        gg.loglog(data_x, data_y, symbol)
    #plt.pause(0.01)

def plot_logxy(data_x, data_y, x_label, y_label, title, symbol=None, nro_fig=None):
    if nro_fig is None:
        ff = plt.figure()
    else:
        ff = plt.figure(nro_fig)
    ff.clf()
    gg = ff.add_subplot(111)
    if symbol is None:
        gg.semilogx(data_x, data_y)
    else:
        gg.semilogx(data_x, data_y, symbol)
    gg.set_xlabel(x_label)
    gg.set_ylabel(y_label)
    gg.set_title(title)
    gg.grid(True)
    #plt.pause(0.01)
    return gg

def plot_logxy_add(gg, data_x, data_y, symbol=None):
    if symbol is None:
        gg.semilogx(data_x, data_y)
    else:
        gg.semilogx(data_x, data_y, symbol)
    #plt.pause(0.01)



def plot_image(img, x_label, y_label, title, nro_fig=None,
               xlim=None, ylim=None, zlim=None, colormap=None, q_grid=True,
               aspect="auto", q_colorbar=False):
    if nro_fig is None:
        ff = plt.figure()
    else:
        ff = plt.figure(nro_fig)
    ff.clf()
    gg = ff.add_subplot(111)

    if zlim is None:
        v_min = np.min(img[np.isfinite(img)])
        v_max = np.max(img[np.isfinite(img)])
    else:
        v_min = zlim[0]
        v_max = zlim[1]
    print(v_min, v_max)

    if xlim is None:
        x0 = -0.5
        x1 = img.shape[1] - 0.5
    else:
        x0 = xlim[0]
        x1 = xlim[1]

    if ylim is None:
        y0 = -0.5
        y1 = img.shape[0] - 0.5
    else:
        y0 = ylim[0]
        y1 = ylim[1]

    if colormap is None:
        cmap = None
    else:
        cmap = colormap

    y = gg.imshow(img, vmin=v_min, vmax=v_max, extent=[x0, x1, y0, y1],
              aspect=aspect, cmap=colormap)

    gg.set_xlabel(x_label)
    gg.set_ylabel(y_label)
    gg.set_title(title)
    if q_grid == True:
        gg.grid(True)
    #plt.pause(0.01)
    if q_colorbar == True:
        plt.colorbar(y)
    return gg

def plot_mesh2d(m, title, line_symbol=None, pnt_symbol=None, nro_fig=None,
                aspect="auto"):
    """ Tracé d'un maillage 2D """
    if nro_fig is None:
        ff = plt.figure()
    else:
        ff = plt.figure(nro_fig)
    ff.clf()

    gg = ff.add_subplot(111)
    gg.triplot(m.pnt[:,0], m.pnt[:,1], m.sP, marker=pnt_symbol,
               linestyle=line_symbol)
    gg.set_aspect(aspect)
    if title is not None:
        gg.set_title(title)
    return gg
