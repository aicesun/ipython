import numpy as np
from collections import Iterable, OrderedDict
from math import floor, ceil
from sets import Set
import matplotlib
import matplotlib.pyplot as plt
from openquake.hazardlib import gsim, imt
from openquake.hazardlib.scalerel.wc1994 import WC1994

AVAILABLE_GSIMS = gsim.get_available_gsims()

PARAM_DICT = {'magnitudes': [],
              'distances': [],
              'distance_type': 'rjb',
              'vs30': [],
              'strike': None,
              'dip': None,
              'rake': None,
              'ztor': None,
              'hypocentre_location': (0.5, 0.5),
              'hypo_loc': (0.5, 0.5),
              'msr': WC1994()}

PLOT_UNITS = {'PGA': 'g',
              'PGV': 'cm/s',
              'SA': 'g',
              'IA': 'm/s',
              'CSV': 'g-sec',
              'RSD': 's',
              'MMI': ''}

DISTANCE_LABEL_MAP = {'repi': 'Epicentral Dist.',
                      'rhypo': 'Hypocentral Dist.',
                      'rjb': 'Joyner-Boore Dist.',
                      'rrup': 'Rupture Dist.',
                      'rx': 'Rx Dist.'}

FIG_SIZE = (7, 5)

# RESET Axes tick labels
matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)

class DistanceIMTTrellis(MagnitudeIMTTrellis):
    """
    Trellis class to generate a plot of the GMPE attenuation with distance
    """
    XLABEL = "%s (km)"
    YLABEL = "Median %s (%s)"
    def __init__(self, magnitudes, distances, gsims, imts, params, 
            stddevs="Total", **kwargs):
        """
       
        """
        if isinstance(magnitudes, float):
            magnitudes = [magnitudes]

        super(DistanceIMTTrellis, self).__init__(magnitudes, distances, gsims,
            imts, params, stddevs, **kwargs)
    
    def _build_plot(self, ax, i_m, gmvs):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        distance_vals = getattr(self.dctx, self.distance_type)

        assert (self.plot_type=="loglog") or (self.plot_type=="semilogy")
        
        for gmpe in self.gsims:
            self.labels.append(gmpe.__class__.__name__)
            if self.plot_type == "semilogy":
                line, = ax.semilogy(distance_vals,
                                  gmvs[gmpe.__class__.__name__][i_m][0, :],
                                  '-',
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]
            else:
                line, = ax.loglog(distance_vals,
                                  gmvs[gmpe.__class__.__name__][i_m][0, :],
                                  '-',
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]

            self.lines.append(line)
            ax.grid(True)
            #ax.set_title(i_m, fontsize=12)
            
            ax.set_xlim(min_x, max_x)
            self._set_labels(i_m, ax)

        
    def _set_labels(self, i_m, ax):
            """
            Sets the labels on the specified axes
            """
            ax.set_xlabel("%s (km)" % DISTANCE_LABEL_MAP[self.distance_type],
                          fontsize=16)
            if 'SA(' in i_m:
                units = PLOT_UNITS['SA']
            else:
                units = PLOT_UNITS[i_m]
            ax.set_ylabel("Median %s (%s)" % (i_m, units), fontsize=16)


class DistanceSigmaIMTTrellis(DistanceIMTTrellis):
    """

    """
    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = OrderedDict()
        for gmpe in self.gsims:
            gmvs.update([(gmpe.__class__.__name__, {})])
            for i_m in self.imts:
                gmvs[gmpe.__class__.__name__][i_m] = np.zeros([len(self.rctx),
                                                               self.nsites],
                                                               dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx,
                             rct,
                             self.dctx,
                             imt.from_string(i_m),
                             [self.stddevs])
                        gmvs[gmpe.__class__.__name__][i_m][iloc, :] = sigmas[0]
                    except KeyError:
                        gmvs[gmpe.__class__.__name__][i_m] = []
                        break
                        
                        
        return gmvs

    def _build_plot(self, ax, i_m, gmvs):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        distance_vals = getattr(self.dctx, self.distance_type)

        assert (self.plot_type=="loglog") or (self.plot_type=="semilogy")
        
        for gmpe in self.gsims:
            self.labels.append(gmpe.__class__.__name__)
            if self.plot_type == "loglog":
                line, = ax.semilogx(distance_vals,
                                  gmvs[gmpe.__class__.__name__][i_m][0, :],
                                  '-',
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]
            else:
                line, = ax.plot(distance_vals,
                                gmvs[gmpe.__class__.__name__][i_m][0, :],
                                '-',
                                linewidth=2.0,
                                label=gmpe.__class__.__name__)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]


            self.lines.append(line)
            ax.grid(True)
            #ax.set_title(i_m, fontsize=12)
            
            ax.set_xlim(min_x, max_x)
            self._set_labels(i_m, ax)

        
    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("%s (km)" % DISTANCE_LABEL_MAP[self.distance_type],
                      fontsize=16)
        ax.set_ylabel(self.stddevs + " Std. Dev.", fontsize=16)