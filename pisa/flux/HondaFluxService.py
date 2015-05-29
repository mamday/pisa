#
# HondaFluxService.py
#
# This flux service provides flux values for a grid of energy / cos(zenith)
# bins. It loads a flux table as provided by Honda (for now only able to use
# azimuth-averaged data) and uses spline interpolation to provide the integrated
# flux per bin.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import bisect
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.resources.resources import open_resource

#Global definition of primaries for which there is a neutrino flux
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']
#primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

class FluxService(object):
    def __init__(self, flux_file=None, smooth=0.05,flux_bins=None,prim_list=primaries,**kwargs):
        super(FluxService, self).__init__(**kwargs)
        logging.info("Loading atmospheric flux table %s" %flux_file)

        #Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        #columns in files are in the same order
        cols = ['energy']+prim_list
        self.flux_dict = dict(zip(cols, table))
        for key in self.flux_dict.iterkeys():

            #There are 20 lines per zenith range
            self.flux_dict[key] = np.array(np.split(self.flux_dict[key], 20))
            if not key=='energy':
                self.flux_dict[key] = self.flux_dict[key].T

        #Set the zenith and energy range
        self.flux_dict['energy'] = self.flux_dict['energy'][0]
        self.flux_dict['coszen'] = flux_bins 

        #Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')
        # do this in log of energy and log of flux (more stable)
        logE, C = np.meshgrid(np.log10(self.flux_dict['energy']), self.flux_dict['coszen'])
        self.spline_dict = {}
        for ptype in prim_list:
            #Get the logarithmic flux
            if(ptype in self.flux_dict.keys()):
              valid_parts = [list(i[:len(flux_bins)]) for i in self.flux_dict[ptype]]
              log_flux = np.log10(valid_parts).T
              #Get a spline representation
              spline =  bisplrep(logE, C, log_flux, s=smooth)
              #and store
              self.spline_dict[ptype] = spline

    def get_flux(self, ebins, czbins, prim):
        return None

class MuonFluxService(FluxService):
    """Load a cosmic ray muon flux table from .dat files produced from
       muon gun distributions. For now output is stored in .dat files and
       not computed directly from splines
    """
    def __init__(self,prim_list=['muons'],**kwargs):
        super(MuonFluxService, self).__init__(prim_list=['muons'],**kwargs)

    def get_flux(self, ebins, czbins, prim='muons'):
        '''Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith).'''
        #TODO: Currently there is no use for prim except to make this function behave like the others 
        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        sep_pt = bisect.bisect_left(czbins,0.0)
        downbins = czbins[:sep_pt]
        upbins = czbins[sep_pt-1:]
        czvals = get_bin_centers(downbins)
        upvals = get_bin_centers(upbins)
        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        down_table = bisplev(np.log10(evals), czvals, self.spline_dict['muons'])
        down_table = np.power(10., down_table).T
        up_table = np.array([np.array(len(evals)*[0.0]) for i in upvals])
        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        #czbin_sizes = 2.*np.pi*get_bin_sizes(upbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table = np.concatenate((down_table,up_table))
        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])
        return return_table.T


class HondaFluxService(FluxService):
    """Load a neutrino flux from Honda-styles flux tables in units of
       [GeV^-1 m^-2 s^-1 sr^-1] and return a 2D spline interpolated
       function per flavour.  For now only supports azimuth-averaged
       input files.
    """

    def __init__(self,**kwargs):
        super(HondaFluxService, self).__init__(**kwargs)

    def get_flux(self, ebins, czbins, prim):
        """Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary."""

        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)

        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        if(prim in self.spline_dict):
          return_table = bisplev(np.log10(evals), czvals, self.spline_dict[prim])
          return_table = np.power(10., return_table).T

          #Flux is given per sr and GeV, so we need to multiply
          #by bin width in both dimensions
          #Get the bin size in both dimensions
          ebin_sizes = get_bin_sizes(ebins)
          czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
          bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

          return_table *= np.abs(bin_sizes[0]*bin_sizes[1])

          return return_table.T


