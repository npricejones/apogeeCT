"""
chemicaltag.py contains functions to perform chemical tagging with DBSCAN with
a variety of input parameters.

Natalie Price-Jones, UofT, 2020
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from clustering_stats import membercount
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics.pairwise import euclidean_distances

plt.rc('font', family='serif',size=18)

class DBSCAN_tagging(object):
    """
    Class of functions for chemical tagging and creating fake data for testing.
    """
    def __init__(self,elems,elemkeys,numstars):
        """
        Read in elements to be used for chemical tagging and their keys, as well
        as the number of stars considered for chemical tagging.

        elems:
        elemkeys:       Keys to access element columns in numpy structured
                        array or pandas DataFrame
        numstars:       Number of stars to be used for chemical tagging.

        """
        self.elems = elems
        self.elemkeys = elemkeys
        self.numstars = numstars
        self.numelems = len(elems)
        return None

    def realdata(self,allStar,goodstars):
        """
        Get tagging array from APOGEE allStar file.

        allStar:        APOGEE allStar structured array.
        goodstars:      Boolean array that is True for stars that should be
                        used for chemical tagging.

        Returns tagging array.
        """
        # Get array that is just abundances for chemical tagging.
        self.abundances = np.zeros((self.numstars,self.numelems),dtype=float)
        # Store properties of abundances
        self.meanlist = []
        self.stdslist = []
        # Populate columns of abundances array
        for k,key in enumerate(self.elemkeys):
            self.abundances[:,k] = allStar[goodstars][key]
            self.meanlist.append(np.mean(allStar[goodstars][key]))
            self.stdslist.append(np.std(allStar[goodstars][key]))
        return self.abundances

    def independent_gaussian(self,allStar,goodstars):
        """
        Create a fake set of abundance data with each dimension independently
        normally distributed with the mean and standard deviation of the
        actual dataset.

        allStar:        APOGEE allStar structured array.
        goodstars:      Boolean array that is True for stars that should be
                        used for chemical tagging.

        Returns tagging array
        """
        self.abundances = self.realdata(allStar,goodstars)
        for k,key in enumerate(self.elemkeys):
            self.abundances[:,k] = np.random.normal(loc=self.meanlist[k],
                                                    scale=self.stdlist[k],
                                                    size=self.numstars)
        return self.abundances

    def multivariate_gaussian(self,allStar,goodstars):
        """
        Create a fake set of abundance data with each dimension normally
        distributed with the mean and standard deviation of the
        actual dataset.

        allStar:        APOGEE allStar structured array.
        goodstars:      Boolean array that is True for stars that should be
                        used for chemical tagging.

        Returns tagging array
        """
        self.abundances = self.realdata(allStar,goodstars)
        covprops = np.cov(self.abundances.T)
        self.abundances = np.random.multivariate_normal(self.meanlist,
                                                        covprops,
                                                        size=self.numstars)
        return self.abundances

    def gaussian_mixture_model(self,allStar,goodstars,n_components=3,
                               max_iter=200,n_init=10,**kwargs):
        """
        Create a fake dataset using a Gaussian Mixture Model to mimic the
        large-scale structure in abundance space.

        allStar:        APOGEE allStar structured array.
        goodstars:      Boolean array that is True for stars that should be
                        used for chemical tagging.
        n_components:   Number of components in chemical space, default: 3
        max_iter:       Maximum iterations of the GMM, default: 200
        n_init:         Number of initializations, default: 1
        **kwargs:       Other sklearn.mixture kwargs

        Returns tagging array
        """
        # Initialize Gaussian Mixture Model
        bgmm = BayesianGaussianMixture(n_components=n_components,
                                       covariance_type='full',
                                       max_iter=max_iter,
                                       n_init=n_init,**kwargs)
        self.abundances = self.realdata(allStar,goodstars)
        # Model the abundance data
        model = bgmm.fit(self.abundances)
        start = 0
        # Use the independent components to construct actual abundances, with
        # the number of abundances drawn from each component determined by
        # the weight of that component
        for n in range(n_components-1):
            component_size = int(self.numstars*model.weights_[n])
            component = np.random.multivariate_normal(model.means_[n],
                                                      model.covariances_[n],
                                                      size=component_size)
            self.abundances[start:start+component_size]=component
            start+=component_size
        component_size = self.numstars-start
        lastcomponent = np.random.multivariate_normal(model.means_[-1],
                                                      model.covariances_[-1],
                                                      size=component_size)
        self.abundances[start:]=lastcomponent
        return self.abundances

    def simulated(self,fname=None,abunkey='labn',genfn='choosestruct',
                  workdir='./'):
        """
        Create a simulated abundance space that mimics the properties of the
        real abundance space.

        fname:      hdf5 file where simulated abundances are stored. If
                    unspecified, new data is generated.
        abunkey:    Key in hdf5 file where abundance data is found.
        genfn:      Function used to generate cluster centres in simulated data.
        # TODO: FIX WORKDIR RELATIONSHIPS
        """
        if not isinstance(fname,str):
            os.system(f'python3 makefake.py -n {self.numstars} -g {genfn} -w {workdir}')
            # TODO: FIX hdf5 default naming
            fs = glob.glob('{0}/case12*.hdf5'.format(workdir))
            fname=fs[-1]
        # Read in simulated data
        fake = h5py.File(fname)
        abun = np.array(fake[abunkey])
        # Find appropriate columns in simulated data
        simkeys = {'C':0,'N':1,'O':2,'NA':3,'MG':4,'AL':5,'SI':6,'S':7,'K':8,
                   'CA':9,'TI':10,'V':11,'MN':12,'FE':13,'NI':14}
        cols = [simkeys[i] for i in self.elems]
        self.abundances = np.zeros((len(abun),self.numelems),dtype=float)
        for k,key in enumerate(self.elemkeys):
            self.abundances[:,k] = abun[:,cols[k]]
            # Correct abundance ratios to be with respect to 'FE' instead of 'H'
            if key != 'FE_H':
                self.abundances[:,k]-=abun[:,simkeys['FE']]
        return self.abundances

    def typical_distance(self,N=1e4):
        """
        Calculate the typical distance between stars in the sample in the
        chosen abundance space.

        N:      Number of stars to use in distance calculation.

        Returns None.
        """
        # Get indexes of stars to use for distance calculation (no repeats)
        dinds = np.random.choice(np.arange(len(self.abundances)),
                                 size=int(N),replace=False)
        # Calculate the pairwise distances between the stars
        dists = euclidean_distances(self.abundances[dinds],
                                    self.abundances[dinds])
        self.typ = np.median(dists)
        return None

    def find_centers(self,labellist):
        """
        Calculate central positions of groups identified by DBSCAN.

        labellist:      List of stellar labels

        Returns mean abundance values for each abundance in each cluster.
        """
        # Get the number of clusters
        labs = np.unique(labellist)
        ncls = len(labs)
        centers = np.zeros((ncls,self.abundances.shape[1]))
        for l,lab in enumerate(labs):
            # Find star
            match = np.where(labellist==lab)
            members = self.abundances[match]
            centers[l] = np.mean(members,axis=0)
        return centers


    def DBSCAN_scales(self,scalerange=[0.02,0.03,0.04],Nptsrange = [2,3,4],
                      savename=None,minsize=15,workdir='./',metric='euclidean',
                      n_jobs=1,plot=True):
        """
        Run DBSCAN on the current abundance space for a variety of epsilon and
        Npts values.

        scalerange:         Scale factors by which to multiply the typical
                            distance in chemical space to derive an epsilon
                            value.
        Nptsrange:          Npts values to use.
        savename:           Name of hdf5 file in which to save result.
        minsize:            Minimum size of a group to identify scale
        workdir:            To be deprecated
        metric:             Distance metric used for DBSCAN
        n_jobs:             Number of jobs for DBSCAN to use
        plot:               Whether to plot number of groups as a function
                            of eps

        Returns None
        """
        self.typical_distance()
        # DBSCAN label storage
        labellist = []
        # Number of groups found
        ncls = []
        # Number of groups larger than minsize
        nclsmin = []
        # Eps values
        epsval = []
        # Npts values
        Nptsval = []
        # Run DBSCAN in series for each choice of scale and Npts
        for s,scale in enumerate(scalerange):
            for n,npts in enumerate(Nptsrange):
                db = DBSCAN(min_samples=npts,
                             eps=scale*self.typ,n_jobs=n_jobs,
                             metric=metric).fit(self.abundances)
                ncls.append(len(np.unique(db.labels_))-1)
                pcount,plabs = membercount(db.labels_)
                large = (plabs!=-1) & (pcount>=minsize)
                nclsmin.append(len(pcount[large]))
                labellist.append(db.labels_)
                epsval.append(scale*self.typ)
                Nptsval.append(npts)
        # Plot number of groups found for each parameter choice
        if plot:
            plt.figure(figsize=(8,6))
            ax = plt.subplot(111)
            ax.set_yscale('log')
            im=plt.scatter(epsval,ncls,color=Nptsval,marker='o',label='total')
            plt.scatter(epsval,nclsmin,color=Nptsval,marker='s',
                        label=f'>= {minsize} members')
            plt.colorbar(im,label='Npts')
            plt.xlabel('scale factor')
            plt.ylabel('number of clusters')
            legend = plt.legend(loc='best')
            legend.get_frame().set_linewidth(0.0)
            plt.savefig(savename+'ncls.pdf')
        # Find choice of eps and npts that maximize the number of clusters
        scalechoice = np.where(ncls == np.max(ncls))
        # If multiple parameters meet this then just use the first one
        scaleind = scalechoice[0][0]
        labs = labellist[scaleind]
        # Count the number of members in each group
        pcount,plabs = membercount(labellist[scaleind])
        # Calculate group silhouette coefficient
        sils = silhouette_samples(self.abundances,labs)
        clustersils = np.array([np.mean(sils[labs==lab]) for lab in plabs])
        # Choose good groups
        goodlabs = (pcount>=minsize) & (plabs > -1) & (clustersils > 0)
        goodinds = []
        c=0
        for l,lab in enumerate(labs):
            if lab in plabs[goodlabs]:
                goodinds.append(l)
        nstars = len(goodinds)
        ncls = len(plabs[goodlabs])
        print(f'{savename} - {nstars} stars in {ncls} clusters')
        # Remove output file if it exists (prevents h5py error)
        os.system('rm -f {1}/{0}.hdf5'.format(savename,workdir))
        output = h5py.File('{1}/{0}.hdf5'.format(savename,workdir),'a')
        # Store everything relevant to the run
        output['abundances']=self.abundances
        output['labels']=labs
        output['plabs']=plabs
        output['pcount']=pcount
        output['goodlabs']=goodlabs
        output['goodinds']=goodinds
        output['sils']=sils
        output['props']=props
        output.close()
        return None
