import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means=KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, _, i = k_means.fit(x)
            self.variances=np.zeros((self.n_cluster,D,D))
            for k in range(self.n_cluster):
                self.variances[k]=np.identity(D)
            self.pi_k=np.ones((self.n_cluster))*1/self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means=np.random.random_sample((self.n_cluster,D))
            self.variances=np.zeros((self.n_cluster,D,D))
            for k in range(self.n_cluster):
                self.variances[k]=np.identity(D)
            self.pi_k=np.ones((self.n_cluster))*1/self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        l=self.compute_log_likelihood(x)
        for itr in range(self.max_iter):
            gamma=np.zeros((N,self.n_cluster))
            temp=np.zeros((N,self.n_cluster))
            for i in range(N):
                for k in range(self.n_cluster):
                    v_k=self.variances[k]
                    while(np.linalg.det(v_k)==0):
                        v_k+=0.001*np.identity(v_k.shape[0])
                    GD=1/(2*np.pi*np.linalg.det(v_k)**0.5)*np.exp(-1/2*np.matmul(np.matmul([(x[i]-self.means[k])],np.linalg.inv(v_k)),np.transpose([x[i]-self.means[k]])))[0][0]
                    temp[i][k]=self.pi_k[k]*GD
                for k in range(self.n_cluster):
                    gamma[i][k]=temp[i][k]/np.sum(temp[i])
            N_k=np.sum(gamma,axis=0)
            for k in range(self.n_cluster):
                temp1=np.zeros((D))
                temp2=np.zeros((D,D))
                for i in range(N):
                    temp1+=gamma[i][k]*x[i]
                self.means[k]=temp1/N_k[k]
                for i in range(N):
                    temp2+=gamma[i][k]*np.matmul(np.transpose([x[i]-self.means[k]]), [x[i]-self.means[k]])
                self.variances[k]=temp2/N_k[k]
                self.pi_k[k]=N_k[k]/N
            l_new=self.compute_log_likelihood(x)
            if abs(l-l_new)<=self.e:
                return itr
            l=l_new
        return itr
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        samples=[]
        for n in range(N):
            samples.append([])
            r = np.random.multinomial(1, self.pi_k)
            for k in range(r.shape[0]):
                if r[k]==1:
                    break
            sample = np.random.multivariate_normal(self.means[k], self.variances[k], 1)[0]
            for s in sample:
                samples[n].append(s)
        return np.array(samples)
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N, D = x.shape
        l=0
        for n in range(N):
            px=0
            for k in range(self.n_cluster):
                v_k=self.variances[k]
                while(np.linalg.det(v_k)==0):
                    v_k+=0.001*np.identity(v_k.shape[0])
                px_z_k=1/(2*np.pi*np.linalg.det(v_k)**0.5)*np.exp(-1/2*np.matmul(np.matmul([(x[n]-self.means[k])],np.linalg.inv(v_k)),np.transpose([x[n]-self.means[k]])))[0][0]
                pz_k=self.pi_k[k]
                px+=pz_k*px_z_k
            l+=float(np.log(px))
        return l
        # DONOT MODIFY CODE BELOW THIS LINE
