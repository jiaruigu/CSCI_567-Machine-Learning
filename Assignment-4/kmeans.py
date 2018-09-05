import numpy as np

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        '''
        centroids=x[np.random.choice(np.arange(x.shape[0]),self.n_cluster,replace=False)]
        j_old=9999999
        for itr in range(self.max_iter):
            r=np.zeros((N,self.n_cluster))
            membership=list(map(lambda x_i:np.argmin(list(map(lambda c: np.linalg.norm(x_i-c)**2, centroids))),x))
            for i in range(N):
            	r[i][membership[i]]=1
        	j_new=
        '''
        centroids=x[np.random.choice(np.arange(x.shape[0]),self.n_cluster,replace=False)]
        j_old=9999999
        membership=np.zeros((N))
        for itr in range(self.max_iter):
            j_new=0
            r=np.zeros((N,self.n_cluster))
            for i in range(N):
                dist_min=9999999
                for k in range(self.n_cluster):
                    dist = np.linalg.norm(x[i]-centroids[k])**2
                    if dist<dist_min:
                        dist_min=dist
                        k_min=k
                membership[i]=k_min
                r[i][k_min]=1
                j_new+=dist_min
            j_new=1/N*j_new
            if abs(j_old-j_new)<=self.e:
                break
            j_old=j_new
            for k in range(self.n_cluster):
                temp1,temp2=0,0
                for i in range(N):
                    temp1+=r[i][k]*x[i]
                    temp2+=r[i][k]
                centroids[k]=temp1/temp2
        return centroids, membership, itr
        
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        centroids=x[np.random.choice(np.arange(x.shape[0]),self.n_cluster,replace=False)]
        j_old=9999999
        for itr in range(self.max_iter):
            j_new=0
            r=np.zeros((N,self.n_cluster))
            for i in range(N):
                dist_min=9999999
                for k in range(self.n_cluster):
                    dist = np.linalg.norm(x[i]-centroids[k])**2
                    if dist<dist_min:
                        dist_min=dist
                        k_min=k
                r[i][k_min]=1
                j_new+=dist_min
            j_new=1/N*j_new
            if abs(j_old-j_new)<=self.e:
                break
            j_old=j_new
            for k in range(self.n_cluster):
                temp1,temp2=0,0
                for i in range(N):
                    temp1+=r[i][k]*x[i]
                    temp2+=r[i][k]
                centroids[k]=temp1/temp2
        centroid_labels=np.zeros((self.n_cluster))
        for k in range(self.n_cluster):
            vote=np.zeros((np.max(y)+1))
            for i in range(N):
                vote[y[i]]+=r[i][k]
            centroid_labels[k]=np.argmax(vote)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        y=np.zeros((N))
        membership=list(map(lambda x_i:np.argmin(list(map(lambda c: np.linalg.norm(x_i-c)**2, self.centroids))),x))
        for i in range(N):
            y[i]=self.centroid_labels[membership[i]]
        return y
        # DONOT CHANGE CODE BELOW THIS LINE
