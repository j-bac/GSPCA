import scipy
import numpy as np

def GSPCA( data, labels, nComp, param ):
    #GSPCA calculates generalised advanced supervised PCA with respect to [1].
    #   [ V, D ] = GSPCA( data, labels, nComp, kind ) return n-by-nComp
    #               matrix V with PCs as columns and diagonal nComp-by-nComp
    #               matrix D with eigenvalues corresponding to PCs. 
    #   data is n-by-m matrix of data (covariance matrix is unacceptable). Data
    #       MUST be centred before.
    #   labels is numeric vector with n elements. The same labels corresponds
    #       to points of the same class. Number of unique values in labels is
    #       L. Classes are numerated in the order of increasing value of labels.
    #   nComp is number of required component.
    #   param is parameter of method:
    #       scalar numeric value is parameter of intraclass attraction: the
    #           functional to maximise is mean squared distances between points
    #           of different classes minus param multiplied to sum of mean
    #           squared distances between points of each class
    #       numeric vector with L elements is vector of attractions in each
    #           class: the functional to maximise is mean squared distances
    #           between points of different classes minus sum of  sum of mean
    #           squared distances between points of each class multiplied by
    #           corresponding element of vector param.
    #       numeric matrix L-by-L is matrix of repulsion coefficients. The
    #           elements upper than main diagonal are coefficients of repulsion
    #           between corresponding clusses. The diagonal elements are
    #           attraction coefficients for corresponding classes.
    #
    #References
    #1. Mirkes, Evgeny M., Gorban, Alexander N., Zinovyev, Andrei Y.,
    #   Supervised PCA, Available online in https://github.com/Mirkes/SupervisedPCA/wiki
    #2. Gorban, Alexander N., Zinovyev, Andrei Y. “Principal Graphs and Manifolds”, 
    #   Chapter 2 in: Handbook of Research on Machine Learning Applications and Trends: 
    #   Algorithms, Methods, and Techniques, Emilio Soria Olivas et al. (eds), 
    #   IGI Global, Hershey, PA, USA, 2009, pp. 28-59.
    #3. Zinovyev, Andrei Y. "Visualisation of multidimensional data" Krasnoyarsk: KGTU,
    #   p. 180 (2000) (In Russian).
    #4. Koren, Yehuda, and Liran Carmel. "Robust linear dimensionality
    #   reduction." Visualization and Computer Graphics, IEEE Transactions on
    #   10.4 (2004): 459-470.
    #
    #Licensed from CC0 1.0 Universal - Author Evgeny Mirkes https://github.com/Mirkes/SupervisedPCA/blob/master/

    #Get sizes of data
    n, m = data.shape
    data = data.astype(float)
    labels = labels.astype(float)
    # List of classes
    labs = np.unique(labels)
    # Number of classes
    L = len(labs)
    # Check the type of nComp
    if nComp > m or nComp < 1:
        raise ValueError('Incorrect value of nComp: it must be positive integer equal to or less than m')

    # Form matrix of coefficients
    if type(param) in [int,float]:
        coef = np.ones((L,L))
        coef = coef + np.diag((param - 1) * np.diag(coef))
    elif len(param.shape) == 1:
        if len(param) != L:
            raise ValueError(['Argument param must be scalar, or vector with L elements of L-by-L matrix,\n where L is number of classes (unique values in labels)'])
        coef = np.ones((L,L))
        coef = coef + np.diag(np.diag(param - 1))
    elif len(param.shape) == 2:
        [a, b] = param.shape
        if a != L or b != L:
            raise ValueError(['Argument param must be scalar, or vector with L elements of L-by-L matrix,\n where L is number of classes (unique values in labels)'])
    else:
        raise ValueError(['Argument param must be scalar, or vector with L elements of L-by-L matrix,\n where L is number of classes (unique values in labels)'])

    # Symmetrize coef matrix
    coef = coef - np.tril(coef, -1) + np.triu(coef, 1).T

    # Calculate diagonal terms of Laplacian matrix without devision by
    # number of elements in class
    diagV = np.diag(coef)
    diagC = np.sum(coef,axis=0) - diagV

    # Calculate transformed covariance matrix
    M = np.zeros((m,m))
    means = np.zeros((L, m))
    # Loop to form the diagonal terms and calculate means
    for c in range(L):
        # Get index of class
        ind = labels == labs[c]
        # Calculate mean
        means[c, :] = np.mean(data[ind, :],axis=0)
        # Calculate coefficient for Identity term
        nc = np.sum(ind,axis=0)
        coefD = diagC[c] / nc - 2 * diagV[c] / (nc - 1)
        # Add the diagonal term
        M = (M + 2 * diagV[c] * nc / (nc - 1) * (means[[c], :].T @ means[[c], :])
            + coefD * data[ind, :].T @ data[ind, :])

    # Loop for off diagonal parts
    for c in range(L - 1):
        for cc in range(c + 1, L):
            tmp = means[[c], :].T @ means[[cc], :]
            M = M - coef[c, cc] * (tmp + tmp.T)

    #Request calculations from eigs
    if nComp<m-1:
        D, V = scipy.sparse.linalg.eigs(M, nComp)
    else:
        D, V = scipy.linalg.eig(M)
    ind = np.argsort(D)[::-1]
    V = V[:,ind]
    D = D[ind]

    return V, D