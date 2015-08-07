import numpy as np

def myhist(a, bins=10, range=None, normed=False, weights=None,
              density=None):
    """
    Compute the histogram of a set of data.
    
    Jiaxin Han: Updated algorithm to handle large dynamic range. 2015-07-03 18:57:40
    
    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a sequence,
        it defines the bin edges, including the rightmost edge, allowing
        for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored.
    normed : bool, optional
        This keyword is deprecated in Numpy 1.6 due to confusing/buggy
        behavior. It will be removed in Numpy 2.0. Use the density keyword
        instead.
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that this latter behavior is
        known to be buggy with unequal bin widths; use `density` instead.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in `a`
        only contributes its associated weight towards the bin count
        (instead of 1).  If `normed` is True, the weights are normalized,
        so that the integral of the density over the range remains 1
    density : bool, optional
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
        Overrides the `normed` keyword if given.
    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    See Also
    --------
    histogramdd, bincount, searchsorted, digitize
    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is::
      [1, 2, 3, 4]
    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and the
    second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which *includes*
    4.
    Examples
    --------
    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([ 0.25,  0.25,  0.25,  0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))
    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([ 0.5,  0. ,  0.5,  0. ,  0. ,  0.5,  0. ,  0.5,  0. ,  0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> np.sum(hist*np.diff(bin_edges))
    1.0
    """

    a = np.asarray(a)
    if weights is not None:
        weights = np.asarray(weights)
        if np.any(weights.shape != a.shape):
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()

    if (range is not None):
        mn, mx = range
        if (mn > mx):
            raise AttributeError(
                'max must be larger than min in range parameter.')

    if not np.iterable(bins):
        if np.isscalar(bins) and bins < 1:
            raise ValueError(
                '`bins` should be a positive integer.')
        if range is None:
            if a.size == 0:
                # handle empty arrays. Can't determine range, so use 0-1.
                range = (0, 1)
            else:
                range = (a.min(), a.max())
        mn, mx = [mi + 0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = np.linspace(mn, mx, bins + 1, endpoint=True)
    else:
        bins = np.asarray(bins)
        if (np.diff(bins) < 0).any():
            raise AttributeError(
                'bins must increase monotonically.')

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = int
    else:
        ntype = weights.dtype
    n = np.zeros(len(bins)-1, ntype)

    if len(a)>0:
        bin_index=np.digitize(a, bins)
        n=np.bincount(bin_index, weights=weights, minlength=len(bins)+1)[1:-1]

    #block = 65536
    #if weights is None:
        #for i in arange(0, len(a), block):
            #sa = sort(a[i:i+block])
            #n += np.r_[sa.searchsorted(bins[:-1], 'left'),
                       #sa.searchsorted(bins[-1], 'right')]
    #else:
        #zero = array(0, dtype=ntype)
        #for i in arange(0, len(a), block):
            #tmp_a = a[i:i+block]
            #tmp_w = weights[i:i+block]
            #sorting_index = np.argsort(tmp_a)
            #sa = tmp_a[sorting_index]
            #sw = tmp_w[sorting_index]
            #cw = np.concatenate(([zero, ], sw.cumsum()))
            #bin_index = np.r_[sa.searchsorted(bins[:-1], 'left'),
                              #sa.searchsorted(bins[-1], 'right')]
            #n += cw[bin_index]

    #n = np.diff(n)

    if density is not None:
        if density:
            db = np.array(np.diff(bins), float)
            return n/db/n.sum(), bins
        else:
            return n, bins
    else:
        # deprecated, buggy behavior. Remove for Numpy 2.0
        if normed:
            db = np.array(np.diff(bins), float)
            return n/(n*db).sum(), bins
        else:
            return n, bins       
		  
