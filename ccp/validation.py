import numpy as np

class BiclusterValidator:
    """ Class dedicated to the different evaluation metrics to evaluate the quality of biclusters
    the defined methods are all extracted from papers.

    Raises:
        ValueError: lengths of the lists must not be empty

    Returns:
        _type_: _description_
    """


    def prelic_bicluster_relevance(self, bset1, bset2):

        """ Function that implements Prelić et al. [92] proposed the measures of relevance SPRel defined as in REF[3] paper 
        'Metaheuristic Biclustering Algorithms: From State-of-the-Art to Future Opportunities'
        Args:
            bset1 (list : Bicluster): set of the first biclusters as first argument.
            bset2 (list : Biclsuter): set of reference biclusters as second argument

        Returns:
            float: the relevance score betweent the two sets.
        """

        if len(bset1) == 0 or len(bset2) == 0:
            raise ValueError("Lists must contain at least one element ")

            
        max_rows = np.zeros(len(bset1))
        max_cols = np.zeros(len(bset2))

        for index, b1 in enumerate(self.bset1):
            
            row_rel_values = np.array([np.intersect1d(b1.rows, bref.rows).shape[0]/ np.union1d(b1.rows, bref.rows).shape[0] for bref in self.ref_biclusters])

            col_rel_values = np.array([np.intersect1d(b1.columns, bref.columns).shape[0]/ np.union1d(b1.columns, bref.columns).shape[0] for bref in self.ref_biclusters])
            
            max_rows[index], max_cols[index] = np.max(row_rel_values), np.max(col_rel_values)

        
        res = np.mean(max_rows) * np.mean(max_cols)
        return np.sqrt(res)

    def prelic_bicluster_recovery(self, btest, bref):
        """
        Function that implements recovery metric defined as the inverse of relevance by switching the arguments bref and btest : 
        SPRec (B, B* ) = SPRel (B* , B),
        Args:
            btest (list  Bicluster): set of biclusters to be evaluated.
            bref (list : Bicluster): priori knowledge - set of reference biclusters

        Returns:
            float: score of recovery between the 2 sets
        """
        return self.prelic_bicluster_relevance(bref, btest)


