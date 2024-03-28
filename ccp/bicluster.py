class Bicluster:
    """A blueprint that represents a bicluster instance.
    In this class we assue that a bicluster is defined by its set of rows, columns and msr
    """

    def __init__(self, rows, columns, msr_score):
        self.rows = rows
        self.columns = columns
        self.msr_score = msr_score
