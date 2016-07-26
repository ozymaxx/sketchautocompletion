import numpy as np
class FileIO:
    """File class used for saving an loading """
    def __init__(self):
        pass
    def save(self,isFull, names, feature, f):
        """"Save features to csv file
        isFull: array of 1 or 0
        names: name array of the instances
        feature: featureList
        f: name of the file to be saved
        """

        import numpy as np
        import pandas as pd

        df = pd.DataFrame(data = isFull)
        df.columns = ['isFull']
        df1 = pd.DataFrame(data=feature)
        result = pd.concat([df, df1], axis=1, join_axes=[df1.index])

        df = pd.DataFrame(names)
        df.columns =['names']

        result = pd.concat([df, result], axis=1, join_axes=[result.index])

        result.to_csv(f, mode = 'w', index = False)

    def load(self, f):
        """"Load features from csv file
        INPUT
        f: name of the file to be saved
        OUTPUT
        isFull: array of 1 or 0
        names: name array of the instances
        feature: featureList

        """
        import pandas as pd
        a = pd.read_csv(f)
        names = a['names'].tolist()
        isFull = a['isFull'].as_matrix()
        features = a[a.columns[2:]].as_matrix()
        return names,isFull,features
