import numpy as np
class FileIO:
    def __init__(self):
        pass
    def save(self,isFull, names, feature, f):
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
        import pandas as pd
        a = pd.read_csv(f)
        newNames = a['names'].tolist()
        newIsFull = a['isFull'].as_matrix()
        newfeature = a[a.columns[2:]].as_matrix()
        return newNames,newIsFull,newfeature


isFull = np.array([0,1,1])
names = ['a','b','c']
feature = [[1,2],[3,4],[5,6]]


f = FileIO()
f.save(isFull,names,feature, "noldu.csv")
k = f.load("noldu.csv")
print k
a = 5