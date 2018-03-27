# -*- coding: UTF-8 -*-
#!/usr/bin/python
import numpy as np
import time
t0=time.time()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
train_data=np.load('/workspace/data/fyzhang/feature/video-queryZZ.npy')
pca=PCA(n_components=200,copy=True, whiten=True,svd_solver='auto')
pca.fit(train_data)
means = pca.mean_
smaller_data = pca.transform(train_data)
np.save('base_data_pcazz.npy', smaller_data)
print( smaller_data.shape)
t1=time.time()-t0
print(t1)
print smaller_data
from sklearn.utils.extmath import fast_dot
test_data=np.load('/workspace/data/fyzhang/feature/video-baseZZ.npy')
td = test_data - means
tdd = fast_dot(td, pca.components_.T)
test_data=pca.transform(test_data)
np.save('query_data_pcazz.npy', test_data)
print(test_data.shape)
print test_data

