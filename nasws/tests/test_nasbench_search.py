
from nasws.cnn.search_space.nasbench101.model_search import *
from nasws.cnn.search_space.nasbench101.model import NasBenchNet as NasbenchNetOriginal

from nasws.cnn.search_space.nasbench101.nasbench_api_v2 import NASBench_v2

nasbench = NASBench_v2('/home/yukaiche/pycharm/automl/data/nasbench/nasbench_only108.tfrecord', only_hash=True)

#%%
hashs = []
ind = 0
for ind, (k, v) in enumerate(nasbench.hash_dict.items()):
     if ind < 10:
         # print(k, v)
         hashs.append(k)

_hash = hashs[0]
_hash2 = hashs[1]

input_channels = 3
# print(nasbench.hash_to_model_spec(_hash))
spec_1 = nasbench.hash_to_model_spec(_hash)
spec_2 = nasbench.hash_to_model_spec(_hash2)

model1 = NasbenchNetOriginal(input_channels, spec_1)
model2 = NasbenchNetOriginal(input_channels, spec_2)
#%%
model_search = NasBenchNetSearch(input_channels, spec_1)


x = torch.randn(8, 3, 32, 32)

# y = model_search(x)
#%%
model_search.change_model_spec(spec_1)
#%%
stem = model_search.stem(x)
layer0 = model_search.stacks['stack0']['module0']
# proj_op = layer0.op['vertex_1'].proj_ops[0]
# print(proj_op)
# print(layer0.op['vertex_1'].proj_ops[0](stem).size())

# print(layer0.op['vertex_1'](stem))
print(layer0(stem))


model_search.change_model_spec(spec_2)
stem = model_search.stem(x)
layer0 = model_search.stacks['stack0']['module0']
print(layer0(stem))
