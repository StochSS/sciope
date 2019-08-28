import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data_from_julia import load_data
import numpy as np
from AutoRegressive_model import simulate, prior
# from MovingAverage_model import simulate, prior
from sklearn.metrics import mean_absolute_error
import pickle
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
from vilar import Vilar_model
from man_statistics import peak_finder, man_statistics

# choose neural network model
# nnm = CNNModel(input_shape=(401,3), output_shape=(15))
nnm = PEN_CNNModel(input_shape=(201,3), output_shape=(15), pen_nr=10)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))

nnm.load_model()


#ABC algorithm


modelname = "vilar_ACR_200_401"
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )
print("data shape: ", data.shape)
# data_exp = np.expand_dims( np.expand_dims(data,axis=0), axis=2 )
# print("data_exp shape: ", data_exp.shape)

data_pred = nnm.predict(data)
data_pred = denormalize_data(data_pred,dmin,dmax)

center_pred = np.ones((1,15))*0.5
center_pred = denormalize_data(center_pred,dmin,dmax)

num_timestamps=401
endtime=200

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)


simulate = Vilar_.simulate

example_ts = simulate(data_pred)
center_ts = simulate(center_pred)

print("example_ts shape: ", example_ts.shape)
print("data shape: ", data.shape)

itt=10
example_ts_set = [simulate(data_pred) for i in range(itt)]
center_ts_set = [simulate(center_pred) for i in range(itt)]
data_set = [simulate(np.asarray(true_param)) for i in range(itt)]

data_set_stat=[]
for TS in data_set:
    s=man_statistics(TS)
    print("s values: ", s.values())
    data_set_stat_i = []
    for v in s.values():
        data_set_stat_i.append(v)
    data_set_stat.append(data_set_stat_i)

data_set_stat = np.asarray(data_set_stat)
print("data_set_stat shape: ", data_set_stat.shape)

data_set_stat = np.mean(data_set_stat,axis=0)



# print("data stat:", man_statistics(data))
# print("example_ts stat:", man_statistics(example_ts))
# print("center_ts stat:", man_statistics(center_ts))


colors = [[1,0,0],[0,1,0],[0,0,1]]


max_sp = np.zeros((3))
f, ax = plt.subplots(3,1,figsize=(30,30))# ,sharex=True,sharey=True)
f.suptitle('',fontsize=16)
i=0
for ts in data[0,:,:].T:
    print("ts shape: ", ts.shape)
    p = peak_finder(ts)
    print("p: ", p)
    ax[0].scatter(p,ts[p])
    max_sp[i] = np.max(ts)
    l = np.ones(ts.shape)*max_sp[i]
    ax[0].plot(l,c=colors[i])
    ax[0].plot(ts,c=colors[i])
    i += 1
print("max sp: ", max_sp)
i=0
for ts in example_ts[0,:,:].T:
    print("ts shape: ", ts.shape)
    l = np.ones(ts.shape) * max_sp[i]
    ax[1].plot(l, c=colors[i])
    ax[1].plot(ts, c=colors[i])
    i += 1

i=0
for ts in center_ts[0,:,:].T:
    print("ts shape: ", ts.shape)
    l = np.ones(ts.shape) * max_sp[i]
    ax[2].plot(l, c=colors[i])
    ax[2].plot(ts, c=colors[i])
    i += 1

plt.savefig('data_plots')


