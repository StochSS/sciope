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
import vilar
from scipy import stats, optimize



# choose neural network model
# nnm = CNNModel(input_shape=(401,3), output_shape=(15))

# nnm = PEN_CNNModel(input_shape=(201,3), output_shape=(15), pen_nr=10)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))
clay=[32,48,64,96]

num_timestamps=401
endtime=200

modelname = "vilar_ACR_200_401"
# parameter range
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )

# choose neural network model
nnm = CNNModel(input_shape=(test_ts.shape[1],test_ts.shape[2]), output_shape=15, con_len=3, con_layers=clay, dense_layers=[200,200,200],dataname='vilar_prior1')
nnm.load_model()


#ABC algorithm



print("new example")

true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
data = pickle.load(open('datasets/' + modelname + '/obs_data_pack.p', "rb" ) )
data = data[3]
data = np.expand_dims(data,0)
print("data shape: ", data.shape)
# data_exp = np.expand_dims( np.expand_dims(data,axis=0), axis=2 )
# print("data_exp shape: ", data_exp.shape)

data_pred = nnm.predict(data)
data_pred_denorm = denormalize_data(data_pred,dmin,dmax)



# test_thetas, test_ts = load_spec(modelname=modelname, type = "train")

print("test test data example")
test_thetas_min = np.min(test_thetas,0)
test_thetas_min_text = ["{0:.1f}".format(s) for s in test_thetas_min]

test_thetas_max = np.max(test_thetas,0)
test_thetas_max_text = ["{0:.1f}".format(s) for s in test_thetas_max]

print("min: ", test_thetas_min_text)
print("max: ", test_thetas_max_text)

test_pred = nnm.predict(test_ts)

pred_min = np.min(test_pred,0)
pred_min_text = ["{0:.1f}".format(s) for s in pred_min]
pred_max = np.max(test_pred,0)
pred_max_text = ["{0:.1f}".format(s) for s in pred_max]


print("predict min: ", pred_min_text)
print("predict max: ", pred_max_text)

para_names = vilar.get_parameter_names()

for i in range(15):
    pk = para_names[i]
    pks = pk.split("_")
    if len(pks) > 1:
        pk_p = "\hat{\\" + pks[0].lower() + "}_{" + pks[1].upper() + "}"
        pk = pks[0].lower() + "_{" + pks[1].upper() + "}"
    if len(pks) == 3:
        print("len 3: ", pks[2])
        if pks[2] == 'prime':
            pk_p = pk_p + "'"
            pk = pk + "'"

    para_name_p = "$" + pk_p + "$"
    para_names[i] = "$\\" + pk + "$"

print("test pred shape: ", test_pred.shape)
f, ax = plt.subplots(3,5,figsize=(30,10))
bins = np.linspace(-2,2,41)
for x in range(3):
    for y in range(5):
        i = x*5+y
        ax[x, y].set_title("parameter: " + para_names[i])
        d = ax[x, y].hist(test_pred[:,i], bins=bins)
        peak_val = np.max(d[0])
        ax[x, y].plot([0, 0], [peak_val,0])
        ax[x, y].plot([1, 1], [peak_val, 0])

plt.savefig('distplot6')

test_pred_denorm = denormalize_data(test_pred,dmin,dmax)
test_thetas_norm = normalize_data(test_thetas,dmin,dmax)
mean_dev = np.mean(abs(test_thetas_norm-test_pred), axis=0)
print("mean dev shape: ", mean_dev.shape)
print("mean deviation(", np.mean(mean_dev), "):: ", mean_dev)

bpi = np.argsort(mean_dev)[:4] # best_param_ind


lwidth = 3
f,ax = plt.subplots(3,5,figsize=(50,30))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
for x in range(3):
    for y in range(5):
        i = x*5+y
        ax[x,y].set_title(para_names[i])
        # ax[x,y].scatter(test_thetas[:,i],test_pred_denorm[:,i],s=0.4,alpha=0.1)
        # ax[x,y].hist2d(test_thetas[:,i],test_pred_denorm[:,i],bins=50)
        ax[x,y].hist2d(test_thetas_norm[:,i],test_pred[:,i],bins=20)

        # ax[x,y].plot([dmin[i], dmax[i]],[dmin[i],dmax[i]],c='green',ls='--',lw=lwidth)
        # ax[x, y].plot([dmin[i], dmin[i], dmax[i], dmax[i], dmin[i]], [dmin[i], dmax[i], dmax[i], dmin[i], dmin[i]],
        #               lw=lwidth, c='b')

        ax[x, y].plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0],
                      lw=lwidth, c='b')

plt.savefig('accuracyplot6')
print("new save")