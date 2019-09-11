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
from vilar import Vilar_model
from p_values import trunc_norm_pvalue


# choose neural network model
nnm = CNNModel(input_shape=(401,3), output_shape=(15))
# nnm = PEN_CNNModel(input_shape=(201,3), output_shape=(15), pen_nr=10)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))

nnm.load_model()


#ABC algorithm


modelname = "vilar_ACR_200_401"
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
num_timestamps=401
endtime=200

# true_param = np.ones((15))*0.8


Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)


simulate = Vilar_.simulate
print("before simulation")
data = np.array([np.squeeze(simulate(normalize_data(true_param,dmin,dmax))) for i in range(5)])
print("data shape: ", data.shape)

print("data shape: ", data.shape)
data_pred = nnm.predict(data)
data_pred = np.squeeze(data_pred)
print("data pred shape: ", data_pred.shape)
print("data_pred.std: ", np.std(data_pred,axis=0))
print("data.std: ", np.mean(np.std(data,axis=0)))


abc_trial_thetas = pickle.load(open('datasets/' + modelname + '/abc_trial_thetas.p', "rb" ) )
abc_trial_ts = pickle.load(open('datasets/' + modelname + '/abc_trial_ts.p', "rb" ) )

test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )

train_thetas, train_ts = load_spec(modelname=modelname, type = "train")
print("abc_trial_thetas shape: ", abc_trial_thetas.shape)
print("abc_trial_ts shape: ", abc_trial_ts.shape)
# abc_trial_thetas = np.concatenate((abc_trial_thetas,train_thetas,test_thetas),axis=0)
# abc_trial_ts = np.concatenate((abc_trial_ts,train_ts,test_ts),axis=0)
print("abc_trial_thetas shape: ", abc_trial_thetas.shape)
print("abc_trial_ts shape: ", abc_trial_ts.shape)

# abc_trial_thetas = normalize_data(abc_trial_thetas,dmin,dmax)
abc_trial_pred = nnm.predict(abc_trial_ts)
mean_dev = np.mean(abs(normalize_data(abc_trial_thetas,dmin,dmax)-abc_trial_pred), axis=0)
print("mean dev shape: ", mean_dev.shape)
print("mean deviation(", np.mean(mean_dev), "):: ", mean_dev)
nr_of_accept = 100
nr_of_trial = abc_trial_thetas.shape[0]



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


plt.axis('equal')
f, ax = plt.subplots(6,15,figsize=(90,20))# ,sharex=True,sharey=True)
f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
bins_nr = 10
bins = np.linspace(0,1,bins_nr+1)


dist = np.array([np.linalg.norm(abc_trial_pred - data_pred_, axis=1) for data_pred_ in data_pred])
print("dist shape: ", dist.shape)
accepted_ind =np.array([np.argpartition(dist_,nr_of_accept)[0:nr_of_accept] for dist_ in dist])
print("accepted_ind shape: ", accepted_ind.shape)

accepted_para = np.array([abc_trial_thetas[accepted_ind_] for accepted_ind_ in accepted_ind])
# accepted_mean = np.mean(accepted_para, axis=0)

# lower, upper = 0, 1
print("true param: ", true_param)
def nnlf(params, data,lower,upper):
    # print("inside nnlf: data shape: ", data.shape)
    # print("inside nnlf: lower, upper ", lower, upper)
    loc, scale = params
    left_trunc_norm = (lower - loc)/scale
    right_trunc_norm = (upper - loc) / scale
    theta = (left_trunc_norm, right_trunc_norm, loc, scale)
    value = stats.truncnorm.nnlf(theta, data)
    return value

y=0
abc_pred = np.zeros((accepted_para.shape[0],15))
density_functions = np.zeros((accepted_para.shape[0],15,100))

for accepted_para_ in accepted_para:
    print("accepted_para_ shape: ", accepted_para_.shape)
    print("y: ", y)
    for x in range(15):
        print("accepted para mean: ", np.mean(accepted_para_[:, x]))
        print("data pred: ", data_pred[y,x], ", true_param: ", true_param[x])
        ret = ax[y, x].hist(accepted_para_[:, x], density=True, color='green')
        peak_val = np.max(ret[0])
        print("y,x: ", x, y, ", peak_val: ", peak_val)
        ax[y, x].plot([true_param[x], true_param[x]], [0,peak_val], c='black', lw=4)
        # ax[0, x].plot([accepted_mean[x], accepted_mean[x]], [0,peak_val], c='red')
        # ax[y, x].plot([data_pred[y,x], data_pred[y,x]], [0,peak_val], c='orange', ls='--')

        ax[y, x].plot([dmax[x], dmax[x]], [0, peak_val], c='b')
        ax[y, x].plot([dmin[x], dmin[x]], [0, peak_val], c='b')

        loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para_[:, x]), np.std(accepted_para_[:, x])),
                                           args=(accepted_para_[:, x],dmin[x],dmax[x]), disp=False)

        left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
        right_trunc_norm = (dmax[x] - loc_opt) / scale_opt

        l = np.linspace(dmin[x], dmax[x], 100)
        p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
        density_functions[y,x]=p
        ax[y, x].plot(l, p)
        col ='red'
        if loc_opt<dmin[x] or loc_opt>dmax[x]:
            col = 'orange'
            if loc_opt<lower:
                loc_opt=lower
            if loc_opt>upper:
                loc_opt=upper
        ax[y, x].plot([loc_opt, loc_opt], [peak_val, 0], c=col, ls='--')
        abc_pred[y,x]=loc_opt
        if y == 4:
            mae = np.mean(abs(abc_pred[:,x]-true_param[x]),axis=0)
            me = np.mean(abc_pred[:,x]-true_param[x],axis=0)
            nnmae = np.mean(abs(data_pred[:,x]-true_param[x]),axis=0)
            nnme = np.mean(data_pred[:,x]-true_param[x],axis=0)
            print("mae, me: ", mae,me, "nn mea, me: ", nnmae, nnme)
            ax[0, x].set_title(para_names[x] + ", mae: " + '{0:.3f}'.format(mae) + "\n me: " + '{0:.3f}'.format(me))
    y+=1


im_post = np.prod(density_functions,axis=0)
print("im_post shape: ", im_post.shape)
for x in range(15):

    peak_val=np.max(im_post[x])
    ax[5, x].plot([true_param[x], true_param[x]], [0,peak_val], c='black', lw=4)
    ax[5, x].plot(l, im_post[x])
    ax[5, x].plot([1, 1], [0, peak_val], c='b')
    ax[5, x].plot([0, 0], [0, peak_val], c='b')


plt.savefig('posterior_abc7_clean')

mae = np.mean(abs(abc_pred-true_param),axis=0)
me = np.mean(abc_pred-true_param,axis=0)
nnmae = np.mean(abs(data_pred-true_param),axis=0)
nnme = np.mean(data_pred-true_param,axis=0)

save_results = {"abc_mae": mae, "abc_me": me, "nn_mae": nnmae, "nn_me":nnme}
pickle.dump( save_results, open( "abc_results_t08_" + str(nr_of_accept), "wb"))