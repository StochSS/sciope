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
from normalize_data import normalize_data
from load_data import load_spec


# choose neural network model
# nnm = CNNModel(input_shape=(201,3), output_shape=(15))
nnm = PEN_CNNModel(input_shape=(201,3), output_shape=(15), pen_nr=10)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))

nnm.load_model()


#ABC algorithm

# new_para = np.array(prior(n=500000))
# new_data = np.expand_dims(np.array([simulate(p,n=100) for p in new_para]),2)

modelname = "vilar_ACR_100_201"
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )

new_pred = nnm.predict(abc_trial_ts)
mean_dev = np.mean(np.linalg.norm(abc_trial_thetas-new_pred, axis=1))
print("mean deviation: ", mean_dev)


data_exp = np.expand_dims( np.expand_dims(data,axis=0), axis=2 )
data_pred = nnm.predict(data_exp)
dist = np.linalg.norm( new_pred - data_pred,axis=1)
accepted_ind = np.argpartition(dist,500)[0:500]
accepted_para = abc_trial_thetas[accepted_ind]
accepted_mean = np.mean(accepted_para,axis=0)
accepted_std = np.std(accepted_para,axis=0)
print("posterior dev: ", accepted_mean-true_param)
print("posterior std: ", accepted_std)
data_pred = np.squeeze(data_pred)
accepted_dist = dist[accepted_ind]

print("accepted dist mean: ", np.mean(accepted_dist), ", max: ", np.max(accepted_dist), ", min: ", np.min(accepted_dist))


more_data =np.expand_dims( np.array([simulate(true_param) for i in range(10)]),2)


more_pred = nnm.predict(more_data)

more_pred_dev = np.linalg.norm(np.squeeze(more_pred)-true_param, axis=1)
print("more pred error: ", more_pred_dev)
print("mean more pred error: ", np.mean(more_pred_dev))

import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.axis('equal')
# col = np.zeros((500,3))
# col[:,1]=(accepted_dist - np.min(accepted_dist)) / (np.max(accepted_dist)-np.min(accepted_dist))
# circle1 = plt.Circle((true_param[0],true_param[1]),mean_dev,color='r', fill = False)
# plt.gcf().gca().add_artist(circle1)
#plt.scatter(abc_trial_thetas[:, 0], abc_trial_thetas[:, 1], color="orange", s=2)
plt.scatter(accepted_para[:, 0], accepted_para[:, 1], color="green", s=2)

plt.scatter(true_param[0],true_param[1], color="gray", marker="x")
plt.scatter(accepted_mean[0],accepted_mean[1], color="red", marker="x")
plt.scatter(data_pred[0],data_pred[1], color="gray", marker="o")
# plt.scatter(more_pred[:,0],more_pred[:,1], color="gold", marker="o")



plt.plot([-2,2,0,-2],[1,1,-1,1],color="red")
#plt.plot([-2,2,0,-2],[-1,-1,1,-1],color="red")


plt.savefig('')