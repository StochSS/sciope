# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vilar Model: Multi-Armed Bandits based Approximate Bayesian Computation Test Run
"""

# Imports
from sciope.utilities.priors import uniform_prior
from sciope.inference import bandits_abc
from sciope.utilities.distancefunctions import naive_squared as ns
import summaries_tsa as tsa
from sciope.utilities.mab import mab_direct as md


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vilar
from sklearn.metrics import mean_absolute_error
import dask

# Load data
data = np.loadtxt("datasets/vilar_dataset_specieA_1trajs_201time.dat", delimiter=",")

true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
print("true param shape: ", np.array(true_params).shape)

# Set up the prior
dmin = [30, 200, 0, 30, 30-20, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80-30]
dmax = [70+50, 600, 1+2, 70, 70, 10+5, 12+5, 1+2, 2+5, 0.5+1, 1.5+2, 1.5+1, 3+5, 70+40, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
dist_fun = ns.NaiveSquaredDistance(use_logger=False)

# Set up summaries
sum_stats = tsa.SummariesTSFRESH()


#Check data shape
print("data shape: ", data.shape)
data = np.array([data])
print("data shape: ", data.shape)

# Removing Nan values from summaries list
ss=sum_stats.compute(data).compute()
idx=np.where(~np.isnan(ss))
# Removing small values from summaries list
idxx=np.where(abs(ss[idx])>10)
#print("idxx[0]: ", idxx[0])
idxxx=idx[1]#[idxx[0]]

#print("ss[0][idxxx]: ", ss[0][idxxx])
print("ss shape: ", ss.shape)
#print("idxxx:", idxxx)

sum_stats.set_returning_features(idxxx)

ss=sum_stats.compute(data).compute()

#Generate some points to ensure summary statistics to not be constants

trial_param = [mm_prior.draw() for i in range(5)]
trial_param = dask.compute(trial_param)
trial_sim = [vilar.simulate(np.array(t)) for t in trial_param]
trial_ss = [sum_stats.compute(s) for s in trial_sim]
trial_dist = [dist_fun.compute(ss,s) for s in trial_ss]
trial_dist = dask.compute(trial_dist)
#print("trial_dist shape: ", np.array(trial_dist).shape)
#print("max dist: ", np.max(np.array(trial_dist),axis=0))
max_dist=np.max(np.array(trial_dist),axis=0).squeeze()
#print("max dist shape", max_dist.shape)
#print("min max: ", np.sort(max_dist))

# Removing small distances from summaries list
idxx=np.where(abs(np.min(trial_dist,axis=0)>1000))
#print("idxx: ", idxx)
#print("idxxx: ", idxxx)
idxxx2=idxxx[idxx[2]]
#print("idxxx shape: ", idxxx.shape)

sum_stats.set_returning_features(idxxx2)

ss=sum_stats.compute(data).compute()
print("trial_param shape: ", np.array(trial_param).shape)
tp = np.array(trial_param).squeeze()
print("tp shape: ", tp.shape)

plt.scatter(tp[:,0],tp[:,1])


trial_ss = [sum_stats.compute(s) for s in trial_sim]
trial_dist = [dist_fun.compute(ss,s) for s in trial_ss]
trial_dist = dask.compute(trial_dist)

print("trial_dist after reductions: ", trial_dist)
print("trial dist shape: ", trial_dist.shape)
max_dist=np.max(np.array(trial_dist),axis=0).squeeze()

print("sorted dist: ", np.sort(max_dist))

trial_param = [mm_prior.draw() for i in range(50)]
trial_param = dask.compute(trial_param)
trial_sim = [vilar.simulate(np.array(t)) for t in trial_param]
trial_ss = [sum_stats.compute(s) for s in trial_sim]
trial_dist = [dist_fun.compute(ss,s) for s in trial_ss]
trial_dist = dask.compute(trial_dist)
max_dist=np.max(np.array(trial_dist),axis=0).squeeze()
print("sorted dist: ", np.sort(max_dist))


# Select MAB variant
mab_algo = md.MABDirect(bandits_abc.arm_pull)



#vilar test
sim=vilar.simulate(np.array(true_params))
print("sim shape: ",sim.shape)



# Set up ABC
epsilon=0.000001
print("epsilon: ", epsilon)


ss= sum_stats.compute(data).compute()
#ss = np.array([sum_stats.compute([data[:,i]]) for i in range(10)])
print("ss shape: ", ss.shape)
#print("ss: ", ss)
#print("sum_stats features: ", sum_stats.features.keys())


abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=epsilon, prior_function=mm_prior, k=5,
                                      distance_function=dist_fun,
                                      summaries_function=sum_stats,
                                      mab_variant=mab_algo)

# Perform ABC; require 30 samples
abc_instance.infer(num_samples=20, batch_size=50)

# Results
print('Inferred parameters: ', abc_instance.results['inferred_parameters'])
print('Inference error in MAE: ', mean_absolute_error(true_params, abc_instance.results['inferred_parameters']))
print('Trial count:', abc_instance.results['trial_count'])

dist=np.array(abc_instance.results['distances'])
print("dist shape: ", dist.shape)
print('Mean distance:', np.mean(dist))
#added code


#shape = (sample,parameter)
posterior = np.array(abc_instance.results['accepted_samples']).squeeze()

f, axes = plt.subplots(3, 5)
f.set_figheight(10 * 3)
f.set_figwidth(10 * 5)
fontsize = 46
postmax_dev = []
postmean_dev = []
pred_dev = []
i = 0
for j in range(5):
    for k in range(3):

       # pk = para_k[i]
        #pks = pk.split("_")
        #if len(pks) > 1:
        #    pk_p = "\hat{\\" + pks[0].lower() + "}_{" + pks[1].upper() + "}"
        #    pk = pks[0].lower() + "_{" + pks[1].upper() + "}"
        #if len(pks) == 3:
        #    print("len 3: ", pks[2])
        #    if pks[2] == 'prime':
        #        pk_p = pk_p + "'"
        #        pk = pk + "'"

        #para_name_p = "$" + pk_p + "$"
        #para_name = "$\\" + pk + "$"

        #axes[k][j].set_xlim(0, 1)
        #axes[k][j].set_title('' + para_name, fontsize=fontsize)
        axes[k][j].hist(posterior[:, i], 10, density=True)
        t = np.linspace(0, 1, 100)

        ax_ymin, ax_ymax = axes[k][j].get_ylim()
        axes[k][j].vlines(true_params[0][i], ax_ymin, ax_ymax, label='target')
        # axes[i].vlines(post_max, ax_ymin, ax_ymax,color='orange',label='maximum ABC posterior ')
        # axes[i].vlines(post_mean, ax_ymin, ax_ymax,color='red',label='mean ABC posterior ')
        axes[k][j].vlines(abc_instance.results['inferred_parameters'][0][i], ax_ymin, ax_ymax, color='green', label='predicted')
        axes[k][j].legend()

        i += 1

        # print("i: ", i)
# postmax_dev = np.array(postmax_dev)
# postmean_dev = np.array(postmean_dev)
# pred_dev = np.array(pred_dev)

f.savefig('subplot_posteriorhistograms_bandits')