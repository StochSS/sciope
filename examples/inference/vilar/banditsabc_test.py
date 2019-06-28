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
import vilar
from sklearn.metrics import mean_absolute_error

# Load data
data = np.loadtxt("datasets/vilar_dataset_specieA_1trajs_201time.dat", delimiter=",")

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
idx=np.where(~np.isnan(ss))[1]
sum_stats.set_returning_features(idx)

# Select MAB variant
mab_algo = md.MABDirect(bandits_abc.arm_pull)

true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]


#vilar test
sim=vilar.simulate(np.array(true_params))
print("sim shape: ",sim.shape)



# Set up ABC
epsilon=0.00000001
print("epsilon: ", epsilon)


ss= sum_stats.compute(data).compute()
#ss = np.array([sum_stats.compute([data[:,i]]) for i in range(10)])
print("ss shape: ", ss.shape)
#print("ss: ", ss)
print("sum_stats features: ", sum_stats.features.keys())


abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=epsilon, prior_function=mm_prior, k=3,
                                      distance_function=dist_fun,
                                      summaries_function=sum_stats,
                                      mab_variant=mab_algo)

# Perform ABC; require 30 samples
abc_instance.infer(num_samples=200, batch_size=10)

# Results
print('Inferred parameters: ', abc_instance.results['inferred_parameters'])
print('Inference error in MAE: ', mean_absolute_error(true_params, abc_instance.results['inferred_parameters']))
print('Trial count:', abc_instance.results['trial_count'])

#added code
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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