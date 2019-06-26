# Copyright 2019 Fredrik Wrede, Prashant Singh and Andreas Hellander
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
The vilar Model: Approximate Bayesian Computation Test Run
"""

from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference
import summaries_tsa as tsa
import numpy as np
import vilar
from sklearn.metrics import mean_absolute_error
import dask
from sciope.utilities.distancefunctions.euclidean import EuclideanDistance

# Load the observed/fixed dataset
data = np.loadtxt("datasets/vilar_dataset_specieA_50trajs_15time.dat", delimiter=",")

# Set up summary stats
sum_stats = tsa.SummariesTSFRESH()

# Setup the prior
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
euc = EuclideanDistance(use_logger=False)

# Set up ABC
abc_instance = abc_inference.ABC(data, vilar.simulate, epsilon=0.01, prior_function=mm_prior,
                                 summaries_function=sum_stats, distance_function=euc)

inf_res = abc_instance.infer(num_samples=200, batch_size=100)

# Compare results
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
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

f.savefig('subplot_posteriorhistograms_tsfresh_minimal')