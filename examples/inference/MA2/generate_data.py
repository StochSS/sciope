# Copyright 2019 Mattias Åkesson, Prashant Singh, Fredrik Wrede and Andreas Hellander
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
Generate data from the Moving Averages 2 (MA2) model
"""
from ma2_model import simulate, prior
import pickle
import numpy as np

sim = simulate

true_param = [0.6, 0.2]  # moving average2

data = simulate(true_param)
modelname = 'moving_average2'

n = 1000000
train_thetas = np.array(prior(n=n))
train_ts = np.expand_dims(np.array([simulate(p, n=100) for p in train_thetas]), 2)

validation_thetas = np.array(prior(n=10000))
validation_ts = np.expand_dims(np.array([simulate(p, n=100) for p in validation_thetas]), 2)

test_thetas = np.array(prior(n=10000))
test_ts = np.expand_dims(np.array([simulate(p, n=100) for p in validation_thetas]), 2)

abc_trial_thetas = np.array(prior(n=500000))
abc_trial_ts = np.expand_dims(np.array([simulate(p, n=100) for p in abc_trial_thetas]), 2)

with open('datasets/' + modelname + '/true_param.p', "wb") as f:
    pickle.dump(true_param, f)
with open('datasets/' + modelname + '/obs_data.p', "wb") as f:
    pickle.dump(data, f)

with open('datasets/' + modelname + '/train_thetas.p', "wb") as f:
    pickle.dump(train_thetas, f)
with open('datasets/' + modelname + '/train_ts.p', "wb") as f:
    pickle.dump(train_ts, f)

with open('datasets/' + modelname + '/validation_thetas.p', "wb") as f:
    pickle.dump(validation_thetas, f)
with open('datasets/' + modelname + '/validation_ts.p', "wb") as f:
    pickle.dump(validation_ts, f)

with open('datasets/' + modelname + '/test_thetas.p', "wb") as f:
    pickle.dump(test_thetas, f)
with open('datasets/' + modelname + '/test_ts.p', "wb") as f:
    pickle.dump(test_ts, f)

with open('datasets/' + modelname + '/abc_trial_thetas.p', "wb") as f:
    pickle.dump(abc_trial_thetas, f)
with open('datasets/' + modelname + '/abc_trial_ts.p', "wb") as f:
    pickle.dump(abc_trial_ts, f)
