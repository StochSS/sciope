# # # # # # # # # # # # # # #
# Simulation: The Vilar model
# # # # # # # # # # # # # # #
#
# 5 trajectories, averaged 
# and the output is also averaged 
#
# Initialize
import gillespy 
import matplotlib.pyplot as plt
import numpy as np

# Load the model definition
model_doc = gillespy.StochMLDocument.from_file("StochSS_model/vilar_oscillator_AIYDNg/models/data/vilar_oscillator.xml")


# Create the model from definition
model = model_doc.to_model("Vilar")
numTrajs = 5
numTimeStamps = 1000

testParam = model.get_parameter('alpha_A')
testParam.set_expression(50.0)

testParam = model.get_parameter('alpha_a_prime')
testParam.set_expression(500.0)

testParam = model.get_parameter('alpha_r')
testParam.set_expression(0.01)

testParam = model.get_parameter('alpha_r_prime')
testParam.set_expression(50.0)

testParam = model.get_parameter('beta_a')
testParam.set_expression(50.0)

testParam = model.get_parameter('beta_r')
testParam.set_expression(5.0)

testParam = model.get_parameter('delta_ma')
testParam.set_expression(10.0)

testParam = model.get_parameter('delta_mr')
testParam.set_expression(0.5)

testParam = model.get_parameter('delta_a')
testParam.set_expression(1.0)

testParam = model.get_parameter('delta_r')
testParam.set_expression(0.2)

testParam = model.get_parameter('gamma_a')
testParam.set_expression(1.0)

testParam = model.get_parameter('gamma_r')
testParam.set_expression(1.0)

testParam = model.get_parameter('gamma_c')
testParam.set_expression(2.0)

testParam = model.get_parameter('Theta_a')
testParam.set_expression(50.0)

testParam = model.get_parameter('Theta_r')
testParam.set_expression(100.0)

# Set parameters
# model.set_parameter('alpha_A', 50.0)
# model.set_parameter('alpha_a_prime', 500)
# model.set_parameter('alpha_r', 0.01)
# model.set_parameter('alpha_r_prime', 50)
# model.set_parameter('beta_a', 50)
# model.set_parameter('beta_r', 5)
# model.set_parameter('delta_ma', 10)
# model.set_parameter('delta_mr', 0.5)
# model.set_parameter('delta_a', 1)
# model.set_parameter('delta_r', 0.2)
# model.set_parameter('gamma_a', 1)
# model.set_parameter('gamma_r', 1)
# model.set_parameter('gamma_c', 2)
# model.set_parameter('Theta_a', 50)
# model.set_parameter('Theta_r', 100)


# Simulate
model.tspan = range(numTimeStamps)
res = model.run(number_of_trajectories=numTrajs)
#print res
#res = model.run()

# Post-process
colSum = np.ndarray(shape=(1000,10)) * 0
for i in res:
	colSum = np.add(colSum, i)
	
# Average over trajectories
colSum = colSum / numTrajs

# Average count of each species over 1000 time steps
out = colSum.mean(axis=0)
print out.shape
print out

# Plot it!
#plt.plot(res[0][:,0], res[0][:,1:-1])
plt.plot(colSum[:,0], colSum[:,1:-1])
plt.show()


