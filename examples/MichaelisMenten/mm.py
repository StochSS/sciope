import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import gillespy

class MichaelisMenten(gillespy.Model):
    """
        This is an example showcasing a simple Michaelis Menten reaction.
    """

    def __init__(self, parameter_values=[1.0, 110.0, 10.0, 10.0]):

        # Initialize the model.
        gillespy.Model.__init__(self, name="MichaelisMenten")
        #print parameter_values
        #print parameter_values.shape
        
        # Parameters
        k1 = gillespy.Parameter(name='k1', expression=parameter_values[0])
        Km = gillespy.Parameter(name='Km', expression=parameter_values[1])
        mu = gillespy.Parameter(name='mu', expression=parameter_values[2])
        Vmax = gillespy.Parameter(name='Vmax', expression=parameter_values[3])
        
        self.add_parameter([k1,Km,mu,Vmax])
        
        # Species
        S = gillespy.Species(name='S', initial_value=0)
        P = gillespy.Species(name='P', initial_value=0)
        
        self.add_species([S,P])
        
        # Reactions
        rxn1 = gillespy.Reaction(
                name = 'S production',
                reactants = {},
                products = {S:1},
                rate = k1 )
                

        rxn2 = gillespy.Reaction(
                name = 'P production',
                reactants = {P:1},
                products = {},
                rate = mu )

        rxn3 = gillespy.Reaction(
                name = 'S conversion to P',
                reactants = {S:1},
                products = {P:1},
                rate = mu )

        self.add_reaction([rxn1,rxn2,rxn3])
        self.timespan(np.linspace(0,20,101))



if __name__ == '__main__':

    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = MichaelisMenten([1.0, 110.0, 10.0, 10.0])
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    num_trajectories = 250
    '''
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract time values
    time = np.array(simple_trajectories[0][:,0]) 

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    
    meanTrajs = S_trajectories.mean(1);
    print meanTrajs.item(9)
    '''
    # Generate some data for parameter inference
    numDataPoints = 2000
    dataToWrite = np.ndarray(shape=(numDataPoints, 1), dtype=float, order='F') * 0
    idx=0
    for x in range(0, numDataPoints):
    	# run the model
    	simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    	
    	# extract just the trajectories for S into a numpy array
    	S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    	meanTrajs = S_trajectories.mean(1);
    	dataToWrite[idx] = meanTrajs.item(9)
    	idx += 1
    	
    # Write it to file
    np.savetxt("mmDataset.dat", dataToWrite, delimiter=",")

def simulate(param):
    # If the parameters are not within sensible range, complain about it
    if np.any(param < 0):
    	return 99999999
    
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = MichaelisMenten(parameter_values=param)
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    num_trajectories = 250
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract time values
    time = np.array(simple_trajectories[0][:,0]) 

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    
    # Save values to text - 10th value of mean
    meanTrajs = S_trajectories.mean(1);
    simulatedValue = meanTrajs.item(9)
    return simulatedValue

    
def simulateTS(param):
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = MichaelisMenten(parameter_values=param)
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    num_trajectories = 250
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract time values
    time = np.array(simple_trajectories[0][:,0]) 

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    
    meanTrajs = S_trajectories.mean(1);
    return meanTrajs


