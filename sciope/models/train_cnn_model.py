
import pickle

# Get the data
data_path = '/home/ubuntu/sciope/sciope/utilities/datagenerator/ds_vilar_ft100_ts501_tr1_speciesall/ds_vilar_ft100_ts501_tr1_speciesall0.p'

dataset = pickle.load(open( data_path, "rb" ) )