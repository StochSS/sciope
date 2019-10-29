from train_function import train_routine


num_timestamps=401
endtime=200
modelname = "vilar_ACR_prior6_" + str(endtime) + "_" + str(num_timestamps)
# parameter range
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
species = [0]
dataname = "C"
models = ['CNN', 'PEN', 'DNN']
for i in range(3):
    train_routine(modelname=modelname, dmin=dmin, dmax=dmax, species=species,dataname=dataname, model=models[i])


