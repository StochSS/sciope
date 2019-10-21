from train_vilar_all_species_func import train_routine

species = [[8],[6,8]]
datanames = ["R","C_and_R"]
for i in range(2):
    train_routine(species=species[i],dataname=datanames)