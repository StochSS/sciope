from train_vilar_all_species_func import train_routine

completed_subsets = [[0,1],[6,8],[5,7],[0,2]]

for s1 in range(8):
    for s2 in range(s1+1,9):
        if [s1,s2] not in completed_subsets:
            train_routine(species=[s1,s2])