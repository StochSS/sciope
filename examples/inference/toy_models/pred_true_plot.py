import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def heatmap(true_thetas,pred_thetas, dmin, dmax, true_point=None, pred_point=None):

    f, ax = plt.subplots(3,5,figsize= (50,20))

    for x in range(3):
        for y in range(5):
            j = x * 5 + y
            ax[x,y].hist2d(true_thetas[:,j],pred_thetas[:,j],bins=50)

    plt.savefig('true_pred_plot')


