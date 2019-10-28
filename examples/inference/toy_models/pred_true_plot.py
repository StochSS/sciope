import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def heatmap(true_thetas,pred_thetas, dmin, dmax, true_point=None, pred_point=None,name=""):

    f, ax = plt.subplots(3,5,figsize= (50,20))

    for x in range(3):
        for y in range(5):
            j = x * 5 + y
            ax[x,y].plot([dmin[j], dmin[j], dmax[j], dmax[j], dmin[j]],[dmin[j], dmax[j], dmax[j], dmin[j], dmin[j]],c='w',lw=4)
            ax[x,y].hist2d(true_thetas[:,j],pred_thetas[:,j],bins=21,cmap='terrain', range=[[dmin[j], dmax[j]],[dmin[j], dmax[j]]])
            ax[x,y].plot([dmin[j], dmin[j], dmax[j], dmax[j], dmin[j]],[dmin[j], dmax[j], dmax[j], dmin[j], dmin[j]],c='w',lw=4)
            ax[x,y].plot([dmin[j],dmax[j]],[dmin[j],dmax[j]], lw=5, c='w', ls=':')

            if pred_point is not None:
                # print("j: ", j, ", true_point[j]: ", true_point[j])
                for pp in pred_point[:,j]:
                    # print("pp: ", pp)
                    ax[x,y].scatter(true_point[j], pp, c='r', s=50, alpha=1)


    plt.savefig('true_pred_plot'+name)


def heatmap2(true_thetas,pred_thetas, dmin, dmax, true_point=None, pred_point=None,name=""):

    f, ax = plt.subplots(3,5,figsize= (50,20))
    nr = 10
    print("true_thetas shape: ", true_thetas.shape)

    for x in range(3):
        for y in range(5):
            j = x * 5 + y

            bins = np.linspace(dmin[j],dmax[j],nr+1)
            dx = (dmax[j]-dmin[j])/nr
            print("dx: ", dx)
            image = []
            for i in range(nr):
                print("bins[",i,"]: ", bins[i])
                print("true thetas min: ", np.min(true_thetas[:,j]))
                ind = np.where(abs(true_thetas[:,j]-bins[i]+dx/2) < dx/2)



                print("ind shape: ", true_thetas[ind,j].shape)
                print("pred thetas min/max: ", np.min(pred_thetas[ind,j]), np.max(pred_thetas[ind,j]), ", shape: ", pred_thetas[ind,j].shape)
                ret=np.histogram(pred_thetas[ind,j],bins=bins,density=True)
                image.append(ret[0])
            image = np.array(image).T
            print("image shape: ", image.shape)
            ax[x,y].imshow(image,cmap='terrain',extent=[dmin[j],dmax[j],dmin[j],dmax[j]])





    plt.savefig('heatmap2'+name)
