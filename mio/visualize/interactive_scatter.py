# Imports
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.text import Annotation
import seaborn as sns
import numpy as np
from ipywidgets import widgets
from IPython.display import display




def interactive_plot(self, data,indices,labels,title, species):

        def onpick3(event):
            ind = event.ind
            self.current_ind = ind
            text.set_position((data[ind,0], data[ind,1]))
            text.set_text(str(ind) + str(labels[ind])+ str(self.user_labels[ind]))

            #ts = ts[ind]
            ts = ts_data[ind].T[sp_idx] #species
            t = ts_data[ind].T[0] #timepoints

            maxi = str(int(max(ts)[0]))
            mini = str(int(min(ts)[0]))

            ax2.set_xlim(0,max(t))
            ax2.set_ylim(0,1.15)
            inc = max(t)/10
            ax2.set_xticks(np.arange(min(t),max(t)+inc, inc))

            ts = (ts - min(ts))/float(max(ts) - min(ts))
            ax2.set_yticks(np.arange(min(ts), max(ts)+.1, 0.5))

            time_s.set_visible(True)
            time_s.set_ydata(ts)
            time_s.set_xdata(t)

            arg_max =np.argmax(ts)
            arg_min = np.argmin(ts)

            max_plot.set_ydata(ts[arg_max])
            max_plot.set_xdata(t[arg_max])
            max_plot.set_label('Max: '+maxi)

            ax2.legend(loc='upper right', shadow=True)

            fig.canvas.draw()
        
        N = len(np.unique(labels))
        ts_data = self.data[self.batch_idx[0]:self.batch_idx[1]]
        sp_idx = self.ts_analysis.data.columns.get_loc(species) - 2 # -2 since .data contains 2 extra columns
        if N < len(data)/2 :
            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(0,N,N+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


        test = ts_data[0].T[sp_idx]

        fig, [[ax,ax2],[ax3,ax4]] = plt.subplots(2, 2, figsize=(15,7), gridspec_kw = {'width_ratios':[1, 1.3], 'height_ratios': [1, 0.1]})
        text = ax.text(0, 0, "", va="bottom", ha="left")
        text.set_text('test')


        if N < len(data)/2:
            line = ax.scatter(data[:, 0], data[:, 1], picker=True, c=labels,cmap=cmap,norm=norm)
        else:
            line = ax.scatter(data[:, 0], data[:, 1], picker=True, c=labels)
            plt.colorbar(line)
        time_s,  = ax2.plot(test, visible=True, alpha=0.7)

        max_plot, = ax2.plot(0,0,'ro', markersize= 12, label='Max')


        ax2.set_xlabel("time (hours)", style='italic', size=10)
        ax2.set_ylabel("copy number",style='italic', size=10)

        ax2.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        max_data = int(max(data[:, 0]))
        min_data = int(min(data[:, 0]))
        #ax.set_xticks(np.arange(min_data,max_data + 1, (max_data - min_data)/5))

        max_data = int(max(data[:, 1]))
        min_data = int(min(data[:, 1]))
        #ax.set_yticks(np.arange(min_data,max_data + 1, (max_data - min_data)/5))

        def submit_idx(text):
            label = int(text)
            self.user_labels[self.current_ind] = label
            text_box_idx.set_val('')
            fig.canvas.draw()

        def submit_cluster(text):
            label = int(text)
            idxs, = np.where(labels == labels[self.current_ind])
            self.user_labels[idxs] = label
            text_box_cluster.set_val('')
            line.set_array(self.user_labels)
            fig.canvas.draw()


        #axbox = plt.axes([0.1, 0.01, 0.3, 0.05])
        text_box_idx = TextBox(ax3, 'Label point')
        text_box_cluster = TextBox(ax4, 'Label cluster')


        def enter_axes(event):
            text_box_idx.on_submit(submit_idx)
            text_box_cluster.on_submit(submit_cluster)
            event.canvas.draw()

        fig.canvas.mpl_connect('axes_enter_event', enter_axes)
        fig.canvas.mpl_connect('pick_event', onpick3)

        plt.show()

def interative_scatter(data, labels):
    """interactive plot taken from:
    https://medium.com/@gorjanz/data-analysis-in-python-interactive-scatterplot-with-matplotlib-6bb8ad2f1f18
         """
    sns.set()

    print("now visualizing scatterlplot...")

    # add the values one by one to the scatterplot
    
    instances_colors = labels
    axis_values_x = data[:,0]
    axis_values_y = data[:,1]


    # draw a scatter-plot of the generated values
    fig = plt.figure(figsize=(20, 16))
    ax = plt.subplot()


    # extract the scatterplot drawing in a separate function so we ca re-use the code
    def draw_scatterplot():
        ax.scatter(
            axis_values_x,
            axis_values_y,
            c=instances_colors,
            picker=True
        )


    # draw the initial scatterplot
    draw_scatterplot()


    # create and add an annotation object (a text label)
    def annotate(axis, text, x, y):
        text_annotation = Annotation(text, xy=(x, y), xycoords='data')
        axis.add_artist(text_annotation)


    # define the behaviour -> what happens when you pick a dot on the scatterplot by clicking close to it
    def onpick(event):
        # step 1: take the index of the dot which was picked
        ind = event.ind

        # step 2: save the actual coordinates of the click, so we can position the text label properly
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata

        # just in case two dots are very close, this offset will help the labels not appear one on top of each other
        offset = 0

        # if the dots are to close one to another, a list of dots clicked is returned by the matplotlib library
        for i in ind:
            # step 3: take the label for the corresponding instance of the data
            label = labels[i]

            # step 4: log it for debugging purposes
            print("index", i, label)

            # step 5: create and add the text annotation to the scatterplot
            annotate(
                ax,
                label,
                label_pos_x + offset,
                label_pos_y + offset
            )

            # step 6: force re-draw
            ax.figure.canvas.draw_idle()

            # alter the offset just in case there are more than one dots affected by the click
            offset += 0.01


    # connect the click handler function to the scatterplot
    fig.canvas.mpl_connect('pick_event', onpick)

    # create the "clear all" button, and place it somewhere on the screen
    #ax_clear_all = plt.axes([0.0, 0.0, 0.1, 0.05])
    #button_clear_all = Button(ax_clear_all, 'Clear all')
    button_clear_all = widgets.Button(description='Clear all')
    display(button_clear_all)

    # define the "clear all" behaviour
    def onclick(event):
        # step 1: we clear all artist object of the scatter plot
        ax.cla()

        # step 2: we re-populate the scatterplot only with the dots not the labels
        draw_scatterplot()

        # step 3: we force re-draw
        ax.figure.canvas.draw_idle()


    # link the event handler function to the click event on the button
    #button_clear_all.on_clicked(onclick)
    button_clear_all.on_click(onclick)

    # initial drawing of the scatterplot
    plt.plot()
    print ("scatterplot done")

    # present the scatterplot
    plt.show()



    