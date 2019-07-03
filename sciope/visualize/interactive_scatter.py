# Imports
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.text import Annotation
from pandas import DataFrame
import numpy as np
from ipywidgets import widgets
from IPython.display import display


class UserLabel():  # pragma: no cover
    def __init__(self):
        self.highlighted = []

    def clear(self):
        self.highlighted = []

    def add(self, x):
        self.highlighted.append(x)


def annotate(axis, text, x, y): # pragma: no cover
    """Create annotation at position (x,y) """
    text_annotation = Annotation(text, xy=(x, y), xycoords='data')
    axis.add_artist(text_annotation)


def draw_scatterplot(axis, x_values, y_values, labels): # pragma: no cover
    "Plot scatter with given values"
    axis.scatter(
        x_values,
        y_values,
        c=labels,
        picker=True
    )


class UserLabel():  # pragma: no cover
    """Class to keep track of highligted points, will be used for interactive labeling"""

    def __init__(self):
        self.highlighted = []

    def clear(self):
        self.highlighted = []

    def add(self, x):
        self.highlighted.append(x)


def interative_scatter(scatter_data, data_class=None):  # pragma: no cover
    """interactive plot taken from:
    https://medium.com/@gorjanz/data-analysis-in-python-interactive-scatterplot-with-matplotlib-6bb8ad2f1f18

         """
    assert hasattr(data_class,
                   'user_labels'), "For interative exploration to work the DataSet class need attribute 'user_labels'"
    assert scatter_data.shape[0] == data_class.user_labels.shape[
        0], "scatter_data and data contained in DataSet class need the same number of points"

    instances_colors = data_class.user_labels
    axis_values_x = scatter_data[:, 0]
    axis_values_y = scatter_data[:, 1]
    user_labels = UserLabel()

    # draw a subplots
    fig, [[ax, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 8))
    plotting_axes = [ax2, ax3, ax4]

    # draw the initial scatterplot
    draw_scatterplot(ax, axis_values_x, axis_values_y, instances_colors)

    # draw trajectory function
    def draw_trajectory(axis, idx, species_idx):    # pragma: no cover
        ts_data = data_class.ts[idx]
        ts_data = ts_data[:,species_idx,:]
        for point in ts_data:
            axis.plot(point)
        axis.figure.canvas.draw_idle()

    # define the behaviour -> what happens when you pick a dot on the scatterplot by clicking close to it
    def onpick(event):  # pragma: no cover
        # step 1: take the index of the dot which was picked
        ind = event.ind

        # step 2: save the actual coordinates of the click, so we can position the text label properly
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata

        # just in case two dots are very close, this offset will help the labels not appear one on top of each other
        offset = 0.01  # TODO this offset has to be scaled to the data, or the data has to be scaled

        # if the dots are to close one to another, a list of dots clicked is returned by the matplotlib library
        for i in ind:
            # step 3: take the label for the corresponding instance of the data
            label = instances_colors[i]
            user_labels.add(i)

            # step 4: log it for debugging purposes
            print("index", i, label)

            # step 5: create and add the text annotation to the scatterplot
            annotate(
                ax,
                (i, label),
                label_pos_x + offset,
                label_pos_y + offset
            )

            # step 6: force re-draw
            ax.figure.canvas.draw_idle()

            # alter the offset just in case there are more than one dots affected by the click
            offset += 0.01

        # lastly lets draw the trajectory (we can only draw one, so we take the last in "ind")

        for e, plots in enumerate(trajectory_plots):
            draw_trajectory(plotting_axes[e], ind[-1], int(plots.value))

    # connect the click handler function to the scatterplot
    fig.canvas.mpl_connect('pick_event', onpick)

    # create the "clear all" button, and place it somewhere on the screen
    # ax_clear_all = plt.axes([0.0, 0.0, 0.1, 0.05])
    # button_clear_all = Button(ax_clear_all, 'Clear all')
    button_clear_all = widgets.Button(description='Clear all')
    display(button_clear_all)

    # define the "clear all" behaviour
    def onclick(event): # pragma: no cover
        # step 1: we clear all artist object of the scatter plot
        ax.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        # clear highlighted points
        user_labels.clear()

        # step 2: we re-populate the scatterplot only with the dots not the labels
        draw_scatterplot(ax, axis_values_x, axis_values_y, instances_colors)

        # step 3: we force re-draw
        ax.figure.canvas.draw_idle()

    # link the event handler function to the click event on the button
    button_clear_all.on_click(onclick)

    # For interactive labeling of highlighted points
    def set_user_label(event):  # pragma: no cover
        instances_colors[user_labels.highlighted] = user_label_slider.value
        onclick('')

    user_label_slider = widgets.IntSlider(min=-1, max=10)
    button_submit = widgets.Button(description='Submit')
    display(button_submit, user_label_slider)

    button_submit.on_click(set_user_label)

    # Widgets for plotting trajectories
    species = {}
    if 'listOfSpecies' in data_class.configurations.keys():
        indices = data_class.configurations['listOfSpecies']
        for e, s in enumerate(indices):
            species[s] = e

    else:
        indices = range(data_class.ts.shape[2])
        for i in indices:
            species['Index ' + str(i)] = i

    # create drowpdown widgets for selecting species
    trajectory_plots = []
    for key, val in species.items():
        plot = widgets.Dropdown(
            options=species,
            value=val,
            description='Species plot ' + str(val + 1),
            disabled=False,
        )
        trajectory_plots.append(plot)
        display(plot)

    # initial drawing of the scatterplot
    plt.plot()

    # use option_context to limit display 
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    if 'listOfParameters' in data_class.configurations.keys():
        display(DataFrame(data_class.x, columns=data_class.configurations['listOfParameters']))

    # present the scatterplot
    plt.show()
