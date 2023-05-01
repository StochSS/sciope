# Copyright (C) 2017-2023 Prashant Singh, Fredrik Wrede and Andreas Hellander
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Imports for summary statistics module.
'''
import os
import csv
from collections import UserDict, UserList

import numpy
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.text as mtext

try:
    import plotly
    from plotly import subplots
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False

from sciope.utilities.housekeeping import sciope_logger as ml
log = ml.SciopeLogger().get_logger()

common_rgb_values = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff',
    '#800000', '#808000', '#008000', '#800080', '#008080', '#000080', '#ff9999', '#ffcc99',
    '#ccff99', '#cc99ff', '#ffccff', '#62666a', '#8896bb', '#77a096', '#9d5a6c', '#9d5a6c',
    '#eabc75', '#ff9600', '#885300', '#9172ad', '#a1b9c4', '#18749b', '#dadecf', '#c5b8a8',
    '#000117', '#13a8fe', '#cf0060', '#04354b', '#0297a0', '#037665', '#eed284', '#442244',
    '#ffddee', '#702afb'
]

def combine_colors(colors):
    """
    Combine two colors into one with concentrations of 50% of the original colors.

    :param colors: Colors in hexidecimal form to combine.
    :type colors: list(2)

    :returns: Returns the new color in hexidecimal form.
    :rtype: str
    """
    red = int(sum([int(k[:2], 16) * 0.5 for k in colors]))
    green = int(sum([int(k[2:4], 16) * 0.5 for k in colors]))
    blue = int(sum([int(k[4:6], 16) * 0.5 for k in colors]))
    zpad = lambda x: x if len(x)==2 else '0' + x
    color = f"#{zpad(hex(red)[2:])}{zpad(hex(green)[2:])}{zpad(hex(blue)[2:])}"
    return color

class LegendTitle(object):
    """
    Custom handler map for legend group titles.

    :param text_props: \**kwargs: Keyword arguments passed to :py:class:`matplotlib.text`.
    :type text_props: dict
    """
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """
        Return the artist that this HandlerBase generates for the given original artist/handle.
        Full documentation can be found here:
        https://matplotlib.org/stable/api/legend_handler_api.html#matplotlib.legend_handler.HandlerBase.legend_artist
        """
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title

class InferenceRound(UserDict):
    """
    Inference Round Dict created by a StochSS Inference Simulation containing single round, extends the UserDict object.

    :param accepted_samples: A dictionary of accepted sample values created by an inference.
    :type accepted_samples: dict

    :param distances: A list of distances values created by an inference.
    :type distances: list

    :param accepted_count: The number of accepted samples for this round.
    :type accepted_count: int

    :param trial_count: The number of total trials performed in order to converge this round.
    :type trial_count: int

    :param inferred_parameters: The mean of accepted parameter samples.
    :type inferred_parameters: dict

    :param inferred_method: Label for the method used to calculate the inferred parameters.
    :type inferred_method: str
    """
    def __init__(self, accepted_samples, distances, accepted_count, trial_count, inferred_parameters, inferred_method):
        super().__init__(accepted_samples)
        self.distances = distances
        self.accepted_count = accepted_count
        self.trial_count = trial_count
        self.inferred_method = inferred_method
        self.__inferred_parameters = {inferred_method: inferred_parameters}

    def __getattribute__(self, key):
        attribute_map = {
            'accepted_samples': self.data,
            'distances': self.distances,
            'accepted_count': self.accepted_count,
            'trial_count': self.trial_count,
            'inferred_method': self.inferred_method,
            'inferred_parameters': self.__inferred_parameters[self.inferred_method]
        }
        if key in attribute_map:
            return attribute_map[key]
        return UserList.__getattribute__(self, key)

    def __getitem__(self, key):
        if isinstance(key, int):
            param = list(self.data.keys())[key]
            msg = "InferenceRound is of type dictionary. "
            msg += f"Use inference_round['[{param}]'] instead of inference_round[{key}]['{param}']. "
            msg += f"Retrieving inference_round['[{param}]']"
            log.warning(msg)
            return self.data[param]
        if key in self.data:
            return self.data[key]
        attribute_map = {
            'accepted_samples': self.data,
            'distances': self.distances,
            'accepted_count': self.accepted_count,
            'trial_count': self.trial_count,
            'inferred_method': self.inferred_method,
            'inferred_parameters': self.__inferred_parameters[self.inferred_method]
        }
        if key in attribute_map:
            return attribute_map[key]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __plot(self, parameters, bounds, include_pdf=True, include_orig_values=True,
               include_inferred_values=False, title=None, xaxis_label=None, yaxis_label=None):
        nbins = 50
        names = list(self.data.keys())
        dims = len(names)
        fig, axes = plt.subplots(nrows=dims, ncols=dims, figsize=[14, 14])
        if xaxis_label is not None:
            _ = fig.text(0.5, 0.09, xaxis_label, size=18, ha='center', va='center')
        if yaxis_label is not None:
            _ = fig.text(0.08, 0.5, yaxis_label, size=18, ha='center', va='center', rotation='vertical')

        sing_param = []
        doub_param = [(["Parameter Intersections"], [""])]
        pdf_axes = [None] * dims
        for i, (param1, accepted_values1) in enumerate(self.data.items()):
            row = i
            for j, (param2, accepted_values2) in enumerate(self.data.items()):
                col = j
                if i == 0:
                    axes[row, col].set_title(names[j], size=16)
                if j == dims - 1:
                    axes[row, col].set_ylabel(names[i], size=16, rotation=270)
                    axes[row, col].yaxis.set_label_position("right")
                    axes[row, col].yaxis.set_label_coords(1.4, 0.5)

                if i > j:
                    axes[row, col].axis('off')
                elif i == j:
                    color = common_rgb_values[(i)%len(common_rgb_values)]
                    axes[row, col].hist(
                        accepted_values1, label="Histogram", color=color, alpha=0.75,
                        bins=nbins, range=(bounds[0][i], bounds[1][i])
                    )
                    axes[row, col].set_xlim(bounds[0][i], bounds[1][i])
                    sing_param.append(([param1], [""]))
                    sing_param.append(axes[row, col].get_legend_handles_labels())
                    if include_pdf:
                        mean, std = stats.norm.fit(accepted_values1)
                        points = numpy.linspace(min(accepted_values1), max(accepted_values1), 500)
                        pdf = stats.norm.pdf(points, loc=mean, scale=std)

                        pdf_axes[row] = axes[row, col].twinx()
                        pdf_axes[row].plot(points, pdf, label="PDF", color=color)
                        y_ticks = pdf_axes[row].get_yticks()
                        pdf_axes[row].set_ylim(0, max(y_ticks))
                        sing_param.append(pdf_axes[row].get_legend_handles_labels())
                    if include_orig_values:
                        axes[row, col].axvline(parameters[param1], alpha=0.75, color='black')
                    if include_inferred_values:
                        axes[row, col].axvline(
                            self.inferred_parameters[param1], alpha=0.75, color='black', ls='dashed'
                        )
                else:
                    color = combine_colors([
                        common_rgb_values[(i)%len(common_rgb_values)][1:],
                        common_rgb_values[(j)%len(common_rgb_values)][1:]
                    ])
                    axes[row, col].scatter(accepted_values2, accepted_values1, c=color, label=f"{param2} X {param1}")
                    axes[row, col].set_xlim(bounds[0][j], bounds[1][j])
                    axes[row, col].set_ylim(bounds[0][i], bounds[1][i])
                    doub_param.append(axes[row, col].get_legend_handles_labels())

        labels = numpy.array(sing_param)[:, 1, 0].tolist()
        handles = numpy.array(sing_param)[:, 0, 0].tolist()
        labels.extend(numpy.array(doub_param)[:, 1, 0].tolist())
        handles.extend(numpy.array(doub_param)[:, 0, 0].tolist())
        fig.legend(
            handles, labels, loc=(0.05, 0.06), fontsize=12.5, frameon=False, markerscale=1.75,
            handler_map={str: LegendTitle({'fontsize': 14})}
        )
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.25)
        if title is not None:
            _ = fig.text(0.5, 0.92, title, size=20, ha='center', va='center')

        return fig

    def __plot_intersection(self, parameters, bounds, colors, include_pdf=True, include_orig_values=True,
                     include_inferred_values=False, title=None, xaxis_label=None, yaxis_label=None):
        nbins = 50
        names = list(parameters.keys())

        if xaxis_label is None:
            xaxis_label = names[0]
        if yaxis_label is None:
            yaxis_label = names[1]

        fig, axes = plt.subplots(
            nrows=3, ncols=3, figsize=[14, 14], sharex='col', sharey='row',
            gridspec_kw={'width_ratios': [6, 1, 3], 'height_ratios': [3, 1, 6]}
        )
        _ = fig.text(0.5, 0.09, xaxis_label, size=18, ha='center', va='center')
        _ = fig.text(0.08, 0.5, yaxis_label, size=18, ha='center', va='center', rotation='vertical')
        for row in range(2):
            for col in range(1, 3):
                axes[row, col].axis('off')

        rug_symbol = ['|', '_']
        histo_row = histo_col = [0, 2]
        rug_row, rug_col = [1, 2], [0, 1]
        x_key, y_key = ['x', 'y'], ['y', 'x']
        orientation = ['vertical', 'horizontal']
        line_func = [axes[histo_row[0], histo_col[0]].axvline, axes[histo_row[1], histo_col[1]].axhline]

        rugs = [(["Rug"], [""])]
        legend = []
        pdf_axes = [None] * 2
        for i, (param, orig_val) in enumerate(parameters.items()):
            if i >= 2:
                break

            # Create histogram traces
            axes[histo_row[i], histo_col[i]].hist(
                self[param], label="Histogram", color=colors[i], alpha=0.75, orientation=orientation[i],
                bins=nbins, range=(bounds[0][i], bounds[1][i])
            )
            legend.append(([param], [""]))
            legend.append(axes[histo_row[i], histo_col[i]].get_legend_handles_labels())
            if include_pdf:
                mean, std = stats.norm.fit(self[param])
                points = numpy.linspace(min(self[param]), max(self[param]), 500)
                pdf = stats.norm.pdf(points, loc=mean, scale=std)

                if i == 0:
                    pdf_axes[i] = axes[histo_row[i], histo_col[i]].twinx()
                    pdf_axes[i].plot(points, pdf, label="PDF", color=colors[i])
                    y_ticks = pdf_axes[i].get_yticks()
                    pdf_axes[i].set_ylim(0, max(y_ticks))
                else:
                    pdf_axes[i] = axes[histo_row[i], histo_col[i]].twiny()
                    pdf_axes[i].plot(pdf, points, label="PDF", color=colors[i])
                    x_ticks = pdf_axes[i].get_xticks()
                    pdf_axes[i].set_xlim(0, max(x_ticks))
                legend.append(pdf_axes[i].get_legend_handles_labels())
            if include_orig_values:
                line_func[i](orig_val, alpha=0.75, color='black')
            if include_inferred_values:
                line_func[i](self.inferred_parameters[param], alpha=0.75, color='black', ls='dashed')
            # Create rug traces
            rug_args = {
                x_key[i]: self[param], y_key[i]: [param] * self.accepted_count, 's': 50,
                'c': colors[i], 'label': param, 'marker': rug_symbol[i]
            }
            axes[rug_row[i], rug_col[i]].scatter(**rug_args)
            if i == 0:
                axes[rug_row[i], rug_col[i]].set_yticks([])
                axes[rug_row[i], rug_col[i]].yaxis.set_tick_params(labelleft=False)
            else:
                axes[rug_row[i], rug_col[i]].set_xticks([])
                axes[rug_row[i], rug_col[i]].xaxis.set_tick_params(labelbottom=False)
            rugs.append(axes[rug_row[i], rug_col[i]].get_legend_handles_labels())

        legend.extend(rugs)

        axes[2, 0].scatter(self[names[0]], self[names[1]], c=colors[2], label=f"{names[0]} X {names[1]}")
        legend.append((["Intersection"], [""]))
        legend.append(axes[2, 0].get_legend_handles_labels())

        labels = numpy.array(legend)[:, 1, 0].tolist()
        handles = numpy.array(legend)[:, 0, 0].tolist()
        fig.legend(
            handles, labels, loc=(0.78, 0.68), fontsize=12.5, frameon=False, markerscale=1.75,
            handler_map={str: LegendTitle({'fontsize': 14})}
        )
        axes[2, 0].set_xlim(bounds[0][0], bounds[1][0])
        axes[2, 0].set_ylim(bounds[0][1], bounds[1][1])
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        if title is not None:
            _ = fig.text(0.5, 0.92, title, size=20, ha='center', va='center')

        return fig

    def __plotplotly(self, parameters, bounds, include_pdf=True, include_orig_values=True,
                     include_inferred_values=False, title=None, xaxis_label=None, yaxis_label=None):
        nbins = 50
        names = list(self.data.keys())
        sizes = (numpy.array(bounds[1]) - numpy.array(bounds[0])) / nbins

        plotly.offline.init_notebook_mode()
        dims = len(names)
        specs = [[{"secondary_y": True} if x == y else {} for y in range(dims)] for x in range(dims)]
        fig = subplots.make_subplots(
            rows=dims, cols=dims, column_titles=names, row_titles=names, specs=specs,
            x_title=xaxis_label, y_title=yaxis_label, vertical_spacing=0.05, horizontal_spacing=0.05
        )

        for i, (param1, accepted_values1) in enumerate(self.data.items()):
            row = i + 1
            for j, (param2, accepted_values2) in enumerate(self.data.items()):
                col = j + 1
                if i > j:
                    continue
                if i == j:
                    color = common_rgb_values[(i)%len(common_rgb_values)]
                    trace = plotly.graph_objs.Histogram(
                        x=accepted_values1, name="Histogram", legendgroup=param1, showlegend=True, marker_color=color,
                        opacity=0.75, xbins={"start": bounds[0][i], "end": bounds[1][i], "size": sizes[i]},
                        legendgrouptitle={'text': param1}, legendrank=1
                    )
                    fig.add_trace(trace, row, col, secondary_y=False)
                    fig.update_xaxes(row=row, col=col, range=[bounds[0][i], bounds[1][i]])
                    if include_pdf:
                        mean, std = stats.norm.fit(accepted_values1)
                        points = numpy.linspace(min(accepted_values1), max(accepted_values1), 500)
                        pdf = stats.norm.pdf(points, loc=mean, scale=std)
                        trace2 = plotly.graph_objs.Scatter(
                            x=points, y=pdf, mode='lines', line=dict(color=color),
                            name="PDF", legendgroup=param1, showlegend=True, legendrank=1
                        )
                        fig.add_trace(trace2, row, col, secondary_y=True)
                    if include_inferred_values:
                        fig.add_vline(
                            self.inferred_parameters[param1], row=row, col=col, exclude_empty_subplots=True,
                            layer='above', opacity=0.75, line={"color": "black", "dash": "dash"}
                        )
                    if include_orig_values:
                        fig.add_vline(
                            parameters[param1], row=row, col=col, layer='above', opacity=0.75, line={"color": "black"}
                        )
                else:
                    color = combine_colors([
                        common_rgb_values[(i)%len(common_rgb_values)][1:],
                        common_rgb_values[(j)%len(common_rgb_values)][1:]
                    ])
                    scatter_kwa = {
                        'x': accepted_values2, 'y': accepted_values1, 'mode': 'markers', 'marker_color': color,
                        'name': f"{param2} X {param1}", 'legendgroup': "intersections", 'showlegend': True
                    }
                    if i == 0 and j == 1:
                        scatter_kwa['legendgrouptitle'] = {'text': "Parameter Intersections"}
                    trace = plotly.graph_objs.Scatter(**scatter_kwa)
                    fig.append_trace(trace, row, col)
                    fig.update_xaxes(row=row, col=col, range=[bounds[0][j], bounds[1][j]])
                    fig.update_yaxes(row=row, col=col, range=[bounds[0][i], bounds[1][i]])

        fig.update_layout(height=1000, legend=dict(
            groupclick="toggleitem", x=0, y=0, tracegroupgap=0, itemsizing="constant"
        ))
        if title is not None:
            title = {'text': title, 'x': 0.5, 'xanchor': 'center'}
            fig.update_layout(title=title)
        if include_pdf:
            def update_annotations(annotation):
                if annotation.x >= 0.94:
                    annotation.update(x=0.96)
            fig.for_each_annotation(update_annotations)

        return fig

    def __plotplotly_intersection(self, parameters, bounds, colors, include_pdf=True, include_orig_values=True,
                     include_inferred_values=False, title=None, xaxis_label=None, yaxis_label=None):
        nbins = 50
        names = list(parameters.keys())
        sizes = (numpy.array(bounds[1]) - numpy.array(bounds[0])) / nbins

        if xaxis_label is None:
            xaxis_label = names[0]
        if yaxis_label is None:
            yaxis_label = names[1]

        specs = [[{"secondary_y": True}, {}, {}], [{}, {}, {}], [{}, {}, {}]]
        fig = subplots.make_subplots(
            rows=3, cols=3, x_title=xaxis_label, y_title=yaxis_label, horizontal_spacing=0.01,
            vertical_spacing=0.01, column_widths=[0.6, 0.1, 0.3], row_heights=[0.3, 0.1, 0.6],
            shared_xaxes=True, shared_yaxes=True, specs=specs
        )

        bins = ['xbins', 'ybins']
        histo_row = histo_col = [1, 3]
        rug_row, rug_col = [2, 3], [1, 2]
        x_key, y_key = ['x', 'y'], ['y', 'x']
        line_func = [fig.add_vline, fig.add_hline]
        rug_symbol = ['line-ns-open', 'line-ew-open']
        xaxis_func = [fig.update_xaxes, fig.update_yaxes]
        yaxis_func = [fig.update_yaxes, fig.update_xaxes]
        secondary_y = [True, False]

        for i, (param, orig_val) in enumerate(parameters.items()):
            if i >= 2:
                break

            # Create histogram traces
            histo_trace = plotly.graph_objs.Histogram(
                marker_color=colors[i], opacity=0.75,
                name="Histogram", legendgroup=param, showlegend=True, legendgrouptitle={'text': param}, legendrank=1
            )
            histo_trace[x_key[i]] = self[param]
            histo_trace[bins[i]] = {"start": bounds[0][i], "end": bounds[1][i], "size": sizes[i]}
            fig.add_trace(histo_trace, histo_row[i], histo_col[i], secondary_y=False)
            xaxis_func[i](row=histo_row[i], col=histo_col[i], range=[bounds[0][i], bounds[1][i]])
            if include_pdf:
                mean, std = stats.norm.fit(self[param])
                points = numpy.linspace(min(self[param]), max(self[param]), 500)
                pdf = stats.norm.pdf(points, loc=mean, scale=std)
                histo_trace2 = plotly.graph_objs.Scatter(
                    mode='lines', line=dict(color=colors[i]),
                    name="PDF", legendgroup=param, showlegend=True, legendrank=1
                )
                fig.add_trace(histo_trace2, histo_row[i], histo_col[i], secondary_y=secondary_y[i])
                if i > 0:
                    fig.data[-1].update(xaxis='x10')
                fig.data[-1].update(**{x_key[i]: points, y_key[i]: pdf})
                fig.update_layout(xaxis10={
                    'overlaying': 'x9', 'side': 'top', 'layer': 'above traces', 'anchor': 'free', 'position': 0.59
                })
            if include_inferred_values:
                line_func[i](
                    self.inferred_parameters[param], row=histo_row[i], col=histo_col[i], exclude_empty_subplots=True,
                    layer='above', opacity=0.75, line={"color": "black", "dash": "dash"}
                )
            if include_orig_values:
                line_func[i](
                    orig_val, row=histo_row[i], col=histo_col[i], layer='above', opacity=0.75, line={"color": "black"}
                )
            # Create rug traces
            rug_trace = plotly.graph_objs.Scatter(
                mode='markers', marker={'color': colors[i], 'symbol': rug_symbol[i]},
                name=param, legendgroup="rug", showlegend=True, legendrank=2,
                legendgrouptitle={'text': "Rug"}
            )
            rug_trace[x_key[i]] = self[param]
            rug_trace[y_key[i]] = [param] * self.accepted_count
            fig.append_trace(rug_trace, rug_row[i], rug_col[i])
            xaxis_func[i](row=rug_row[i], col=rug_col[i], range=[bounds[0][i], bounds[1][i]])
            yaxis_func[i](row=rug_row[i], col=rug_col[i], showticklabels=False)

        trace = plotly.graph_objs.Scatter(
            x=self[names[0]], y=self[names[1]], mode='markers', marker_color=colors[2],
            name=f"{names[0]} X {names[1]}", legendgroup="intersection", showlegend=True,
            legendgrouptitle={'text': "Intersection"}
        )
        fig.append_trace(trace, 3, 1)
        fig.update_xaxes(row=3, col=1, range=[bounds[0][0], bounds[1][0]])
        fig.update_yaxes(row=3, col=1, range=[bounds[0][1], bounds[1][1]])

        fig.update_layout(height=1000, legend=dict(
            groupclick="toggleitem", x=0.75, y=1, tracegroupgap=0, itemsizing="constant"
        ))
        if title is not None:
            title = {'text': title, 'x': 0.5, 'xanchor': 'center'}
            fig.update_layout(title=title)

        return fig

    @classmethod
    def build_from_inference_round(cls, data, names):
        """
        Build an InferenceRound object using the provided inference round results.

        :param data: The results from an inference round.
        :type data: dict

        :param names: List of the parameter names.
        :type names: list

        :returns: An InferenceRound object using the provided inference results.
        :rtype: InferenceResult
        """
        _accepted_samples = numpy.vstack(data['accepted_samples']).swapaxes(0, 1)

        accepted_samples = {}
        inferred_parameters = {}
        for i, name in enumerate(names):
            accepted_samples[name] = _accepted_samples[i]
            inferred_parameters[name] = data['inferred_parameters'][i]

        return InferenceRound(
            accepted_samples, data['distances'], data['accepted_count'], data['trial_count'],
            inferred_parameters, "mean"
        )

    def calculate_inferred_parameters(self, key=None, method=None):
        """
        Calculate the inferred parameters using the given method and cached using the given key.

        :param key: Key used to cache the inferred parameters.
                    If method is None, key is a reference to a supported function.
        :type key: str

        :param method: A callable function or method used to calculate the inferred parameters.
                       Needs to accept a single numpy.ndarray argument.
        :type method: Callable

        :returns: The calculated inferred parameters.
        :rtype: dict

        :raises ValueError: method and key are None or method is not callable.
        """
        if key is None and method is None:
            raise ValueError("key or method must be set.")

        if key is None:
            i = 0
            key = "custom"
            while key in self.__inferred_parameters:
                i += 1
                key = f"custom{i}"

        if key in self.__inferred_parameters:
            self.inferred_method = key
            return self.__inferred_parameters[key]

        if method is None:
            key_methods = {"mean": numpy.mean, "median": numpy.median}
            if not (isinstance(key, str) and key in key_methods):
                raise ValueError(f"{key} is not a supported function key.  Supported keys: {tuple(key_methods.keys())}")
            method = key_methods[key]

        if not callable(method) or type(method).__name__ not in ("function", "method"):
            raise ValueError("method must be a callable function or method.")

        inferred_parameters = {}
        for param, accepted_values in self.items():
            inferred_parameters[param] = method(accepted_values)
        self.inferred_method = key
        self.__inferred_parameters[key] = inferred_parameters
        return inferred_parameters

    def plot(self, parameters, bounds, use_matplotlib=False, save_fig=None, return_plotly_figure=False, **kwargs):
        """
        Plot the results of the inference round.

        :param parameters: Dictionary of the parameters and original values.
        :type parameters: dict

        :param bounds: List of bounds for of the parameter space.
        :type bounds: list

        :param use_matplotlib: Whether or not to plot using MatPlotLib.
        :type use_matplotlib: bool

        :param save_fig: \**kwargs: Keyword arguments passed to :py:class:`matplotlib.pyplot.savefig`
                           for saving round plots. Ignored if use_matplotlib is False.
        :type save_fig: dict

        :param return_plotly_figure: Whether or not to return the figure. Ignored if use_matplotlib is set.
        :type return_plotly_figure: bool

        :param include_pdf: Whether or not to include the probability distribution curve.
        :type include_pdf: bool

        :param include_orig_values: Whether or not to include a line marking the original parameter values.
        :type include_orig_values: bool

        :param include_inferred_values: Whether or not to include a line marking the inferred parameter values.
        :type include_inferred_values: bool

        :param xaxis_label: The label for the x-axis
        :type xaxis_label: str

        :param yaxis_label: The label for the y-axis
        :type yaxis_label: str

        :param title: The title of the graph
        :type title: str

        :returns: Plotly figure object if return_plotly_figure is set else None.
        :rtype: plotly.Figure
        """
        if use_matplotlib:
            fig = self.__plot(parameters, bounds, **kwargs)
            if save_fig is not None:
                fig.savefig(**save_fig)
            return None

        if not PLOTLY_INSTALLED:
            raise ImportError("Unable to plot results.  To continue, install plotly or set 'use_matplotlib' to 'True'")

        fig = self.__plotplotly(parameters, bounds, **kwargs)

        if return_plotly_figure:
            return fig
        plotly.offline.iplot(fig)
        return None

    def plot_intersection(self, parameters, bounds, colors=None, color_ndxs=None,
                          use_matplotlib=False, save_fig=None, return_plotly_figure=False, **kwargs):
        """
        Plot the results of the inference round.

        :param parameters: Dictionary of two parameters and their original values.
        :type parameters: dict

        :param bounds: List of bounds for the provided parameters.
        :type bounds: list

        :param colors: List of three colors.
        :type colors: list

        :param color_ndxs: List of two color indicies. Ignored if colors is set.
        :type color_ndxs: list

        :param use_matplotlib: Whether or not to plot using MatPlotLib.
        :type use_matplotlib: bool

        :param save_fig: \**kwargs: Keyword arguments passed to :py:class:`matplotlib.pyplot.savefig`
                           for saving intersection plots. Ignored if use_matplotlib is False.
        :type save_fig: dict

        :param return_plotly_figure: Whether or not to return the figure. Ignored if use_matplotlib is set.
        :type return_plotly_figure: bool

        :param include_pdf: Whether or not to include the probability distribution curve.
        :type include_pdf: bool

        :param include_orig_values: Whether or not to include a line marking the original parameter values.
        :type include_orig_values: bool

        :param include_inferred_values: Whether or not to include a line marking the inferred parameter values.
        :type include_inferred_values: bool

        :param xaxis_label: The label for the x-axis
        :type xaxis_label: str

        :param yaxis_label: The label for the y-axis
        :type yaxis_label: str

        :param title: The title of the graph
        :type title: str

        :returns: Plotly figure object if return_plotly_figure is set else None.
        :rtype: plotly.Figure
        """
        if colors is None:
            if color_ndxs is None:
                colors = [
                    common_rgb_values[(0)%len(common_rgb_values)], common_rgb_values[(1)%len(common_rgb_values)]
                ]
            else:
                colors = [
                    common_rgb_values[(color_ndxs[0])%len(common_rgb_values)],
                    common_rgb_values[(color_ndxs[1])%len(common_rgb_values)]
                ]
            colors.append(combine_colors([colors[0][1:], colors[1][1:]]))

        if use_matplotlib:
            fig = self.__plot_intersection(parameters, bounds, colors, **kwargs)
            if save_fig is not None:
                fig.savefig(**save_fig)
            return None

        if not PLOTLY_INSTALLED:
            raise ImportError("Unable to plot results.  To continue, install plotly or set 'use_matplotlib' to 'True'")

        fig = self.__plotplotly_intersection(parameters, bounds, colors, **kwargs)

        if return_plotly_figure:
            return fig
        plotly.offline.iplot(fig)
        return None

    def to_csv(self, path):
        """
        Generate the csv results for the round.

        :param path: The path to the csv file.
        :type path: str
        """
        headers = ["Sample ID", *list(self.data.keys()), "Distances"]
        accepted_samples = numpy.array(list(self.data.values())).swapaxes(0, 1)

        with open(path, 'w', newline='', encoding="utf-8") as csv_fd:
            csv_writer = csv.writer(csv_fd)
            csv_writer.writerow(headers)
            for i, accepted_sample in enumerate(accepted_samples):
                line = accepted_sample.tolist()
                line.insert(0, i + 1)
                if isinstance(self.distances[i], list):
                    line.extend(self.distances[i])
                else:
                    line.append(self.distances[i])
                csv_writer.writerow(line)

    def to_dict(self):
        """
        Return the results of the round as a dictionary.

        :returns: The results of the round.
        :rtype: dict
        """
        accepted_samples = numpy.array(list(self.data.values())).swapaxes(0, 1)
        return {
            'accepted_samples': accepted_samples,
            'distances': self.distances,
            'accepted_count': self.accepted_count,
            'trial_count': self.trial_count,
            'inferred_parameters': numpy.array(self.inferred_parameters.values())
        }

class InferenceResults(UserList):
    """
    List of InferenceRound objects created by a StochSS Inference Simulation, extends the UserList object.

    :param data: A list of inference round objects
    :type data: list

    :param parameters: Dictionary of the parameters and original values.
    :type parameters: dict

    :param bounds: List of bounds for of the parameter space.
    :type bounds: list
    """
    def __init__(self, data, parameters, bounds):
        super().__init__(data)
        self.parameters = parameters
        self.bounds = bounds

    def __getattribute__(self, key):
        if key in ('distances', 'accepted_count', 'trial_count', 'inferred_parameters'):
            if len(self.data) > 1:
                msg = f"Results is of type list. Use results[i]['{key}'] instead of results['{key}']"
                log.warning(msg)
            return getattr(InferenceResults.__getattribute__(self, key='data')[-1], key)
        return UserList.__getattribute__(self, key)

    def __getitem__(self, key):
        if key == 'data':
            return UserList.__getitem__(self, key)
        if isinstance(key, str):
            if len(self.data) > 1:
                msg = f"Results is of type list. Use results[i]['{key}'] instead of results['{key}']"
                log.warning(msg)
            return self.data[0][key]
        return UserList.__getitem__(self,key)

    def __add__(self, other):
        c_type = type(other).__name__
        if c_type != "InferenceResults":
            raise ValueError(f'{c_type} cannot be added to InferenceResults.')

        if self.parameters != other.parameters:
            raise ValueError("InferenceResults object contain difference parameters.")

        if not numpy.all(numpy.array(self.bounds) == numpy.array(other.bounds)):
            raise ValueError("InferenceResults object contain difference priors.")

        return InferenceResults(
            data=(self.data + other.data), parameters=self.parameters, bounds=self.bounds
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __plot(self, include_orig_values=True, include_inferred_values=False,
                     title=None, xaxis_label="Parameter Values", yaxis_label=None):
        if yaxis_label is None:
            yaxis_label = {"histo": "Accepted Samples", "pdf": "Probability"}
        elif isinstance(yaxis_label, str):
            yaxis_label = {"histo": yaxis_label, "pdf": yaxis_label}

        cols = 2
        rows = int(numpy.ceil(len(self.parameters)/cols))
        names = list(self.parameters.keys())
        histo_fig, histo_axes = plt.subplots(nrows=rows, ncols=cols, figsize=[14, 7 * rows])
        _ = histo_fig.text(0.5, 0.09, xaxis_label, size=18, ha='center', va='center')
        _ = histo_fig.text(0.08, 0.5, yaxis_label['histo'], size=18, ha='center', va='center', rotation='vertical')

        pdf_fig, pdf_axes = plt.subplots(nrows=rows, ncols=cols, figsize=[14, 7 * rows])
        _ = pdf_fig.text(0.5, 0.09, xaxis_label, size=18, ha='center', va='center')
        _ = pdf_fig.text(0.08, 0.5, yaxis_label['pdf'], size=18, ha='center', va='center', rotation='vertical')

        if len(self.parameters) < rows * cols:
            histo_axes[-1, -1].axis('off')
            pdf_axes[-1, -1].axis('off')

        nbins = 50
        for i, inf_round in enumerate(self.data):
            base_opacity = 0.5 if len(self.data) <= 1 else (i / (len(self.data) - 1) * 0.5)

            for j, (param, accepted_values) in enumerate(inf_round.data.items()):
                row = int(numpy.ceil((j + 1) / cols)) - 1
                col = (j % cols)

                name = f"round {i + 1}"
                color = common_rgb_values[i % len(common_rgb_values)]
                opacity = base_opacity + 0.25
                # Create histogram trace
                histo_axes[row, col].hist(
                    accepted_values, label=name, color=color, alpha=opacity,
                    bins=nbins, range=(self.bounds[0][j], self.bounds[1][j])
                )
                # Create pdf trace
                mean, std = stats.norm.fit(accepted_values)
                points = numpy.linspace(min(accepted_values), max(accepted_values), 500)
                pdf = stats.norm.pdf(points, loc=mean, scale=std)
                pdf_axes[row, col].plot(points, pdf, label=name, color=color)

                if i == len(self.data) - 1:
                    histo_axes[row, col].set_title(names[j], size=16)
                    pdf_axes[row, col].set_title(names[j], size=16)
                    if include_orig_values:
                        histo_axes[row, col].axvline(self.parameters[param], alpha=0.75, color='black')
                        pdf_axes[row, col].axvline(self.parameters[param], alpha=0.75, color='black')
                    if include_inferred_values:
                        histo_axes[row, col].axvline(
                            inf_round.inferred_parameters[param], alpha=0.75, color='black', ls='dashed'
                        )
                        pdf_axes[row, col].axvline(
                            inf_round.inferred_parameters[param], alpha=0.75, color='black', ls='dashed'
                        )

        histo_handles, histo_labels = histo_axes[0, 0].get_legend_handles_labels()
        histo_fig.legend(
            histo_handles, histo_labels, loc=(0.905, 0.83), fontsize=12.5, frameon=False, labelspacing=1.2
        )
        pdf_handles, pdf_labels = pdf_axes[0, 0].get_legend_handles_labels()
        pdf_fig.legend(
            pdf_handles, pdf_labels, loc=(0.905, 0.83), fontsize=12.5, frameon=False, labelspacing=1.2
        )
        if title is not None:
            _ = histo_fig.text(0.5, 0.92, title, size=20, ha='center', va='center')
            _ = pdf_fig.text(0.5, 0.92, title, size=20, ha='center', va='center')

        return histo_fig, pdf_fig

    def __plotplotly(self, include_orig_values=True, include_inferred_values=False,
                     title=None, xaxis_label="Parameter Values", yaxis_label=None):
        if yaxis_label is None:
            yaxis_label = {"histo": "Accepted Samples", "pdf": "Probability"}
        elif isinstance(yaxis_label, str):
            yaxis_label = {"histo": yaxis_label, "pdf": yaxis_label}

        cols = 2
        rows = int(numpy.ceil(len(self.parameters)/cols))
        names = list(self.parameters.keys())
        histo_fig = subplots.make_subplots(
            rows=rows, cols=cols, subplot_titles=names, vertical_spacing=0.075,
            x_title=xaxis_label, y_title=yaxis_label['histo']
        )
        pdf_fig = subplots.make_subplots(
            rows=rows, cols=cols, subplot_titles=names, vertical_spacing=0.075,
            x_title=xaxis_label, y_title=yaxis_label['pdf']
        )

        nbins = 50
        sizes = (numpy.array(self.bounds[1]) - numpy.array(self.bounds[0])) / nbins
        for i, inf_round in enumerate(self.data):
            base_opacity = 0.5 if len(self.data) <= 1 else (i / (len(self.data) - 1) * 0.5)

            for j, (param, accepted_values) in enumerate(inf_round.data.items()):
                row = int(numpy.ceil((j + 1) / cols))
                col = (j % cols) + 1

                name = f"round {i + 1}"
                color = common_rgb_values[i % len(common_rgb_values)]
                opacity = base_opacity + 0.25
                # Create histogram trace
                trace = plotly.graph_objs.Histogram(
                    x=accepted_values, name=name, legendgroup=name, showlegend=j==0, marker_color=color,
                    opacity=opacity, xbins={"start": self.bounds[0][j], "end": self.bounds[1][j], "size": sizes[j]}
                )
                histo_fig.append_trace(trace, row, col)
                # Create PDF trace
                mean, std = stats.norm.fit(accepted_values)
                points = numpy.linspace(min(accepted_values), max(accepted_values), 500)
                pdf = stats.norm.pdf(points, loc=mean, scale=std)
                trace2 = plotly.graph_objs.Scatter(
                    x=points, y=pdf, name=name, legendgroup=name, showlegend=j==0, mode='lines', line=dict(color=color)
                )
                pdf_fig.append_trace(trace2, row, col)
                pdf_fig.update_xaxes(row=row, col=col, range=[self.bounds[0][j], self.bounds[1][j]])

                if i == len(self.data) - 1:
                    if include_orig_values:
                        histo_fig.add_vline(
                            self.parameters[param], row=row, col=col, layer='above', opacity=0.75,
                            line={"color": "black"}
                        )
                        pdf_fig.add_vline(
                            self.parameters[param], row=row, col=col, layer='above', opacity=0.75,
                            line={"color": "black"}
                        )
                    if include_inferred_values:
                        histo_fig.add_vline(
                            inf_round.inferred_parameters[param], row=row, col=col, exclude_empty_subplots=True,
                            layer='above', opacity=0.75, line={"color": "black", "dash": "dash"}
                        )
                        pdf_fig.add_vline(
                            inf_round.inferred_parameters[param], row=row, col=col, exclude_empty_subplots=True,
                            layer='above', opacity=0.75, line={"color": "black", "dash": "dash"}
                        )

        height = 500 * rows
        histo_fig.update_layout(barmode='overlay', height=height)
        pdf_fig.update_layout(height=height)
        if title is not None:
            title = {'text': title, 'x': 0.5, 'xanchor': 'center'}
            histo_fig.update_layout(title=title)
            pdf_fig.update_layout(title=title)

        return histo_fig, pdf_fig

    @classmethod
    def build_from_inference_results(cls, data, parameters, bounds):
        """
        Build an InferenceResult object using the provided inference.

        :param data: The results from an inference result.
        :type data: list or dict

        :param parameters: Dictionary of the parameters and original values.
        :type parameters: dict

        :param bounds: List of bounds for of the parameter space.
        :type bounds: list

        :returns: An InferenceResult object using the provided inference results.
        :rtype: InferenceResult
        """
        if isinstance(data, dict):
            data = [data]

        inf_rounds = []
        names = list(parameters.keys())
        for inf_r in data:
            inf_round = InferenceRound.build_from_inference_round(inf_r, names)
            inf_rounds.append(inf_round)
        return InferenceResults(inf_rounds, parameters, bounds)

    def calculate_inferred_parameters(self, key=None, method=None, ndx=None):
        """
        Calculate the inferred parameters using the given method and cached using the given key.

        :param key: Key used to cache the inferred parameters.
                    If method is None, key is a reference to a supported function.
        :type key: str

        :param method: A callable function or method used to calculate the inferred parameters.
                       Needs to accept a single numpy.ndarray argument.
        :type method: Callable

        :param ndx: Index of the inference round to plot.
        :type ndx: int

        :returns: The calculated inferred parameters for the indicated round if ndx is set else the final round.
        :rtype: dict

        :raises ValueError: method and key are None or method is not callable.
        """
        if ndx is not None:
            return self[ndx].calculate_inferred_parameters(key=key, method=method)

        for inf_round in self:
            inferred_parameters = inf_round.calculate_inferred_parameters(key=key, method=method)
        return inferred_parameters

    def plot(self, histo_only=True, pdf_only=False, use_matplotlib=False,
             save_histo=None, save_pdf=None, return_plotly_figure=False, **kwargs):
        """
        Plot the results.

        :param : Indicates that only the histogram plot should be returned.
        :type : bool

        :param : Indicates that only the PDF plot should be returned.
        :type : bool

        :param use_matplotlib: Whether or not to plot using MatPlotLib.
        :type use_matplotlib: bool

        :param save_histo: \**kwargs: Keyword arguments passed to :py:class:`matplotlib.pyplot.savefig`
                           for saving histogram plots. Ignored if use_matplotlib is False.
        :type save_histo: dict

        :param save_pdf: \**kwargs: Keyword arguments passed to :py:class:`matplotlib.pyplot.savefig`
                         for saving pdf plots. Ignored if use_matplotlib is False.
        :type save_pdf: dict

        :param return_plotly_figure: Whether or not to return the figure. Ignored if use_matplotlib is set.
        :type return_plotly_figure: bool

        :param include_orig_values: Whether or not to include a line marking the original parameter values.
        :type include_orig_values: bool

        :param include_inferred_values: Whether or not to include a line marking the
                                        inferred parameter values of the final round.
        :type include_inferred_values: bool

        :param xaxis_label: The label for the x-axis
        :type xaxis_label: str

        :param yaxis_label: The label for the y-axis. Dictionaries should be in
                            the following format {'histo':<<label>>, 'pdf':<<label>>}.
        :type yaxis_label: dict | str

        :param title: The title of the graph
        :type title: str

        :returns: Plotly figure object if return_plotly_figure is set else None.
        :rtype: plotly.Figure
        """
        if use_matplotlib:
            histo_fig, pdf_fig = self.__plot(**kwargs)
            if histo_only:
                plt.close(fig=pdf_fig)
            if pdf_only:
                plt.close(fig=histo_fig)
            if save_histo is not None:
                histo_fig.savefig(**save_histo)
            if save_pdf is not None:
                pdf_fig.savefig(**save_pdf)
            return None

        if not PLOTLY_INSTALLED:
            raise ImportError("Unable to plot results.  To continue, install plotly or set 'use_matplotlib' to 'True'")

        histo_fig, pdf_fig = self.__plotplotly(**kwargs)

        if return_plotly_figure:
            if histo_only:
                return histo_fig
            if pdf_only:
                return pdf_fig
            return histo_fig, pdf_fig
        if histo_only:
            plotly.offline.iplot(histo_fig)
        elif pdf_only:
            plotly.offline.iplot(pdf_fig)
        else:
            plotly.offline.iplot(histo_fig)
            plotly.offline.iplot(pdf_fig)
        return None

    def plot_round(self, ndx=None, **kwargs):
        """
        Plot the results of a single inference round.

        :param ndx: Index of the inference round to plot.
        :type ndx: int

        :param \**kwargs: Additional keyword arguments passed to :py:class:`InferenceRound.plot`.

        :returns: Plotly fig object if return_plotly_figure is set else None.
        :rtype: plotly.Figure
        """
        if ndx is None:
            ndx = -1

        inf_round = self.data[ndx]
        return inf_round.plot(self.parameters, self.bounds, **kwargs)

    def plot_round_intersection(self, ndx=None, names=None, **kwargs):
        """
        Plot the results of a inference round intersection.

        :param ndx: Index of the inference round to plot.
        :type ndx: int

        :param names: List of two parameters.
        :type names: list

        :param \**kwargs: Additional keyword arguments passed to :py:class:`InferenceRound.plot`.

        :returns: Plotly fig object if return_plotly_figure is set else None.
        :rtype: plotly.Figure
        """
        if ndx is None:
            ndx = -1

        param_names = list(self.parameters.keys())
        if names is None:
            names = param_names[:2]

        if not ("colors" in kwargs or "color_ndxs" in kwargs):
            kwargs['color_ndxs'] = [param_names.index(names[1]), param_names.index(names[0])]

        inf_round = self.data[ndx]
        bounds = [
            [self.bounds[0][param_names.index(names[1])], self.bounds[0][param_names.index(names[0])]],
            [self.bounds[1][param_names.index(names[1])], self.bounds[1][param_names.index(names[0])]]
        ]
        parameters = {names[1]: self.parameters[names[1]], names[0]: self.parameters[names[0]]}
        return inf_round.plot_intersection(parameters, bounds, **kwargs)

    def to_array(self):
        """
        Convert the results object into an array.
        """
        return [inf_round.to_dict() for inf_round in self.data]

    def to_csv(self, path='.', nametag="results_csv"):
        """
        Convert the results to CSV.

        :param path: The location for the new directory and included files. Defaults to current working directory.
        :type path: str

        :param nametag: Unique identifier for to CSV directory.
        :type nametag: str
        """
        directory = os.path.join(path, str(nametag))
        if not os.path.exists(directory):
            os.mkdir(directory)

        headers = ["Round", "Accepted Count", "Trial Count", *list(self.parameters.keys())]

        inf_path = os.path.join(directory, "inference-overview.csv")
        with open(inf_path, 'w', newline='', encoding="utf-8") as csv_fd:
            csv_writer = csv.writer(csv_fd)
            csv_writer.writerow(headers)
            for i, inf_round in enumerate(self.data):
                inf_round.to_csv(path=os.path.join(directory, f"round{i + 1}-details.csv"))

                line = [i + 1, inf_round.accepted_count, inf_round.trial_count]
                line.extend(list(inf_round.inferred_parameters.values()))
                csv_writer.writerow(line)
