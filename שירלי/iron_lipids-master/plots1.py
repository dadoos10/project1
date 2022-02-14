from toolBox import *
class Plots1:
    class MarkerHandler(HandlerBase):
        """subclass for markers legend"""

        def create_artists(self, legend, tup, xdescent, ydescent, width, height, fontsize,
                           trans):
            return [plt.Line2D([width / 4], [height / 4.], ls="", marker=tup[1], color='b',
                               transform=trans)]

    def __init__(self, x_axis, y_axis, x_label, y_label, title, data):
        """ init the instance of plot"""
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.data = data

    def colors_and_markers(self):
        """ the function responsible for create list of markers and colors for the elements in
        the plot, according to iron and lipid types. """
        # do colors by iron type and markers by lipid type
        colors = [iron_type_colors[iron_type] for iron_type in self.data[IRON_TYPE]]
        markers_list = [lipid_type_markers[lipid_type] for lipid_type in self.data[LIPID_TYPE]]

        markers_types = []
        markers_labels = []

        for lipid_type in np.unique(self.data[LIPID_TYPE]):
            if lipid_type not in markers_labels:
                markers_labels.append(lipid_type)
                markers_types.append(lipid_type_markers[lipid_type])
        return colors, markers_list, markers_types, markers_labels

    def plot(self):
        """ the main function, plots the data into scatter plot"""
        colors, markers_list, markers_types, markers_labels = self.colors_and_markers()
        for i in range(len(self.y_axis)):
            plt.scatter(self.x_axis[i], self.y_axis[i], c=colors[i], marker=markers_list[i])

        min_scale = self.x_axis if (self.x_axis.min() < self.y_axis.min()) else self.y_axis
        plt.plot([min_scale.min(), min_scale.max()], [min_scale.min(), min_scale.max()], 'k--',
                 lw=2)

        # create legend of colors
        labels_legend_color = list()
        for iron_type in iron_type_colors:
            labels_legend_color.append(mpatches.Patch(color=iron_type_colors[iron_type],
                                                      label=str(iron_type)))
        legend1 = plt.legend(list(zip(colors, markers_types)), markers_labels,
                             handler_map={tuple: Plots1.MarkerHandler()}, loc=4)
        plt.legend(handles=labels_legend_color, loc='best')
        plt.gca().add_artist(legend1)
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        plt.show()
