import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

class metrics:
    def __init__(self, first_graph, second_graph):
        self.first_graph = first_graph
        self.second_graph = second_graph

        self.degrees, self.first_graph_deg_count, self.second_graph_deg_count = \
            self.get_format_distributions(first_graph, second_graph)

        self.kl_d_dist = self.get_kl_divergence(self.first_graph_deg_count, self.second_graph_deg_count)
        self.KLd = self.kl_d_dist.sum()

        self.first_average_clustering = self.get_av_clustering(first_graph)
        self.second_average_clustering = self.get_av_clustering(second_graph)
        self.av_clustering_diff = abs(self.first_average_clustering - self.second_average_clustering)

        self.first_density = self.get_density(first_graph)
        self.second_density = self.get_density(second_graph)
        self.density_diff = self.first_density - self.second_density

    def full_output(self):
        print("KLd = ", self.KLd)
        print("av_clustering_diff = ", self.av_clustering_diff)
        print("density_diff = ", self.density_diff)
        self.visual()

    def get_density(self, graph):
        g = len(np.unique(graph[:, 0]))
        l = len(graph)
        return (2 * l / ((g ** 2) - g))

    def get_av_clustering(self, graph):
        G = nx.Graph()
        for i in range(len(np.unique(graph[:, 0]))):
            G.add_node(i)
        G.add_edges_from(np.array(graph))
        #         clusterings = nx.clustering(G)
        #         frame = pd.DataFrame({'node':list(range(len(clusterings))),
        #                 'clustering_coef':list(clusterings.values())})

        return nx.average_clustering(G)

    def get_kl_divergence(self, first_dist, second_dist):
        p1 = first_dist / first_dist.sum()
        p1 += 0.000001
        p2 = second_dist / second_dist.sum()
        p2 += 0.000001
        res = p1 * np.log(p1 / p2)
        return (res)

    def get_format_distributions(self, first_graph, second_graph):
        first_graph_deg, first_graph_deg_count = self.get_distribution(first_graph)
        second_graph_deg, second_graph_deg_count = self.get_distribution(second_graph)
        max_len = max((max(first_graph_deg), (max(second_graph_deg))))
        format_first_graph_deg_count = np.zeros(max_len)
        format_first_graph_deg_count[np.array(first_graph_deg) - 1] = first_graph_deg_count
        format_second_graph_deg_count = np.zeros(max_len)
        format_second_graph_deg_count[np.array(second_graph_deg) - 1] = second_graph_deg_count
        return ((np.arange(max_len) + 1).astype(int), format_first_graph_deg_count.astype(int),
                format_second_graph_deg_count.astype(int))

    def get_distribution(self, graph):
        graph = pd.DataFrame(graph.tolist()).rename(columns={0: 'source', 1: 'target'})
        graph['degree'] = 1
        graph_degrees = graph.groupby('source').agg({'degree': 'count'})
        graph_degrees = graph_degrees.groupby('degree').agg({'degree': 'count'})
        return (list(graph_degrees.index), list(graph_degrees.degree))

    def visual(self, full_size=False):
        if full_size:
            last_index = len(self.degrees)
        else:
            last_index = 30
        f, ax = plt.subplots(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        sns.set_color_codes("dark")
        sns.barplot(x=self.degrees[:last_index], y=self.first_graph_deg_count[:last_index],
                    label="Синтетический", color="r", alpha=0.5)
        sns.lineplot(x=self.degrees[:last_index] - 1, y=self.first_graph_deg_count[:last_index], color='r')
        sns.set_color_codes("bright")
        sns.barplot(x=self.degrees[:last_index], y=self.second_graph_deg_count[:last_index],
                    label="Оригинальный", color="b", alpha=0.5)
        sns.lineplot(x=self.degrees[:last_index] - 1, y=self.second_graph_deg_count[:last_index], color="b")
        plt.xlabel("degree")
        plt.ylabel("count")
        plt.subplot(2, 1, 2)
        sns.lineplot(x=self.degrees - 1, y=self.kl_d_dist, color="black",
                     label='KL')
        plt.xlabel("degree")
        plt.ylabel("KL_value")
        plt.legend(ncol=2, loc="upper right", frameon=True)
        plt.xticks(rotation=-45)