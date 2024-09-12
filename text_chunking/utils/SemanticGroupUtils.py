import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from typing import List
import datamapplot
import seaborn as sns
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class SemanticGroupUtils:
    """
    A set of tools to manipulate and visualize semantic text splits
    """

    @staticmethod
    def plot_chunk_differences_and_breakpoints(
        split_texts: List, break_points: np.array, chunk_cosine_distances: np.array
    ) -> None:
        """

        :param split_texts: List of text splits
        :param break_points: Array of break point indices
        :param chunk_cosine_distances: Array of cosine distances between text splts
        :return: plot of embeddings of breakpoints and embedding of the text splits
        """
        cumulative_len = np.cumsum([len(x) for x in split_texts])
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(111)

        cosine_dist_min = 0
        cosine_dist_max = 1.1 * max(chunk_cosine_distances)

        ax.plot(cumulative_len[:-1], chunk_cosine_distances)
        ax.plot(cumulative_len[:-1], chunk_cosine_distances, "rx", markersize=5)
        ax.vlines(
            cumulative_len[break_points],
            ymin=cosine_dist_min,
            ymax=cosine_dist_max,
            colors="r",
            linestyles="--",
            linewidth=0.5,
        )
        ax.set_xlabel("Cumulative characters")
        ax.set_ylabel("Cosine distance between splits")

    @staticmethod
    def plot_2d_semantic_embeddings(semantic_embeddings: List, semantic_text_groups: List[str]) -> None:
        """
        Create plot of semantic embeddings in 2D

        :param semantic_embeddings: Embeddings of text chunks
        :param semantic_text_groups: List of text chunks
        :return:
        """

        # reduce dimensions
        dimension_reducer = UMAP(
            n_neighbors=2, n_components=2, min_dist=0.0, metric="cosine"
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_embeddings)

        # create dataframe of reduced embeddings
        splits_df = pd.DataFrame(
            {
                "reduced_embeddings_x": reduced_embeddings[:, 0],
                "reduced_embeddings_y": reduced_embeddings[:, 1],
                "idx": np.arange(len(reduced_embeddings[:, 0])),
            }
        )

        # scale the chunk end lengths
        splits_df["scaled_chunk_end"] = (
            MinMaxScaler()
            .fit_transform(
                np.cumsum([len(x) for x in semantic_text_groups]).reshape(-1, 1)
            )
            .ravel()
        )
        ax = splits_df.plot.scatter(
            x="reduced_embeddings_x", y="reduced_embeddings_y", c="idx", cmap="viridis"
        )
        # Draw arrows between points
        #X = reduced_embeddings[:, 0]
        #Y = reduced_embeddings[:, 1]

        # for i in range(len(X) - 1):
        #     ax.arrow(X[i], Y[i], X[i + 1] - X[i], Y[i + 1] - Y[i], head_width=0.05, head_length=0.05, fc='black', ec='black', alpha=0.5)

        ax.plot(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            "r-",
            linewidth=0.5,
            alpha=0.5,
        )

    @staticmethod
    def plot_2d_semantic_embeddings_with_clusters(
        semantic_embeddings: List, semantic_text_groups: List, linkage: np.array, n_clusters: int = 30
    ) -> pd.DataFrame:
        """
        Create plot of 2D reduced embeddings with cluster labels
        :param semantic_embeddings:
        :param semantic_text_groups:
        :param linkage:
        :param n_clusters:
        :return:
        """
        cluster_labels = hierarchy.cut_tree(linkage, n_clusters=n_clusters).ravel()

        dimension_reducer = UMAP(
            n_neighbors=2, n_components=2, min_dist=0.0, metric="cosine"
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_embeddings)
        splits_df = pd.DataFrame(
            {
                "reduced_embeddings_x": reduced_embeddings[:, 0],
                "reduced_embeddings_y": reduced_embeddings[:, 1],
                "cluster_label": cluster_labels,
            }
        )
        splits_df["scaled_chunk_end"] = (
            MinMaxScaler()
            .fit_transform(
                np.cumsum([len(x) for x in semantic_text_groups]).reshape(-1, 1)
            )
            .ravel()
        )

        ax = splits_df.plot.scatter(
            x="reduced_embeddings_x",
            y="reduced_embeddings_y",
            c="cluster_label",
            cmap="rainbow",
        )
        ax.plot(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            "r-",
            linewidth=0.5,
            alpha=0.5,
        )
        return splits_df

    @staticmethod
    def create_hierarchical_clustering(
        semantic_group_embeddings, n_components_reduced=10, plot=True
    ):
        """

        :param semantic_group_embeddings:
        :param n_components_reduced:
        :param plot:
        :return:
        """
        dimension_reducer_clustering = UMAP(
            n_neighbors=2,
            n_components=n_components_reduced,
            min_dist=0.0,
            metric="cosine",
        )
        reduced_embeddings_clustering = dimension_reducer_clustering.fit_transform(
            semantic_group_embeddings
        )

        # create the heirarchical linkage
        row_linkage = hierarchy.linkage(
            pdist(reduced_embeddings_clustering),
            method="average",
            optimal_ordering=True,
        )

        if plot:
            g = sns.clustermap(
                pd.DataFrame(reduced_embeddings_clustering),
                row_linkage=row_linkage,
                row_cluster=True,
                col_cluster=False,
                method="average",
                annot=True,
                linewidth=0.5,
                annot_kws={"size": 8, "color": "white"},
                cbar_pos=None,
            )
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_yticklabels(), rotation=0, size=8
            )
        return row_linkage

    @staticmethod
    def plot_chunks_and_summaries(
        semantic_group_embeddings: List, semantic_group_descriptions: List
    ) -> None:
        """

        :param semantic_group_embeddings:
        :param semantic_group_descriptions:
        :return:
        """
        dimension_reducer = UMAP(
            n_neighbors=2, n_components=2, min_dist=0.0, metric="cosine"
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_group_embeddings)

        # plot the reduced embeddings and their descriptions
        fig, ax = datamapplot.create_plot(
            reduced_embeddings,
            labels=semantic_group_descriptions,
            force_matplotlib=True,
            label_wrap_width=10,
            font_family="Urbanist",
            color_label_text=False,
            add_glow=False,
            figsize=(12, 8),
            label_font_size=8,
        )
