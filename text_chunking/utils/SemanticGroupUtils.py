import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from typing import List, Dict
import datamapplot
import seaborn as sns
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class SemanticGroupUtils:
    """
    A set of tools to manipulate and visualize semantic text splits.
    """

    @staticmethod
    def plot_chunk_differences_and_breakpoints(
        split_texts: List[str],
        break_points: np.ndarray,
        chunk_cosine_distances: np.ndarray,
    ) -> None:
        """
        Plots the differences and breakpoints of text chunks based on cosine distances.

        Args:
            split_texts (List[str]): A list of text splits.
            break_points (np.ndarray): An array of break point indices.
            chunk_cosine_distances (np.ndarray): An array of cosine distances between text splits.

        Returns:
            None
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
    def plot_2d_semantic_embeddings(
        semantic_embeddings: List[np.ndarray], semantic_text_groups: List[str], umap_neighbors: int = 5
    ) -> None:
        """
        Creates a 2D plot of semantic embeddings.

        Args:
            semantic_embeddings (List[np.ndarray]): Embeddings of text chunks.
            semantic_text_groups (List[str]): A list of text chunks.

        Returns:
            None
        """
        dimension_reducer = UMAP(
            n_neighbors=umap_neighbors, n_components=2, min_dist=0.0, metric="cosine", random_state=0
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_embeddings)

        splits_df = pd.DataFrame(
            {
                "reduced_embeddings_x": reduced_embeddings[:, 0],
                "reduced_embeddings_y": reduced_embeddings[:, 1],
                "idx": np.arange(len(reduced_embeddings[:, 0])),
            }
        )

        splits_df["chunk_end"] = np.cumsum([len(x) for x in semantic_text_groups])

        ax = splits_df.plot.scatter(
            x="reduced_embeddings_x", y="reduced_embeddings_y", c="idx", cmap="viridis"
        )

        ax.plot(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            "r-",
            linewidth=0.5,
            alpha=0.5,
        )

    @staticmethod
    def plot_2d_semantic_embeddings_with_clusters(
        semantic_embeddings: List[np.ndarray],
        semantic_text_groups: List[str],
        linkage: np.ndarray,
        n_clusters: int = 30,
        umap_neighbors: int = 5
    ) -> pd.DataFrame:
        """
        Creates a 2D plot of reduced embeddings with cluster labels.

        Args:
            semantic_embeddings (List[np.ndarray]): Embeddings of text chunks.
            semantic_text_groups (List[str]): A list of text chunks.
            linkage (np.ndarray): Linkage matrix for hierarchical clustering.
            n_clusters (int, optional): Number of clusters. Defaults to 30.

        Returns:
            pd.DataFrame: DataFrame containing reduced embeddings and cluster labels.
        """
        cluster_labels = hierarchy.cut_tree(linkage, n_clusters=n_clusters).ravel()
        dimension_reducer = UMAP(
            n_neighbors=umap_neighbors, n_components=2, min_dist=0.0, metric="cosine", random_state=0
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_embeddings)

        splits_df = pd.DataFrame(
            {
                "reduced_embeddings_x": reduced_embeddings[:, 0],
                "reduced_embeddings_y": reduced_embeddings[:, 1],
                "cluster_label": cluster_labels,
            }
        )

        splits_df["chunk_end"] = np.cumsum(
            [len(x) for x in semantic_text_groups]
        ).reshape(-1, 1)

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
        semantic_group_embeddings: List[np.ndarray],
        n_components_reduced: int = 4,
        plot: bool = True,
        umap_neighbors: int = 5
    ) -> np.ndarray:
        """
        Creates hierarchical clustering from semantic group embeddings.

        Args:
            semantic_group_embeddings (List[np.ndarray]): Embeddings of semantic groups.
            n_components_reduced (int, optional): Number of components for dimensionality reduction. Defaults to 10.
            plot (bool, optional): Whether to plot the clustering. Defaults to True.

        Returns:
            np.ndarray: Linkage matrix for hierarchical clustering.
        """
        dimension_reducer_clustering = UMAP(
            n_neighbors=umap_neighbors,
            n_components=n_components_reduced,
            min_dist=0.0,
            metric="cosine",
            random_state=0
        )
        reduced_embeddings_clustering = dimension_reducer_clustering.fit_transform(
            semantic_group_embeddings
        )

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
                annot=True,
                linewidth=0.5,
                annot_kws={"size": 8, "color": "white"},
                cbar_pos=None,
                dendrogram_ratio=0.25
            )

            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_yticklabels(), rotation=0, size=8
            )

        return row_linkage

    @staticmethod
    def plot_chunks_and_summaries(
        semantic_group_embeddings: List[np.ndarray],
        semantic_group_descriptions: List[str],
        umap_neighbors: int = 5
    ) -> None:
        """
        Plots the reduced embeddings and their descriptions.

        Args:
            semantic_group_embeddings (List[np.ndarray]): Embeddings of semantic groups.
            semantic_group_descriptions (List[str]): Descriptions of semantic groups.

        Returns:
            None
        """
        dimension_reducer = UMAP(
            n_neighbors=umap_neighbors, n_components=2, min_dist=0.0, metric="cosine", random_state=0
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_group_embeddings)

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

    @staticmethod
    def plot_corpus_and_clusters(
        splits_df: pd.DataFrame, cluster_summaries: Dict[int, Dict[str, str]] = {}
    ) -> pd.DataFrame:
        """
        Plots the progression of a corpus with cluster labels and returns a DataFrame
        indicating the index spans for each cluster segment.

        Args:
            splits_df (pd.DataFrame): A DataFrame containing the corpus data with a 'cluster_label' column.
                                      It should also include a 'chunk_end' column indicating the end of each chunk.
            cluster_summaries (Dict[int, Dict[str, str]], optional): A dictionary containing cluster summaries.
                                      The keys are cluster IDs and the values are dictionaries with a 'summary' key.

        Returns:
            pd.DataFrame: A DataFrame with 'cluster_label' and 'index_span' columns, where 'index_span' is a tuple
                          indicating the start and end indices of each cluster segment.
        """
        df = splits_df

        # Identify the start of a new segment
        df["shifted"] = df["cluster_label"].shift(1)
        df["is_new_segment"] = (df["cluster_label"] != df["shifted"]).astype(int)
        segment_groups = df["is_new_segment"].cumsum()

        # Group by cluster label and segment group
        result = df.groupby(["cluster_label", segment_groups]).apply(
            lambda g: (g.index.min() - 1, g.index.max())
        )

        result = result.reset_index()[["cluster_label", 0]].rename(
            columns={0: "index_span"}
        )
        unique_clusters = len(result["cluster_label"].unique())
        cluster_colors = sns.color_palette("flare", n_colors=unique_clusters)

        fig = plt.figure(figsize=(25, 6))
        ax = fig.add_subplot(111)
        for i, row in result.iterrows():
            cluster_id = row["cluster_label"]
            span = row["index_span"]

            # if we're at the start of the sequence, xmin = -1
            # otherwise xmin is the end of the last group
            if span[0] == -1:
                xmin = 0
            else:
                xmin = splits_df.iloc[span[0]]["chunk_end"]

            xmax = splits_df.iloc[span[1]]["chunk_end"]

            ax.vlines([xmin, xmax], ymin=-1, ymax=1, color="k")
            ax.annotate(text=str(cluster_id), xy=((xmin + xmax) / 2, 0), ha="center")
            ax.axvspan(
                xmin=xmin, xmax=xmax, ymin=-1, ymax=1, color=cluster_colors[cluster_id]
            )

        ax.set_xlabel("Progression of corpus")
        ax.set_yticks([])
        ax.set_title("Corpus and cluster labels")

        if cluster_summaries:
            cluster_summaries = {
                idx: k["summary"] for idx, k in cluster_summaries.items()
            }
            for k, v in cluster_summaries.items():
                print("{} : {}".format(k, v))

        return result
