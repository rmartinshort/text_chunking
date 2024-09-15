from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from text_chunking.splitting.SemanticSplitGenerator import SemanticSplitGenerator
from text_chunking.utils.SemanticGroupUtils import SemanticGroupUtils
from text_chunking.llm.chain import ChunkSummarizer
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class SemanticClusterVisualizer:
    """
    A class for visualizing semantic clusters of text documents using embeddings and language models.
    """

    def __init__(
        self,
        api_key: str,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0,
        embeddings_model: str = "text-embedding-3-small",
    ) -> None:
        """
        Initializes the SemanticClusterVizualizer with language model and embedding configurations.

        Args:
            api_key (str): API key for accessing language and embedding models.
            llm_model (str, optional): Language model to use. Defaults to "gpt-4o-mini".
            temperature (float, optional): Temperature setting for the language model. Defaults to 0.
            embeddings_model (str, optional): Embeddings model to use. Defaults to "text-embedding-3-small".
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature, api_key=api_key)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model, api_key=api_key)
        self.split_generator = SemanticSplitGenerator(ChunkSummarizer(self.llm))
        self.split_visualizer = SemanticGroupUtils()

    def split_documents(
        self, splitter: Any, documents: str, min_chunk_len: int, verbose: bool = True
    ) -> List[str]:
        """
        Splits documents into chunks using a specified splitter and merges short documents.

        Args:
            splitter (Any): A text splitter object.
            documents (str): The document text to be split.
            min_chunk_len (int): Minimum length for a chunk to avoid merging.
            verbose (bool, optional): Whether to log the splitting process. Defaults to True.

        Returns:
            List[str]: A list of document chunks.
        """
        self.splitter = splitter
        logging.info("Splitting text with original splitter")
        doc_splits = self.splitter.split_text(documents)
        doc_splits = self.merge_short_documents(doc_splits, min_len=min_chunk_len)

        if verbose:
            max_len = 0
            min_len = float("inf")
            lsplits = len(doc_splits)
            sum_len = 0
            for text in doc_splits:
                lt = len(text)
                sum_len += lt
                max_len = max(max_len, lt)
                min_len = min(min_len, lt)
            mean_len = sum_len / lsplits
            logging.info(
                "Creating {} chunks\nMean len: {}\nMax len: {}\nMin len: {}".format(
                    lsplits, mean_len, max_len, min_len
                )
            )

        return doc_splits

    @staticmethod
    def merge_short_documents(split_texts: List[str], min_len: int = 100) -> List[str]:
        """
        Merges short documents into preceding documents to prevent incorrect classification.

        Args:
            split_texts (List[str]): A list of split text documents.
            min_len (int, optional): Minimum length for a document to avoid merging. Defaults to 100.

        Returns:
            List[str]: A list of merged document chunks.
        """
        merged_splits = []
        for text in split_texts:
            if merged_splits and (len(text) < min_len):
                last_text = merged_splits.pop()
                merged_splits.append(last_text + " " + text)
            else:
                merged_splits.append(text)

        return merged_splits

    def embed_original_document_splits(self, doc_splits: List[str]) -> List[np.ndarray]:
        """
        Embeds the original document splits using the specified embeddings model.

        Args:
            doc_splits (List[str]): A list of document splits.

        Returns:
            List[np.ndarray]: A list of embeddings for the document splits.
        """
        original_split_embeddings = self.embeddings.embed_documents(doc_splits)
        return original_split_embeddings

    def embed_semantic_groups(self, semantic_groups: List[str]) -> List[np.ndarray]:
        """
        Embeds semantic groups using the specified embeddings model.

        Args:
            semantic_groups (List[str]): A list of semantic groups.

        Returns:
            List[np.ndarray]: A list of embeddings for the semantic groups.
        """
        semantic_group_embeddings = self.embeddings.embed_documents(semantic_groups)
        return semantic_group_embeddings

    def generate_breakpoints(
        self,
        doc_splits: List[str],
        doc_split_embeddings: List[np.ndarray],
        length_threshold: int = 1e9,
        percentile_threshold: float = 0.95,
        plot: bool = True,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generates breakpoints in document splits based on cosine distances and length threshold.

        Args:
            doc_splits (List[str]): A list of document splits.
            doc_split_embeddings (List[np.ndarray]): A list of embeddings for the document splits.
            length_threshold (int, optional): Minimum length for a chunk to be considered valid. Defaults to 10000.
            percentile_threshold (float, optional): Percentile of cosine distance used to choose breakpoints
            plot (bool, optional): Whether to plot the chunk differences and breakpoints. Defaults to True.
            verbose (bool, optional): Whether to log the process. Defaults to True.

        Returns:
            Tuple[np.ndarray, List[str]]: A tuple containing breakpoints and semantic groups.
        """
        self.split_generator.split_texts = doc_splits
        self.split_generator.split_text_embeddings = doc_split_embeddings

        breakpoints = self.split_generator.build_chunks_stack(length_threshold, cosine_distance_percentile_threshold=percentile_threshold)
        semantic_groups = self.split_generator.build_semantic_groups(breakpoints)
        chunk_cosine_distances = self.split_generator.build_chunk_cosine_distances()

        if plot:
            self.split_visualizer.plot_chunk_differences_and_breakpoints(
                doc_splits, breakpoints, chunk_cosine_distances
            )

        if verbose:
            n_groups = len(semantic_groups)
            max_len = 0
            min_len = float("inf")
            sum_len = 0
            for text in semantic_groups:
                lt = len(text)
                sum_len += lt
                max_len = max(max_len, lt)
                min_len = min(min_len, lt)
            mean_len = sum_len / n_groups
            logging.info(
                "Creating {} semantic groups\nMean len: {}\nMax len: {}\nMin len: {}".format(
                    n_groups, mean_len, max_len, min_len
                )
            )

        return breakpoints, semantic_groups

    def vizualize_semantic_groups(
        self,
        semantic_groups: List[str],
        semantic_group_embeddings: List[np.ndarray],
        n_clusters: int = 10,
    ) -> Tuple[Any, Dict[int, str]]:
        """
        Visualizes semantic groups and optionally clusters them.

        Args:
            semantic_groups (List[str]): A list of semantic groups.
            semantic_group_embeddings (List[np.ndarray]): A list of embeddings for the semantic groups.
            n_clusters (int, optional): Number of clusters to create. Defaults to 10.

        Returns:
            Tuple[Any, Dict[int, str]]: A tuple containing a DataFrame of splits and a dictionary of semantic group clusters.
        """
        self.split_visualizer.plot_2d_semantic_embeddings(
            semantic_group_embeddings, semantic_groups
        )

        if n_clusters > 1:
            cluster_linkage = self.split_visualizer.create_hierarchical_clustering(
                semantic_group_embeddings
            )
            splits_df = self.split_visualizer.plot_2d_semantic_embeddings_with_clusters(
                semantic_group_embeddings,
                semantic_groups,
                cluster_linkage,
                n_clusters=n_clusters,
            )
            semantic_group_clusters = (
                self.split_generator.build_semantic_group_clusters(
                    semantic_groups, splits_df["cluster_label"].values
                )
            )

            return splits_df, semantic_group_clusters

        else:
            return None, {}

    def generate_cluster_labels(
        self, semantic_group_clusters: Dict[int, str], plot: bool = True
    ) -> Dict[int, Dict[str, str]]:
        """
        Generates and optionally plots summaries for semantic group clusters.

        Args:
            semantic_group_clusters (Dict[int, str]): A dictionary of semantic group clusters.
            plot (bool, optional): Whether to plot the summaries. Defaults to True.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary containing summaries and original texts for each cluster.
        """
        semantic_group_summaries = self.split_generator.build_semantic_group_summaries(
            list(semantic_group_clusters.values())
        )

        if plot:
            summaries = [v["text"] for v in semantic_group_summaries.values()]
            semantic_group_cluster_embeddings = self.embeddings.embed_documents(
                summaries
            )
            self.split_visualizer.plot_chunks_and_summaries(
                semantic_group_cluster_embeddings,
                [v["summary"] for v in semantic_group_summaries.values()],
            )

        return semantic_group_summaries
