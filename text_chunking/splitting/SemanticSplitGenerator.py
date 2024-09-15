import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
from typing import List, Dict, Any
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class SemanticSplitGenerator:
    """
    A class for generating semantic splits of text based on embeddings and cosine distances.
    """

    def __init__(
        self,
        llm_chain: Any,
        split_texts: List[str] = None,
        split_text_embeddings: List[np.ndarray] = None,
    ) -> None:
        """
        Initializes the SemanticSplitGenerator with a summarizer, split texts, and their embeddings.

        Args:
            llm_chain (Any): A language model chain used for summarization.
            split_texts (List[str], optional): A list of text splits. Defaults to None.
            split_text_embeddings (List[np.ndarray], optional): A list of embeddings for the text splits. Defaults to None.
        """
        self.summarizer = llm_chain
        self.split_texts = split_texts
        self.split_text_embeddings = split_text_embeddings

    def build_chunk_cosine_distances(self) -> np.ndarray:
        """
        Calculates the cosine distances between consecutive text chunk embeddings.

        Returns:
            np.ndarray: An array of cosine distances between consecutive embeddings.
        """
        len_embeddings = len(self.split_text_embeddings)
        cdists = np.empty(len_embeddings - 1)

        for i in range(1, len_embeddings):
            cdists[i - 1] = cosine(
                self.split_text_embeddings[i], self.split_text_embeddings[i - 1]
            )

        return cdists

    @staticmethod
    def get_text_length(group_of_splits: List[str]) -> int:
        """
        Calculates the total length of a group of text splits.

        Args:
            group_of_splits (List[str]): A list of text splits.

        Returns:
            int: The total length of the text splits.
        """
        return sum(len(text) for text in group_of_splits)

    @staticmethod
    def get_breakpoints(
        embeddings: List[np.ndarray],
        start: int = 0,
        end: int = None,
        threshold: float = 0.95,
    ) -> np.ndarray:
        """
        Identifies breakpoints in embeddings based on cosine distance threshold.

        Args:
            embeddings (List[np.ndarray]): A list of embeddings.
            start (int, optional): The starting index for processing. Defaults to 0.
            end (int, optional): The ending index for processing. Defaults to None.
            threshold (float, optional): The percentile threshold for determining significant distance changes. Defaults to 0.95.

        Returns:
            np.ndarray: An array of indices where breakpoints occur.
        """
        if end is not None:
            embeddings_windowed = embeddings[start:end]
        else:
            embeddings_windowed = embeddings[start:]

        len_embeddings = len(embeddings_windowed)
        cdists = np.empty(len_embeddings - 1)

        for i in range(1, len_embeddings):
            cdists[i - 1] = cosine(embeddings_windowed[i], embeddings_windowed[i - 1])

        difference_threshold = np.percentile(cdists, 100 * threshold, axis=0)
        difference_exceeding = np.argwhere(cdists >= difference_threshold).ravel()

        return difference_exceeding

    def build_chunks_stack(
        self, length_threshold: int = 20000, cosine_distance_percentile_threshold: float = 0.95
    ) -> np.ndarray:
        """
        Builds a stack of text chunks based on length and cosine distance thresholds.

        Args:
            length_threshold (int, optional): Minimum length for a text chunk to be considered valid. Defaults to 20000.
            cosine_distance_percentile_threshold (float, optional): Cosine distance percentile threshold for determining breakpoints. Defaults to 0.95.

        Returns:
            np.ndarray: An array of indices representing the breakpoints of the chunks.
        """
        S = [(0, len(self.split_texts))]
        all_breakpoints = set()
        while S:
            id_start, id_end = S.pop()
            updated_breakpoints = self.get_breakpoints(
                self.split_text_embeddings,
                start=id_start,
                end=id_end,
                threshold=cosine_distance_percentile_threshold,
            )
            updated_breakpoints += id_start
            updated_breakpoints = np.concatenate(
                (np.array([id_start - 1]), updated_breakpoints, np.array([id_end]))
            )
            for index in updated_breakpoints:
                text_group = self.split_texts[id_start : index + 1]
                if (len(text_group) > 2) and (
                    self.get_text_length(text_group) >= length_threshold
                ):
                    S.append((id_start, index))
                id_start = index + 1
            all_breakpoints.update(updated_breakpoints)

        return np.array(sorted(all_breakpoints))[1:-1]

    def build_semantic_groups(self, breakpoints: List[int]) -> List[str]:
        """
        Constructs semantic groups from text splits using specified breakpoints.

        Args:
            breakpoints (List[int]): A list of indices representing breakpoints.

        Returns:
            List[str]: A list of concatenated text strings for each semantic group.
        """
        start_index = 0
        grouped_texts = []
        # add end criteria
        breakpoints = np.append(breakpoints, [-1])
        for break_point in breakpoints:

            # we're at the end of the text
            if break_point == -1:
                grouped_texts.append(
                    " ".join([x for x in self.split_texts[start_index:]])
                )

            else:

                grouped_texts.append(
                    " ".join([x for x in self.split_texts[start_index : break_point + 1]])
                )

            start_index = break_point + 1

        return grouped_texts

    def build_semantic_group_clusters(
        self, semantic_groups: List[str], cluster_ids: List[int]
    ) -> Dict[int, str]:
        """
        Aggregates semantic groups into clusters based on cluster IDs.

        Args:
            semantic_groups (List[str]): A list of semantic group texts.
            cluster_ids (List[int]): A list of cluster IDs corresponding to each semantic group.

        Returns:
            Dict[int, str]: A dictionary where keys are cluster IDs and values are aggregated text strings.
        """
        cluster_groups = defaultdict(str)
        for label, text in zip(cluster_ids, semantic_groups):
            cluster_groups[label] += " " + text

        return cluster_groups

    def build_semantic_group_summaries(
        self, semantic_groups_to_summarize: List[str], verbose: bool = True
    ) -> Dict[int, Dict[str, str]]:
        """
        Generates summaries for semantic groups using the summarizer.

        Args:
            semantic_groups_to_summarize (List[str]): A list of semantic groups to summarize.
            verbose (bool, optional): Whether to log the summaries. Defaults to True.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary where keys are indices and values are dictionaries containing summaries and original texts.
        """
        if isinstance(semantic_groups_to_summarize, dict):
            semantic_groups_to_summarize = semantic_groups_to_summarize.values()

        summaries = []
        for i, chunk in enumerate(semantic_groups_to_summarize):
            input_dict = {
                "current_chunk": chunk,
            }

            res = self.summarizer.run_and_count_tokens(input_dict)
            if verbose:
                logging.info("-" * 20)
                logging.info("Chunk {} summary: {}".format(i + 1, res[0]))
            summaries.append(res[0])

        summary_dict = {}
        idx = 0
        for summary, texts in zip(summaries, semantic_groups_to_summarize):
            summary_dict[idx] = {"summary": summary, "text": texts}
            idx += 1

        return summary_dict
