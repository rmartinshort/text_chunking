import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
from typing import List
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class SemanticSplitGenerator:

    def __init__(self, llm_chain, split_texts=None, split_text_embeddings=None):

        self.summarizer = llm_chain
        self.split_texts = split_texts
        self.split_text_embeddings = split_text_embeddings

    def build_chunk_cosine_distances(self):
        """

        :return:
        """

        len_embeddings = len(self.split_text_embeddings)
        cdists = np.empty(len_embeddings - 1)

        for i in range(1, len_embeddings):
            cdists[i - 1] = cosine(
                self.split_text_embeddings[i], self.split_text_embeddings[i - 1]
            )

        return cdists

    @staticmethod
    def get_text_length(group_of_splits):
        """

        :param group_of_splits:
        :return:
        """
        ln = 0
        for text in group_of_splits:
            ln += len(text)
        return ln

    @staticmethod
    def get_breakpoints(embeddings, start: int = 0, end: int = None, threshold: int = 0.95):
        """

        :param embeddings:
        :param start:
        :param end:
        :param threshold:
        :return:
        """

        if end:
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
        self, length_threshold: int = 20000, cosine_distance_threshold: int = 0.95
    ):
        """

        :param length_threshold:
        :param cosine_distance_threshold:
        :return:
        """

        S = [(0, len(self.split_texts))]
        all_breakpoints = set()
        while S:
            id_start, id_end = S.pop()
            updated_breakpoints = self.get_breakpoints(
                self.split_text_embeddings,
                start=id_start,
                end=id_end,
                threshold=cosine_distance_threshold,
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

    def build_semantic_groups(self, breakpoints: List):
        """

        :param breakpoints:
        :return:
        """

        start_index = 0
        grouped_texts = []
        # add ability to collect the indices of the original splits that are in each group
        for break_point in breakpoints:
            grouped_texts.append(
                "".join([x for x in self.split_texts[start_index : break_point + 1]])
            )
            start_index = break_point + 1

        return grouped_texts

    def build_semantic_group_clusters(self, semantic_groups, cluster_ids):

        cluster_groups = defaultdict(str)
        for label, text in zip(cluster_ids, semantic_groups):
            cluster_groups[label] += " " + text

        return cluster_groups

    def build_semantic_group_summaries(
        self, semantic_groups_to_summarize, verbose=True
    ):

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
                logging.info("Chunk {} summary: {}".format(i+1, res[0]))
            summaries.append(res[0])

        # collect result into dictionary
        summary_dict = {}
        idx = 0
        for summary, texts in zip(summaries, semantic_groups_to_summarize):
            summary_dict[idx] = {"summary": summary, "text": texts}
            idx += 1

        return summary_dict
