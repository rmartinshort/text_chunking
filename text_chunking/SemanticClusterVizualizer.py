from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from text_chunking.splitting.SemanticSplitGenerator import SemanticSplitGenerator
from text_chunking.utils.SemanticGroupUtils import SemanticGroupUtils
from text_chunking.llm.chain import ChunkSummarizer
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class SemanticClusterVizualizer:

    def __init__(
        self,
        api_key,
        llm_model="gpt-4o-mini",
        temperature=0,
        embeddings_model="text-embedding-3-small",
    ):

        self.llm = ChatOpenAI(model=llm_model, temperature=temperature, api_key=api_key)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model, api_key=api_key)
        self.split_generator = SemanticSplitGenerator(ChunkSummarizer(self.llm))
        self.split_visualizer = SemanticGroupUtils()

    def split_documents(
        self, splitter, documents, min_chunk_len: int, verbose: bool = True
    ):
        """
        RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=0,
            is_separator_regex=False
        )
        """

        self.splitter = splitter
        logging.info("Splitting text with original splitter")
        doc_splits = self.splitter.split_text(documents)
        doc_splits = self.merge_short_documents(doc_splits, min_len=min_chunk_len)

        if verbose:
            # print some stats about the chunks that have been created
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
    def merge_short_documents(split_texts, min_len=100):
        """
        If we find a document with length < min len, just attach it to the end
        of the previous document. This prevents a situation where tiny document chunks
        might be incorrectly classified. There should be a relationship between min len and
        chunk size

        :param min_len:
        :return:
        """

        merged_splits = []
        for text in split_texts:
            if merged_splits and (len(text) < min_len):
                last_text = merged_splits.pop()
                merged_splits.append(last_text + " " + text)
            else:
                merged_splits.append(text)

        return merged_splits

    def embed_original_document_splits(self, doc_splits):
        """

        :param doc_splits:
        :return:
        """

        original_split_embeddings = self.embeddings.embed_documents(doc_splits)
        return original_split_embeddings

    def embed_semantic_groups(self, semantic_groups):
        """

        :param semantic_groups:
        :return:
        """

        semantic_group_embeddings = self.embeddings.embed_documents(semantic_groups)
        return semantic_group_embeddings

    def generate_breakpoints(
        self,
        doc_splits,
        doc_split_embeddings,
        length_threshold=10000,
        plot=True,
        verbose=True,
    ):
        """

        :param doc_splits:
        :param doc_split_embeddings:
        :param length_threshold:
        :param plot:
        :return:
        """

        self.split_generator.split_texts = doc_splits
        self.split_generator.split_text_embeddings = doc_split_embeddings

        breakpoints = self.split_generator.build_chunks_stack(length_threshold)
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
            lsplits = len(doc_splits)
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
        self, semantic_groups, semantic_group_embeddings, n_clusters=10
    ):
        """

        :param semantic_groups:
        :param semantic_group_embeddings:
        :param n_clusters:
        :return:
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

            return None, None

    def generate_cluster_labels(self, semantic_group_clusters, plot=True):

        semantic_group_summaries = self.split_generator.build_semantic_group_summaries(
            semantic_group_clusters
        )

        if plot:
            # plot the grouped splits and their summaries
            summaries = [v["text"] for k, v in semantic_group_summaries.items()]
            semantic_group_cluster_embeddings = self.embeddings.embed_documents(
                summaries
            )
            self.split_visualizer.plot_chunks_and_summaries(
                semantic_group_cluster_embeddings,
                [v["summary"] for k, v in semantic_group_summaries.items()],
            )

        return semantic_group_summaries
