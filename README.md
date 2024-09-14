# text_chunking

Exploration of semantic chunking and chunk classification. This package allows 
semantic chunking of text and produces visuals that allow us to how and why the 
chunks are being made. 

## Example use case 

First, install the requirements and put your open ai api key into a .env file
```
from text_chunking.SemanticClusterVizualizer import SemanticClusterVizualizer
from text_chunking.utils.secrets import load_secrets
from text_chunking.datasets.test_text_dataset import TestText, TextTextNovel
from langchain_text_splitters import RecursiveCharacterTextSplitter
import seaborn as sns

sns.set_context("talk")
secrets = load_secrets()

semantic_chunker = SemanticClusterVizualizer(api_key=secrets["OPENAI_API_KEY"])

# set up a standard splitter
splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=0,
        is_separator_regex=False
)

# split the document into chunks
original_split_texts = semantic_chunker.split_documents(
    splitter, 
    TestText.testing_text, 
    min_chunk_len=100, 
    verbose=True
)

# run embeddings
original_split_text_embeddings = semantic_chunker.embed_original_document_splits(original_split_texts)

# generate breakpoints, use length threshold to decide which 
# sections to further subdivide 
breakpoints, semantic_groups = semantic_chunker.generate_breakpoints(
    original_split_texts,
    original_split_text_embeddings,
    length_threshold=1000
)

# embed the groups that have been made from the breakpoints
semantic_group_embeddings = semantic_chunker.embed_semantic_groups(semantic_groups)

# cluster the groups
splits_df, semantic_group_clusters = semantic_chunker.vizualize_semantic_groups(
    semantic_groups,
    semantic_group_embeddings,
    n_clusters=8
)

# generate cluster summaries
cluster_summaries = semantic_chunker.generate_cluster_labels(
    semantic_group_clusters, plot=True
)

# generate cluster bounds
semantic_cluster_bounds = semantic_chunker.split_visualizer.plot_corpus_and_clusters(
    splits_df, cluster_summaries
)
```