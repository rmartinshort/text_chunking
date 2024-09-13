import langchain
from langchain.prompts import PromptTemplate
from dataclasses import dataclass


@dataclass
class ChunkSummaryPrompt:
    system_prompt: str = """
        You are an expert at summarization and information extraction from text. You will be given a chunk of text from a document and your
        task is to summarize what's happening in this chunk using fewer than 10 words. 

        Read through the entire chunk first and think carefully about the main points. Then produce your summary.

        Chunk to summarize: {current_chunk}
    """

    prompt: langchain.prompts.PromptTemplate = PromptTemplate(
        input_variables=["current_chunk"],
        template=system_prompt,
    )
