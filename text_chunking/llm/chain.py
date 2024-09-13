from langchain_core.output_parsers.string import StrOutputParser
from langchain.callbacks import get_openai_callback
from text_chunking.llm.prompt import ChunkSummaryPrompt


class ChunkSummarizer(object):
    def __init__(self, llm):
        self.prompt = ChunkSummaryPrompt()
        self.llm = llm
        self.chain = self._set_up_chain()

    def _set_up_chain(self):
        return self.prompt.prompt | self.llm | StrOutputParser()

    def run_and_count_tokens(self, input_dict):
        with get_openai_callback() as cb:
            result = self.chain.invoke(input_dict)

        return result, cb
