from llama_index.core import Document
from llama_index.core.async_utils import  run_jobs
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from typing import Any, Dict, List, Optional, Sequence, cast
from llama_index.core.settings import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from test_task import task_list
from llama_index.core.extractors import BaseExtractor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.llm import LLM
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
openai_embedding = OpenAIEmbedding(api_key="sk-...", api_base="http://172.10.10.162:18001/v1")
llm = OpenAI(api_key="sk-...", api_base="http://172.10.10.162:18005/v1", temperature=0, max_tokens=12800,timeout=600)
DEFAULT_TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: """


DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: """

FACT_EXTRACT_TEMPLATE = """\
You are an AI assistant specialized in extracting factual claims from text. Your task is to analyze the given draft response and extract all factual statements that require validation. Please follow these guidelines:
    1. Read the provided question and draft response carefully.
    2. Identify all factual claims present in the draft response.
    3. Break down complex statements into simpler, atomic factual claims.
    4. Present each extracted claim as a separate item in a list.
    5. Ensure that each claim is a complete, standalone statement.
    6. Do not include opinions, speculations, or subjective statements.
    7. Focus only on extracting claims directly related to the question and response.
    8. Write one per line.
    Here's an example to guide you:

    Question: When is Frédéric Chopin's father's birthday?
    Draft Response: Frédéric Chopin's father is Nicolas Chopin, he was born on June 17, 1771.

    Extracted Claims:
    1. Frédéric Chopin's father is Nicolas Chopin.
    2. Nicolas Chopin was born on June 17, 1771.

    Now, please analyze the following question and draft response, then extract the factual claims,If the content provided does not contain any factual claims that directly answer the question, please just answer: No! without any explanation:

    Question: {question_str}
    Draft Response: {context_str}

    Extracted Claims:
    """

class TitleExtractor(BaseExtractor):
    """Title extractor. Useful for long documents. Extracts `document_title`
    metadata field.

    Args:
        llm (Optional[LLM]): LLM
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into
            a document-level title
    """

    is_text_node_only: bool = False  # can work for mixture of text and non-text nodes
    llm: LLMPredictorType = Field(description="The LLM to use for generation.")
    nodes: int = Field(
        default=5,
        description="The number of nodes to extract titles from.",
        gt=0,
    )
    node_template: str = Field(
        default=DEFAULT_TITLE_NODE_TEMPLATE,
        description="The prompt template to extract titles with.",
    )
    combine_template: str = Field(
        default=DEFAULT_TITLE_COMBINE_TEMPLATE,
        description="The prompt template to merge titles with.",
    )
    question: str =Field(
        default=DEFAULT_TITLE_NODE_TEMPLATE,
        description="The prompt template to extract titles with.",
    )
    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor = None,
        nodes: int = 1,
        question="",
        node_template: str = DEFAULT_TITLE_NODE_TEMPLATE,
        combine_template: str = DEFAULT_TITLE_COMBINE_TEMPLATE,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        """Init params."""
        if nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            nodes=nodes,
            node_template=node_template,
            combine_template=combine_template,
            num_workers=num_workers,
            **kwargs,
        )
        self.question = question    

    @classmethod
    def class_name(cls) -> str:
        return "TitleExtractor"

    async def aextract(self, nodes):
        nodes_by_doc_id = self.separate_nodes_by_ref_id(nodes)
        titles_by_doc_id = await self.extract_titles(nodes_by_doc_id)
        return [{"document_title": titles_by_doc_id[node.ref_doc_id]} for node in nodes]

    def filter_nodes(self, nodes):
        filtered_nodes= []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            filtered_nodes.append(node)
        return filtered_nodes

    def separate_nodes_by_ref_id(self, nodes) :
        separated_items= {}

        for node in nodes:
            key = node.ref_doc_id
            if key not in separated_items:
                separated_items[key] = []

            if len(separated_items[key]) < self.nodes:
                separated_items[key].append(node)

        return separated_items

    async def extract_titles(self, nodes_by_doc_id) :
        titles_by_doc_id = {}
        for key, nodes in nodes_by_doc_id.items():
            # title_candidates = await self.get_title_candidates(nodes)
            # combined_titles = ", ".join(title_candidates)
            titles_by_doc_id[key] = await self.llm.apredict(
                PromptTemplate(template=FACT_EXTRACT_TEMPLATE),
                context_str=nodes[0].text,
                question_str=self.question,
            )
        return titles_by_doc_id

    async def get_title_candidates(self, nodes):
        title_jobs = [
            self.llm.apredict(
                PromptTemplate(template=self.node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes
        ]
        return await run_jobs(
            title_jobs, show_progress=self.show_progress, workers=self.num_workers
        )
def query(json_obj,task_index):
    template="""
    Answer the question based on the given passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context_str}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {query_str}\nAnswer:
    """
    qa_template = PromptTemplate(template)
    documents = [Document(text=json_obj['context'])]
    chunk_size=1024
    if 'chunk_size' in task_list[task_index]['args']:
        chunk_size = task_list[task_index]['args']['chunk_size']
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=10)
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, page in enumerate(documents):
        page_text = page.text
        cur_text_chunks = text_splitter.split_text(page_text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc_idx = doc_idxs[idx]
        src_page = documents[src_doc_idx]
        nodes.append(node)
    extractors = [
        TitleExtractor(llm=llm,question=json_obj['input']),
    ]
    pipeline = IngestionPipeline(
        transformations=extractors,
    )
    orig_nodes =  pipeline.run(nodes=nodes, in_place=False)
    # vector_index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter],embed_model=openai_embedding)
    # query_engine = vector_index.as_query_engine(llm=llm, text_qa_template=qa_template)
    # configure retriever
    vector_index = VectorStoreIndex(orig_nodes,embed_model=openai_embedding)

    top_k=2
    if 'top_k' in task_list[task_index]['args']:
        top_k = task_list[task_index]['args']['top_k']
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=top_k
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(llm=llm,text_qa_template=qa_template)

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
    )
    response = query_engine.query(json_obj['input'])
    return response.response
if __name__ == "__main__":
    json_obj = {
        "input": "what is the capital of China?",
        "context": "Beijing is the capital of China."
    }
    print(query(json_obj,1))