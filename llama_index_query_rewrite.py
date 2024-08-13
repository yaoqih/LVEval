from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from test_task import task_list
openai_embedding = OpenAIEmbedding(api_key="sk-...", api_base="http://172.10.10.162:18001/v1")
llm = OpenAI(api_key="sk-...", api_base="http://172.10.10.162:18005/v1", temperature=0, max_tokens=12800,timeout=600)
from typing import List

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
import asyncio
from typing import List
from llama_index.core.schema import NodeWithScore
from tqdm.asyncio import tqdm

def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]

class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""
    query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
    )
    query_gen_prompt = PromptTemplate(query_gen_prompt_str)
    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        super().__init__()
    def generate_queries(llm, query_str: str, num_queries: int = 4):
        fmt_prompt = FusionRetriever.query_gen_prompt.format(
            num_queries=num_queries - 1, query=query_str
        )
        response = llm.complete(fmt_prompt)
        queries = response.text.split("\n")
        return queries
    async def run_queries(queries, retrievers):
        """Run queries against retrievers."""
        tasks = []
        for query in queries:
            for i, retriever in enumerate(retrievers):
                tasks.append(retriever.aretrieve(query))

        task_results = [retriever.retrieve(query) for query in queries for i, retriever in enumerate(retrievers) ]

        results_dict = {}
        for i, (query, query_result) in enumerate(zip(queries, task_results)):
            results_dict[(query, i)] = query_result

        return results_dict
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = FusionRetriever.generate_queries(
            self._llm, query_bundle.query_str, num_queries=4
        )
        results = asyncio.run(FusionRetriever.run_queries(queries, self._retrievers))
        final_results = fuse_results(
            results, similarity_top_k=self._similarity_top_k
        )
        return final_results
def query(json_obj,task_index):
    template="""
    Answer the question based on the given passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context_str}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {query_str}\nAnswer:
    """
    qa_template = PromptTemplate(template)
    documents = [Document(text=json_obj['context'])]
    text_splitter = SentenceSplitter(chunk_size=task_list[task_index]['args']['chunk_size'], chunk_overlap=10)
    vector_index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter],embed_model=openai_embedding)
    # query_engine = vector_index.as_query_engine(llm=llm, text_qa_template=qa_template)
    # configure retriever

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(llm=llm,text_qa_template=qa_template)
    # get retrievers
    ## vector retriever
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)

    ## bm25 retriever
    # bm25_retriever = BM25Retriever.from_defaults(
    #     docstore=vector_index.docstore, similarity_top_k=1
    # )
    fusion_retriever = FusionRetriever(
    llm, [vector_retriever], similarity_top_k=3
    )

    query_engine = RetrieverQueryEngine(fusion_retriever,response_synthesizer=response_synthesizer)
    response = query_engine.query(json_obj['input'])

    return response.response
if __name__ == "__main__":
    json_obj = {
        "input": "浙江的省会有西湖吗?",
        "context": "浙江的省会是杭州，杭州有西湖"
    }
    print(query(json_obj,5))