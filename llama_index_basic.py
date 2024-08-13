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
from llama_index.core.postprocessor.types import BaseNodePostprocessor

class TextOrder(BaseNodePostprocessor):

    @classmethod
    def class_name(cls) -> str:
        return "TextOrder"

    def _postprocess_nodes(
        self,
        nodes,
        query_bundle= None,
    ) :
        """Postprocess nodes."""
        ordered_nodes= sorted(
            nodes, key=lambda x: x.node.start_char_idx if x.node.start_char_idx is not None else 0
        )
        return ordered_nodes

postprocessor = TextOrder()

openai_embedding = OpenAIEmbedding(api_key="sk-...", api_base="http://172.10.10.200:18001/v1")
llm = OpenAI(api_key="sk-...", api_base="http://172.10.10.200:18005/v1", temperature=0, max_tokens=12800,timeout=600)
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama
# openai_embedding = OllamaEmbedding(model_name='qwen2:0.5b')
# llm = Ollama(model='qwen2:0.5b')
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
    vector_index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter],embed_model=openai_embedding)
    # query_engine = vector_index.as_query_engine(llm=llm, text_qa_template=qa_template)
    # configure retriever
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
        node_postprocessors=[postprocessor] if 'text_order' in task_list[task_index]['args'] else []
    )
    response = query_engine.query(json_obj['input'])
    return response.response
if __name__ == "__main__":
    json_obj = {
        "input": "what is the capital of China?",
        "context": "Beijing is the capital of China."
    }
    print(query(json_obj,31))