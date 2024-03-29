import time
import logging
from langchain_community.vectorstores import Neo4jVector
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import (EMBEDDING_MODEL , NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
                    NEO4J_DB, NEO4J_INDEX,NEO4J_NODE_LABEL, NEO4J_TXTNODE_PROP,
                    NEO4J_EMBNODE_PROP)
from neo4j.exceptions import ConfigurationError
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
class VDriver(object):
    """ This is a class that allows you to vectorize a text using Neo4j, langcain,
    SemanticChunker, and HuggingFaceBgeEmbeddings
    """
    def __init__(self):
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        try:
            self.__embeddings = HuggingFaceBgeEmbeddings(
                model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        except EnvironmentError:
            self.__embeddings = False
    # def __text_splitter(self):
    #     """ This method use SemanticChunker to split the text into sentences"""
    #     if self.__embeddings:
    #         return SemanticChunker(self.__embeddings)
    #     return False

    def __create_documents(self, content, title):
        # """ This method creates a documents' object from a text """
        sentences = content.split('.')
        docs = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                docs.append(Document(page_content=sentence, metadata={'title': title}))

        return docs

    def __get_db_from_existing_index(self):
        """ This method loads a Neo4j index from the existing one in the db"""
        if self.__embeddings:
            try:
                return Neo4jVector.from_existing_index(
                    self.__embeddings,
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database = NEO4J_DB,
                    index_name = NEO4J_INDEX,
                    node_label=NEO4J_NODE_LABEL,
                    text_node_property=NEO4J_TXTNODE_PROP,
                    embedding_node_property=NEO4J_EMBNODE_PROP,
                )
            except  (ConfigurationError,ValueError):
                return False
        return False
    def clear_db(self):
        """ This method clears the Neo4j database after loading the vectors"""
        db = self.__get_db_from_existing_index()
        if db is not False:
            return db.query("MATCH(n) DETACH DELETE n")
        return False

    def vectorize(self, content, title):
        """ This method clear first the existing index and then vectorize the documents provided by the content"""
        """ This method clear first the existing index and then vectorize the documents provided by the content"""
        start_time = time.time()
        docs = self.__create_documents(content , title)

        chunks = [docs[x:x + 50] for x in range(0, len(docs), 50)]

        for docs_chunked in chunks:
            db = Neo4jVector.from_documents(
                    docs_chunked, self.__embeddings, url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database=NEO4J_DB,
                    index_name=NEO4J_INDEX,
                    node_label=title,
                    text_node_property=NEO4J_TXTNODE_PROP,
                    embedding_node_property=NEO4J_EMBNODE_PROP,
                    create_id_index=True)
            time.sleep(1)
        logging.info("--- %s seconds from_documents---" % (time.time() - start_time))

        return False
    def query(self, query, size=5):
        """ This method queries the Neo4j for documents"""
        db = self.__get_db_from_existing_index()
        if db is not False:
            try:
                return db.similarity_search_with_score(query, k=size)
            except (ConfigurationError,ValueError):
                return False
        return False