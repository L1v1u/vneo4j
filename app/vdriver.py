from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import (EMBEDDING_MODEL , NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
                    NEO4J_DB, NEO4J_INDEX,NEO4J_NODE_LABEL, NEO4J_TXTNODE_PROP,
                    NEO4J_EMBNODE_PROP)

class VDriver(object):
    """ This is a class that allows you to vectorize a text using Neo4j, langcain,
    SemanticChunker, and HuggingFaceBgeEmbeddings
    """
    def __init__(self):
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    def __text_splitter(self):
        """ This method use SemanticChunker to split the text into sentences"""
        return SemanticChunker(self.embeddings)

    def __create_documents(self, content):
        """ This method creates a documents' object from a text """
        return self.__text_splitter().create_documents([content])

    def __get_db_from_existing_index(self):
        """ This method loads a Neo4j index from the existing one in the db"""
        return Neo4jVector.from_existing_index(
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database = NEO4J_DB,
            index_name = NEO4J_INDEX,
            node_label=NEO4J_NODE_LABEL,
            text_node_property=NEO4J_TXTNODE_PROP,
            embedding_node_property=NEO4J_EMBNODE_PROP,
        )

    def clear_db(self):
        """ This method clears the Neo4j database after loading the vectors"""
        return self.__get_db_from_existing_index().query("MATCH(n) DETACH DELETE n")

    def vectorize(self, content):
        """ This method clear first the existing index and then vectorize the documents provided by the content"""
        self.clear_db()
        docs = self.__create_documents(content)
        db = Neo4jVector.from_documents(
            docs, self.embeddings, url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DB,
            index_name=NEO4J_INDEX,
            node_label=NEO4J_NODE_LABEL,
            text_node_property=NEO4J_TXTNODE_PROP,
            embedding_node_property=NEO4J_EMBNODE_PROP,
            create_id_index=True,
        )
        return True
    def query(self, query, size=5):
        """ This method queries the Neo4j for documents"""
        return self.__get_db_from_existing_index().similarity_search_with_score(query, k=size)
