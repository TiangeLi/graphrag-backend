from os import getenv

from ast import literal_eval
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncResult

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from helpers.constants import LARGE_EMBD
from helpers.custom_lcneo4j import Neo4jVector as CustomNeo4jVector

from typing_extensions import Any, Dict
from asyncio import gather
from collections import Counter

NEO4J_URI=getenv("NEO4J_URI")
NEO4J_USERNAME=getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=getenv("NEO4J_PASSWORD")

class VectorIndex(BaseModel):
    index_name: str
    embedding_dims: int
    similarity_function: str
    embd_model: str
    embd_node_prop: str

class DBRetriever(object):
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embd_model = LARGE_EMBD
        self.embedder = OpenAIEmbeddings(model=self.embd_model)
        self.guidelines_vector = VectorIndex(
            index_name="guideline_source_vector",
            embedding_dims=3072,
            similarity_function='cosine',
            embd_model=self.embd_model,
            embd_node_prop="embeddings"
        )
        self.tx_name_vector = VectorIndex(
            index_name="treatment_name_vector",
            embedding_dims=3072,
            similarity_function='cosine',
            embd_model=self.embd_model,
            embd_node_prop="name_embedding"
        )
        self.docs_sim_retriever = self._get_docs_sim_retriever()

    async def a_db_query(self, cypher: str, params: Dict[str, Any] = {}):
        return await self.driver.execute_query(cypher, parameters_=params, result_transformer_=AsyncResult.to_df)

    async def graph_traverse(self, start_node: str, start_node_type: str = '', limit_rels: str = '', depth: int = 1):
        if start_node_type: start_node_type = f":{start_node_type}"
        if limit_rels: limit_rels = f":{limit_rels}"
        cypher = f"""
            MATCH path = (start{start_node_type} {{name: "{start_node}"}})<-[r{limit_rels}*..{depth}]-()
            WITH DISTINCT relationships(path) AS rels, nodes(path) AS ns, start
            UNWIND RANGE(0, SIZE(rels) - 1) AS idx
            WITH ns[idx].name AS parent,  ns[idx + 1] as child
            WITH parent, child, 
                EXISTS((child)<-[{limit_rels}]-()) AS has_children, 
                EXISTS((child)<-[:RECOMMENDED_FOR]-(:TREATMENT)) AS has_tx_recs, 
                EXISTS((child)<-[:PRIMARY_ENTITY]-(:GUIDELINE)) AS has_exerpts
            WITH parent, COLLECT({{node: child.name, contains_considerations: has_children, contains_treatment_recommendations: has_tx_recs, contains_guideline_exerpts: has_exerpts}}) AS children
            RETURN parent, children
            ORDER BY parent"""
        result = await self.a_db_query(cypher)
        return {
            r['parent']: r['children'] 
            for _, r in result.iterrows()
        }
    
    async def a_sort_tx_by_type(self, tx_names: list[str]):
        async def _get_tx_by_type(tx_names: list[str], tx_type: str):
            cypher = f"""
                WITH {tx_names} AS tx_names
                UNWIND tx_names AS tx_name
                MATCH (n:{tx_type.upper()} {{name: tx_name}})
                OPTIONAL MATCH (n)-[:IS_TYPE_OF]->(parent)
                OPTIONAL MATCH (n)<-[:IS_TYPE_OF]-(child)
                WHERE child.name IN tx_names AND NOT parent.name IN tx_names
                WITH n, child, parent, tx_names
                OPTIONAL MATCH (child)-[r:RECOMMENDED_FOR]-()
                WITH n, child, parent, r, tx_names
                ORDER BY SIZE(r.recommended_by) DESC
                RETURN 
                    n.name AS name, 
                    collect(DISTINCT child.name) AS children,
                    CASE 
                        WHEN parent.name IN tx_names THEN null 
                        ELSE parent.name
                    END 
                    AS parent
                """
            result = await self.a_db_query(cypher)
            formatted = []
            all_children = []
            __all_parents = []
            for _, r in result.iterrows():
                all_children.extend(r['children'])
                __all_parents.append(r['parent'])
            parent_counts = Counter(__all_parents)
            all_parents = [parent for parent, count in parent_counts.items() if count > 1 and parent]

            __has_parent = []
            for _, r in result.iterrows():
                if r['name'] not in all_children:
                    if r['parent'] not in all_parents:
                        formatted.append({
                            'name': r['name'],
                            'children': r['children'],
                            'parent': r['parent']
                        })
                    elif r['parent'] in all_parents:
                        __has_parent.append({
                            'name': r['name'],
                            'parent': r['parent'],
                            'children': r['children']
                        })

            for parent in all_parents:
                entry = []
                for child in __has_parent:
                    if child['parent'] == parent:
                        entry.append(child)
                formatted.append({
                    'name': parent,
                    'children': entry,
                    'parent': ''
                })
            
            return formatted
        tx_types = ['SURGICAL', 'MEDICAL', 'CONSERVATIVE']
        response = await gather(*[_get_tx_by_type(tx_names, tx_type) for tx_type in tx_types])
        results = {k: v for k, v in zip(tx_types, response) if v}
        return results
    
    async def a_get_tx_name_by_similarity(self, query: str, tx_type: str = '', top_k: int = 5):
        __first_limit = 10
        query_embd = await self.embedder.aembed_query(query)
        if tx_type: tx_type = f":{tx_type}"
        cypher = f"""
            MATCH (t:TREATMENT{tx_type})
            OPTIONAL MATCH (t)-[r:RECOMMENDED_FOR]->()
            WITH DISTINCT t, vector.similarity.cosine(t.{self.tx_name_vector.embd_node_prop}, {query_embd}) AS similarity,
                CASE 
                    WHEN r IS NULL THEN 0
                    ELSE SIZE(r.recommended_by)
                END 
                AS recommended
            ORDER BY similarity DESC
            LIMIT {__first_limit}
            WITH DISTINCT t, MAX(recommended) AS recommended, similarity
            ORDER BY recommended DESC, similarity DESC
            LIMIT {top_k}
            WITH DISTINCT t
            OPTIONAL MATCH (t)<-[:PRIMARY_ENTITY]-(g:GUIDELINE)
            WITH t, {{name: g.name, content: g.content, metadata: g.metadata}} AS guideline
            RETURN DISTINCT t.name AS name, collect(guideline) AS guidelines
        """
        result = await self.a_db_query(cypher)
        for _, r in result.iterrows():
            if query.lower() in r['name'].lower():
                return {r['name']: r['guidelines']}
        return {r['name']: r['guidelines'] for _, r in result.iterrows()}
    
    async def a_get_considerations_by_similarity(self, query: str, top_k: int = 5):
        query_embd = await self.embedder.aembed_query(query)
        cypher = f"""
            MATCH (c:CONSIDERATION)
            WITH c, vector.similarity.cosine(c.{self.tx_name_vector.embd_node_prop}, {query_embd}) AS similarity
            ORDER BY similarity DESC
            LIMIT {top_k}
            RETURN c.name as name
        """
        result = await self.a_db_query(cypher)
        return [r['name'] for _, r in result.iterrows()]

    async def a_get_docs_by_similarity(self, query: str):
        ret = await self.docs_sim_retriever.asimilarity_search(
            query=query,
            k=4
        )
        return [Document(
            page_content=c.metadata['name'] + "\n\n" + c.page_content,
            metadata=literal_eval(c.metadata['metadata']),
            id = c.metadata['id']
        ) for c in ret]

    def _get_docs_sim_retriever(self):
        # just a simple vector search on raw document chunks
        retrieval_query = """
            MATCH (node)
            RETURN node.content AS text, 1.0 AS score, {
                name: node.name,
                metadata: node.metadata,
                id: node.id
            } AS metadata
        """
        return CustomNeo4jVector.from_existing_index(
            embedding=self.embedder,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=self.guidelines_vector.index_name,
            retrieval_query=retrieval_query
        )