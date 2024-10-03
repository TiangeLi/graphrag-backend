from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .constants import CONV_LLM

template = """Given the query, use the provided context to create an accurate and succinct response.

Follow these instructions:
1. Carefully read the query and understand the user's goal; your response structure should be logical, easy to follow, and aimed at fully answering the user's query.
2. Carefully read the context and understand the information provided.
3. Use the context to inform your response, ensuring that your response is accurate and comprehensive.
4. Remember that the context does not have any organizational structure, so you must extract information systematically into your planned response structure. 
5. Your plan should include ALL the entities that are needed to answer the query, but you should NOT exclude any.

<context>
{context}
</context>"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{query}"),
])

#prompt = ChatPromptTemplate([
#    ('system', template),
#    ('human', '{query}'),
#    ('ai', 'scratchpad: Thinking about the user query, here are some subquestions and my thoughts:\n\n{context}'),
#])

response_chain = prompt | CONV_LLM | StrOutputParser()


"""carefully consider each part of it as a subquestion in order to frame how you will answer the question.

First, answer each subquestion and formulate your thoughts in a scratchpad.
Then, formulate your final response in an accurate and comprehensive manner.

When formulating your final response, follow these guidelines:
1. Structure your response in a way that is easy to understand.
2. Do not simply follow the structure of your scratchpad, as your rough thoughts may be disorganized and redundant.
3. Provide a single, comprehensive response to the user's query, incorporating all relevant information from your scratchpad.
4. Remember to focus on the user's goal in asking their question, and provide a response that is tailored to their needs.

Keep your response concise and to the point!"""