from langchain_core.prompts import ChatPromptTemplate
from helpers.constants import CONVLLM

# ------------------------------------- #

considerations_template = \
"""Minimize use of jargon and specialized terms.

Given the user's query, our database search returned the following list of treatment considerations:

<considerations>
{formatted_considerations}
</considerations>

<task_instructions>
Answer the user's query using the considerations above as a guide. Tailor your response to address the user's query and concerns, but do not return the considerations verbatim.

Be organic and natural in your response. For each consideration:
- determine if it's relevant to the user. only if it is relevant, then include it in the discussion.
- briefly summarize the consideration as it relates to the user's concerns
- in a subsection for readability, briefly list the treatment options for that consideration, but abbreviate or combine similar treatments where able, without losing specificity
- you must include ALL treatment options for each consideration, even if it's repeated from another consideration, so that the user knows all of their options when making a decision
- keep your response concise, and do not directly reference or quote the provided background information to the user.

Finally, provide a very brief summary of the considerations as a whole. If there are obvious treatments that are common to multiple considerations, mention them at the end as potential recommendations.

Remember: your current task is to briefly discuss JUST THE TREATMENT CONSIDERATIONS. 
Once you have finished summarizing the considerations, you will be provided with information about specific treatments the user asked about,
so do not discuss the specific treatments in your response for now.
</task_instructions>

Just FYI, here are the specific treatments the user asked about:

<specific_treatments>
{specific_treatments}
</specific_treatments>"""

# --- #

treatments_template = \
"""Minimize use of jargon and specialized terms.

Given the user's query, our database search returned the following list of treatments:

<specific_treatments>
{specific_treatments}
</specific_treatments>

Here are some detailed information about the treatments, using guideline exerpts from the CUA, AUA, and EAU guidelines on BPH:

<guidelines>
{formatted_treatments}
</guidelines>

<task_instructions>
Based on the user's treatment priorities, provide a targeted summary of the treatments the user asked about.

For each treatment:
- provide a brief targeted summary
- if applicable: briefly highlight relevance to the user's treatment priorities
- provide a VERY brief comparison of any relevant differences between each guideline for the treatments

Remember: these treatments are not necessarily recommended for the patient (they may or may not be): you are discussing them at the specific request of the user.
Therefore, you should note any potential recommendations or concerns only if pertinent specifically to the user's query.
</task_instructions>"""

# ------------------------------------- #

considerations_prompt = ChatPromptTemplate([
    ("system", considerations_template),
    ('human', '{query}'),
])

treatments_prompt = ChatPromptTemplate([
    ("system", treatments_template),
    ('human', '{query}'),
])

considerations_chain = considerations_prompt | CONVLLM
treatments_chain = treatments_prompt | CONVLLM