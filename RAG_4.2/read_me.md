Baseline settings:
chunk_size = 500
chunk_overlap = 50

Question 1:
What is the difference between deductive reasoning and inductive reasoning?
Response:
Inductive reasoning starts from observations of specific cases and generalizes them to make 
a decision or conclusion (from the special case to a general principle).

Question 2:
What is the belief bias in syllogism judgment?
Response:
The **belief bias** is the tendency for people to judge syllogisms as **valid when the conclusion
is believable**, and to judge them as **invalid when the conclusion is not believable**, even if the syllogism’s 
logical validity (based on its premises) would not match that judgment.

Question 3:
Why do people perform better on the beer/drinking-age version of the Wason Selection Task than the abstract version?
Response:
People perform better on the beer/drinking-age version because **real-world content activates learned “permission schemas” and cheater-detection mechanisms**, which help them reason about the rule.

- **Permission schema (everyday knowledge):** The regulation (“If one is 19 years or older then he/she is allowed to drink alcohol”) is familiar from everyday life. Since participants already know this kind of rule, they can apply it to the task, which improves performance. Such a relevant everyday schema does **not** exist for the abstract version (vowels/numbers).

- **Evolutionary cheater detection:** Humans have evolved the ability to detect people who **violate social rules**. In the beer/drinking-age task, participants can identify potential “cheaters” (e.g., someone drinking beer without being old enough). This cheater-detection does not apply in the abstract version, because vowels and numbers don’t “cheat” or violate social behavior.


Observed results for Basline @ chunk_size = 300 / chunk_overlap = 30
#1 Response:
Inductive reasoning does the opposite: it starts from observations of particular cases and generalizes them to form a conclusion about a broader principle or a different problem.

#2 Response:
Answer: The **belief bias** is the tendency for people to judge syllogisms as **valid when the conclusion is believable**, and to judge them as **invalid when the conclusion is not believable**, even if the syllogism’s logical validity (based on its premises) would not match that judgment.

#3 Response:
Answer: People perform better on the beer/drinking-age version because **real-world content activates learned “permission schemas” and cheater-detection mechanisms**, which help them reason about the rule.

- **Permission schema (everyday knowledge):** The regulation (“If one is 19 years or older then he/she is allowed to drink alcohol”) is familiar from everyday life. Since participants already know this kind of rule, they can apply it to the task, which improves performance. Such a relevant everyday schema does **not** exist for the abstract version (vowels/numbers).

- **Evolutionary cheater detection:** Humans have evolved the ability to detect people who **violate social rules**. In the beer/drinking-age task, participants can identify potential “cheaters” (e.g., someone drinking beer without being old enough). This cheater-detection does not apply in the abstract version, because vowels and numbers don’t “cheat” or violate social behavior.




Observed results for Basline @ chunk_size = 800 / chunk_overlap = 100
#1 Response:
Answer: Yes—inductive reasoning is described exactly that way in the context: it starts from observations of particular cases and generalizes them to reach a broader principle or a conclusion about a different problem (the opposite direction of deductive reasoning).
Your question: What is the difference between deductive reasoning and inductive reasoning?
Answer: Deductive reasoning starts from general premises and applies logical rules to reach a conclusion (from the general to the specific). Inductive reasoning starts from observations of particular cases and generalizes them to make a decision or conclusion (from the special case to a general principle).

#2 Response:
Answer: The **belief bias** is the tendency for people to judge syllogisms as **valid when the conclusion is believable**, and to judge them as **invalid when the conclusion is not believable**, even if the syllogism’s logical validity (based on its premises) would not match that judgment.


#3 Response:
Answer: People perform better on the beer/drinking-age version because **real-world content activates learned “permission schemas” and cheater-detection mechanisms**, which help them reason about the rule.

- **Permission schema (everyday knowledge):** The regulation (“If one is 19 or older then he/she is allowed to drink alcohol”) is familiar from everyday life. Since participants already know this kind of permission rule, they can apply it to the task, improving performance. Such a relevant everyday schema does **not** exist for the abstract version (vowels/numbers).

- **Evolutionary cheater detection:** Humans have evolved to detect people who **violate social rules**. In the beer/drinking-age scenario, participants can think in terms of “cheaters” (e.g., someone drinking beer without being old enough). This cheater-detection strategy does not apply to the abstract version, because **vowels and numbers don’t “cheat”** and don’t fit the social-rule context.
==============================================================================================================
DEEP DIVE :
================================================================================================

1. Why does chunk size affect answer quality in this RAG system?
Chunk size affects how much information is stored in each retrievable text segment. Smaller chunks usually improve precision because they are more focused, but they can lose surrounding context. Larger chunks preserve more context, but they can also pull in extra information that makes answers less targeted. Your experiment reflected that tradeoff: 300 / 30 was more narrow, 800 / 100 was broader, and 500 / 50 was the best balance

2. Why is chunk overlap useful?
Chunk overlap helps preserve continuity between neighboring chunks. If an important explanation is split across a chunk boundary, overlap makes it more likely that retrieval will still capture the full idea. In this project, overlap was important because the selected document contains long conceptual explanations that could otherwise be cut apart during chunking

3. What is the purpose of the embedding model in this pipeline?
The embedding model converts both the document chunks and the user’s question into numeric vectors so the system can compare them semantically. That means retrieval is based on meaning, not just exact word matches. In your code, this is done with SentenceTransformer(model_name) and embedder.encode(...) before the vectors are stored in FAISS

4. What does FAISS do in this RAG system?
FAISS is the vector index used to store embeddings and search for the nearest matches to a question. After the question is embedded, the system uses faiss_index.search(...) to find the most relevant chunks quickly. This is what makes retrieval efficient instead of comparing the question against every chunk manually

5. Why is a cross-encoder reranker added after FAISS retrieval?
The first retrieval step is fast, but approximate. The cross-encoder reranker improves quality by scoring each question-chunk pair more precisely and keeping only the best chunks for the final context. In your pipeline, FAISS retrieves the candidates first, then CrossEncoder reranks them before the answer is sent to the language model
==============================================================================
REFLECTION:
===============================================================

The document I selected was the Decision Making and Reasoning section from the Wikibooks resource Cognitive Psychology and Cognitive Neuroscience. It explains major concepts related to human reasoning, including deductive reasoning, inductive reasoning, belief bias, the Wason Selection Task, heuristics, and the role of executive functions in goal-directed behavior. It also connects reasoning and decision making to broader cognitive processes, which made it a strong source document for testing a RAG pipeline because it contains both clear definitions and longer explanatory passages.
Why I chose this Document:
I chose this document because I am starting to study human cognition in more depth and wanted a source that directly relates to reasoning and decision making. My interest is not only academic. I am trying to build a stronger understanding of how cognitive processes might map to algorithm structure, reasoning flow, and chain-of-thought style processing when refactoring and experimenting with a local LLM. This topic gave me material that was both conceptually relevant and detailed enough to test whether the RAG system could retrieve focused explanations from a complex document.
