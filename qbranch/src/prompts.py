"""Prompts for the QBranch build step, the ablation build step, the reader and the judge."""

BUILD_QBRANCH = """You keep a knowledge graph of a growing document collection. A new passage arrived. Do these steps in this order.

Step 1. Infer questions. Write the questions that a user can ask and that this passage answers (at most 6). Cover each important named entity. Then write the questions that this passage raises but does not answer (at most 3). These are facts about its entities that a later document can give, for example the birthplace of a person who is named but not described.

Step 2. Schema. For each question, name the relation that it asks about. Use a short, generic, reusable relation name in snake_case (examples: directed_by, born_in, located_in, member_of, founded_in_year). Use a relation from the list of known relations when one fits. Give the type of the subject and of the object (examples: person, film, city, country, organization, date, number).

Step 3. Bind facts. For each answered question, give the triple(s) from the passage that answer it. For each unanswered question, give an open triple with object null. {title_rule} Replace pronouns with the entity name.

Known relations: {relations}

Passage title: {title}
Passage: {text}

Return only a JSON object with this form:
{{"questions": [{{"q": "...", "answered": true, "answer": "...", "triples": [{{"s": "...", "s_type": "...", "r": "...", "o": "...", "o_type": "..."}}]}},
               {{"q": "...", "answered": false, "triples": [{{"s": "...", "s_type": "...", "r": "...", "o": null, "o_type": "..."}}]}}]}}"""

# Ablation: the same prompt without the open (unanswered) questions.
BUILD_QBRANCH_NOOPEN = """You keep a knowledge graph of a growing document collection. A new passage arrived. Do these steps in this order.

Step 1. Infer questions. Write the questions that a user can ask and that this passage answers (at most 6). Cover each important named entity.

Step 2. Schema. For each question, name the relation that it asks about. Use a short, generic, reusable relation name in snake_case (examples: directed_by, born_in, located_in, member_of, founded_in_year). Use a relation from the list of known relations when one fits. Give the type of the subject and of the object (examples: person, film, city, country, organization, date, number).

Step 3. Bind facts. For each question, give the triple(s) from the passage that answer it. {title_rule} Replace pronouns with the entity name.

Known relations: {relations}

Passage title: {title}
Passage: {text}

Return only a JSON object with this form:
{{"questions": [{{"q": "...", "answered": true, "answer": "...", "triples": [{{"s": "...", "s_type": "...", "r": "...", "o": "...", "o_type": "..."}}]}}]}}"""

BUILD_OPENIE = """You keep a knowledge graph of a growing document collection. A new passage arrived. Extract all the facts in the passage as triples (subject, relation, object), at most 12. Use a short, generic, reusable relation name in snake_case (examples: directed_by, born_in, located_in, member_of, founded_in_year). Use a relation from the list of known relations when one fits. Give the type of the subject and of the object (examples: person, film, city, country, organization, date, number). {title_rule} Replace pronouns with the entity name.

Known relations: {relations}

Passage title: {title}
Passage: {text}

Return only a JSON object with this form:
{{"triples": [{{"s": "...", "s_type": "...", "r": "...", "o": "...", "o_type": "..."}}]}}"""

# Variant (meeting 2026-11-13): random question-answer pairs from the dataset go into the prompt, like an
# FAQ of the collection (partial knowledge of what users ask). The pairs come from held-out questions only.
BUILD_QBRANCH_QA = BUILD_QBRANCH.replace(
    "Known relations: {relations}",
    "Users of this collection asked questions like these examples (with their answers). The examples are not about "
    "this passage. They show which kinds of facts users need. When the passage allows it, write questions (answered "
    "and open) that ask for the same kinds of facts. Each of your questions asks for one fact.\n{examples}\n\n"
    "Known relations: {relations}")

# Variant for C2 on the news stream: the open questions can also be open outcomes of events.
_BG = "These are facts about its entities that a later document can give, for example the birthplace of a person who is named but not described."
BUILD_QBRANCH_EVENTS = BUILD_QBRANCH.replace(
    _BG,
    "These are facts that a later document can give. They can be background facts about its entities (for example "
    "the birthplace of a person who is named but not described). They can also be open outcomes of events or states "
    "that are not decided when the passage is written (for example the verdict of a trial, the result of a vote, "
    "the team that a player will sign with, the date of a release that is not fixed).")
assert BUILD_QBRANCH_QA != BUILD_QBRANCH and BUILD_QBRANCH_EVENTS != BUILD_QBRANCH

TITLE_RULE_WIKI = "The passage title names its main entity: always call the main entity by its title, without the text in parentheses. Call other entities by their full names as the passage writes them."

TITLE_RULE_NEWS = ("The passage is part of a news article. The title is the headline, the source and the date of the article, not an entity. "
                   "Call each entity by its full name (for example Sam Altman, not Altman). Use the date of the article to resolve relative dates such as last week.")

READER = """Answer the question with the evidence below. The evidence can contain facts from a knowledge graph and source passages. Some evidence is not relevant. Questions can need two or more steps (for example, first find a person, then find a fact about that person).

{evidence}

Question: {question}

Think step by step in at most 4 short sentences. Then write the final line in this form:
Answer: <short answer, a few words, no full sentence>
If the evidence does not contain the answer, give your best guess."""

PLAN = """Break the question into a chain of simple sub-questions. Each sub-question asks for one fact. A later sub-question can use the answer of an earlier sub-question: write #1 for the answer of sub-question 1, #2 for the answer of sub-question 2. Use as few sub-questions as possible (at most 4). Keep the names from the question exactly as they are written.

Example. Question: When was the spouse of the director of Jaws born?
{{"steps": ["Who is the director of Jaws?", "Who is the spouse of #1?", "When was #2 born?"]}}

Question: {question}

Return only a JSON object: {{"steps": ["...", "..."]}}"""

CLOSED_BOOK = """Answer the question from your own knowledge.

Question: {question}

Think step by step in at most 4 short sentences. Then write the final line in this form:
Answer: <short answer, a few words, no full sentence>"""

JUDGE = """You check an answer to a question against the gold answer. The answer is correct when it names the same entity, value or fact as the gold answer, even with different words, more detail or a different format. It is not correct when it names something different, or when it is only partly right.

Question: {question}
Gold answer: {gold}
Answer: {pred}

Reply with one word: correct or incorrect."""
