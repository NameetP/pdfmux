# A  Additional prompt data details

## A.1  Labeler-written prompts

We first give slightly more details on our prompt boostrapping process. As previously mentioned, for the majority of the project, we obtained prompts directly from external users of the instruct beta models in the OpenAI API. However, this strategy only works once you have a model that accepts instruction-like prompts. In order to train the very first such model, we asked contractors to write prompts themselves. We asked labelers to write three kinds of prompts:

- **Plain:** We simply ask the labelers to come up with an arbitrary task, while ensuring diversity of tasks.
- **Few-shot:** We ask the labelers to come up with an instruction, and multiple query/response pairs for that instruction. For example, the instruction could be "Give the sentiment for a tweet," and the queries would be tweets and the responses either "Positive" or "Negative." We can then format these as few-shot prompts like those in Brown et al. (2020). With K query-response pairs, we create K training examples using the other K-1 in the context.
- **User-based:** We had a number of use-cases stated in applications to the OpenAI API. We asked labelers to come up with prompts corresponding to these use cases.

In order to preserve the anonymity of the application information, we had a separate labeler create vague high level tasks based on looking at a list of applications, modifying the task descriptions to eliminate any information that were specific to a given application. This data was used to train the first InstructGPT model via supervised learning, which was deployed in beta in the API in early 2021.

## A.2  API user prompts

For API prompts, we use prompts submitted by users to the aforementioned earlier version of the InstructGPT model on the OpenAI API Playground. Throughout the paper, we only use data from the Playground, rather than customers using our model in production, as it was easier to get informed consent: every time a user switched to an InstructGPT model, an alert message would pop up stating that prompts submitted to these models could be used to train future versions of our models. We also communicated this in a message on the developer Slack channel upon launching the beta of the InstructGPT models. We filter out prompts from the training split containing personally identifiable information (PII).

To ensure a diversity of use cases, we heuristically deduplicate prompts by checking for prompts that share a long common prefix, and limited the number of prompts to roughly 200 per organization. In addition, we create train, validation, and test splits based on organization IDs, so that e.g. the validation set contains different use cases than the training set.

We conceptualized API requests as belonging to one of ten use cases: generation, open QA, closed QA, brainstorming, chat, rewriting, summarization, classification, extraction, or other. Below, we show fictional but realistic prompts from a variety of use cases:

### A.2.1  Illustrative user prompts from InstructGPT distribution

| Use Case | Example |
|---|---|
| brainstorming | List five ideas for how to regain enthusiasm for my career |
| brainstorming | What are some key points I should know when studying Ancient Greece? |
| brainstorming | What are 4 questions a user might have after reading the instruction manual for a trash compactor? {user manual} 1. |

Continued on next page

26
