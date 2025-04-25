# Vocalized-Arabic-Parallel-Dataset
The Vocalized Arabic–English Parallel Corpus is a high-quality linguistic resource designed to support advanced research in machine translation and natural language processing. It contains 584,284 aligned sentence pairs in Arabic and English, carefully filtered to ensure semantic and linguistic accuracy.

What sets this corpus apart is that it was constructed by measuring semantic similarity between Arabic and English sentence pairs using a multilingual Sentence-BERT (SBERT) transformer model. Each sentence pair was evaluated by the model, and only those with a semantic similarity score between 0.70 and 0.99 were retained, ensuring high semantic alignment across languages.

The Arabic side of the corpus is available in three versions:

**- Nonvocalized Arabic – ** Standard Arabic text without diacritical marks, representing the common form used in most modern writing.

** - Fully Vocalized Arabic – ** Contains all diacritical marks (harakāt), offering complete phonological and grammatical detail.

** - Partially Vocalized Arabic – ** Includes all diacritics except the final short vowel mark (ʾiʿrāb), which indicates the grammatical function of a word in a sentence. Since this final mark does not affect the word’s core meaning, its omission allows for a balance between phonological information and syntactic neutrality.

By combining high-quality semantic alignment with detailed vocalization variants, this corpus offers a valuable and flexible resource for developing and evaluating Arabic-English translation systems, especially those focused on disambiguation, morphological analysis, and context-aware translation.
