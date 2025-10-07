import spacy
import pandas as pd
import re

# Allowed entity types
ALLOWED_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "LOC",
    "NORP", "WORK_OF_ART", "EVENT"
}

# Generic nouns that we resolve to the nearest previous ORG (heuristic)
GENERIC_NOUNS = {
    "company", "firm", "corporation", "startup",
    "organization", "org", "bank"
}

# Blacklist of unhelpful/reporting verbs
REL_BLACKLIST = {
    "be", "have", "do", "get", "make", "seem",
    "appear", "include", "contain", "use",
    "tell", "say", "report", "announce"
}

# Relation normalization mapping
RELATION_MAP = {
    "found": "founded",
    "founder": "founded",
    "establish": "founded",
    "create": "founded",

    "acquire": "acquired",
    "buy": "acquired",
    "purchase": "acquired",
    "merge": "merged_with",

    "unveil": "unveiled",
    "launch": "launched",
    "release": "released",

    "appoint": "appointed",
    "elect": "elected",
    "name": "appointed",
    "visit": "visited",
    "encourage": "encouraged",
    "urge": "urged",
    "sign": "signed",
    "build": "built",
    "own": "owns",
    "transfer": "transferred",
    "recall": "recalled",
    "sue": "sued"
}

class RuleBasedTripleExtractor:
    def __init__(self, model="en_core_web_sm"):
        """Load spaCy model."""
        self.nlp = spacy.load(model)

    def _token_to_ent_map(self, doc):
        mapping = {}
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                mapping[i] = ent
        return mapping

    def _expand_conjoined_entities(self, ent, token_ent_map, doc):
        results = [ent]
        head_idx = ent.root.i
        for tok in doc:
            if tok.dep_ == "conj" and tok.head.i == head_idx:
                ent2 = token_ent_map.get(tok.i)
                if ent2 and ent2 not in results:
                    results.append(ent2)
        return results

    def _resolve_generic_np(self, np_span, doc):
        root_lemma = np_span.root.lemma_.lower()
        if root_lemma not in GENERIC_NOUNS:
            return None
        for ent in reversed(list(doc.ents)):
            if ent.end <= np_span.start and ent.label_ == "ORG":
                return ent
        return None

    def _normalize_relation(self, lemma: str) -> str:
        return RELATION_MAP.get(lemma, lemma)

    def extract_triples(self, text):
        doc = self.nlp(text)
        token_ent_map = self._token_to_ent_map(doc)
        triples = set()

        for sent in doc.sents:
            # Resolve generic mentions
            generic_resolution = {}
            for np in sent.noun_chunks:
                if np.root.lemma_.lower() in GENERIC_NOUNS:
                    resolved = self._resolve_generic_np(np, doc)
                    if resolved:
                        generic_resolution[np.root.i] = resolved

            for token in sent:
                if token.pos_ not in {"VERB", "AUX"}:
                    continue

                # Subjects
                subj_tokens = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                # Objects
                obj_tokens = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "dative")]
                # Agents
                agent_tokens = []
                for child in token.rights:
                    if child.dep_ == "agent":
                        for gc in child.rights:
                            if gc.dep_ == "pobj":
                                agent_tokens.append(gc)

                subj_ents, obj_ents, agent_ents = [], [], []

                for st in subj_tokens:
                    ent = token_ent_map.get(st.i) or generic_resolution.get(st.i)
                    if ent:
                        subj_ents.extend(self._expand_conjoined_entities(ent, token_ent_map, doc))

                for ot in obj_tokens:
                    ent = token_ent_map.get(ot.i) or generic_resolution.get(ot.i)
                    if ent:
                        obj_ents.extend(self._expand_conjoined_entities(ent, token_ent_map, doc))

                for at in agent_tokens:
                    ent = token_ent_map.get(at.i)
                    if ent:
                        agent_ents.extend(self._expand_conjoined_entities(ent, token_ent_map, doc))

                lemma = token.lemma_.lower()
                if lemma in REL_BLACKLIST:
                    continue
                relation = self._normalize_relation(lemma)

                # Passive
                if subj_tokens and any(st.dep_ == "nsubjpass" for st in subj_tokens) and agent_ents:
                    for a in agent_ents:
                        if a.label_ not in ALLOWED_ENTITY_TYPES:
                            continue
                        for p in subj_ents:
                            if p.label_ not in ALLOWED_ENTITY_TYPES:
                                continue
                            triples.add((a.text, relation, p.text))
                    continue

                # Active
                if subj_ents and obj_ents:
                    for s in subj_ents:
                        if s.label_ not in ALLOWED_ENTITY_TYPES:
                            continue
                        for o in obj_ents:
                            if o.label_ not in ALLOWED_ENTITY_TYPES:
                                continue
                            triples.add((s.text, relation, o.text))
                    continue

        return sorted(triples)

# Preprocessing Function
def preprocess_bbc(df):
    df['title'] = df['title'].fillna('')
    df['content'] = df['content'].fillna('')
    df['category'] = df['category'].fillna('Unknown')
    df['filename'] = df['filename'].fillna('Unknown')

    # Remove extra whitespace
    df['title'] = df['title'].str.strip()
    df['content'] = df['content'].str.strip()

    # Clean multiple spaces
    df['title'] = df['title'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['content'] = df['content'].apply(lambda x: re.sub(r'\s+', ' ', x))

    return df

def normalize_entity(text, nlp):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Run on BBC Dataset
if __name__ == "__main__":
    # Load BBC dataset
    df = pd.read_csv("bbc-news-data.csv", sep="\t", engine="python")

    df = preprocess_bbc(df)

    # Extract triples
    extractor = RuleBasedTripleExtractor(model="en_core_web_sm")
    all_triples = []

    for idx, row in df.iterrows():
        text = f"{row['title']} {row['content']}"
        triples = extractor.extract_triples(text)
        for s, r, o in triples:
            all_triples.append({
                "Subject": normalize_entity(s, extractor.nlp),
                "Relation": r,  # already normalized/lemmatized
                "Object": normalize_entity(o, extractor.nlp),
                "Category": row["category"],
                "Filename": row["filename"]
            })

        if len(all_triples) >= 1000:
            break

    triples_df = pd.DataFrame(all_triples)
    triples_df.to_csv("triples.csv", index=False)
    print(f"Extracted {len(triples_df)} triples → saved to triples.csv")