import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv("newtriples.csv")

lemmatizer = WordNetLemmatizer()

# Step 1: Clean relation text
def clean_relation(rel):
    if pd.isna(rel):
        return None
    rel = rel.strip().lower()
    # Remove punctuation-only relations
    if re.fullmatch(r"[,.;:]+", rel):
        return None
    # Lemmatize verbs (win -> win, signed -> sign, etc.)
    rel = " ".join([lemmatizer.lemmatize(w, 'v') for w in rel.split()])
    return rel

# Step 2: Clean subject/object
def clean_entity(ent):
    if pd.isna(ent):
        return None
    ent = ent.strip()
    # Drop placeholders like "the", "a", single letters
    if len(ent) <= 2 or ent.lower() in {"the", "a", "an", "it"}:
        return None
    return ent

# Apply cleaning
df["Relation"] = df["Relation"].apply(clean_relation)
df["Subject"] = df["Subject"].apply(clean_entity)
df["Object"] = df["Object"].apply(clean_entity)

# Step 3: Drop rows with missing or weak triples
df = df.dropna(subset=["Subject", "Relation", "Object"])

# Step 4: Remove duplicates
df = df.drop_duplicates()

# Step 5: Filter vague relations
BAD_RELATIONS = {"have", "be", "say", "tell", ","}
df = df[~df["Relation"].isin(BAD_RELATIONS)]

# Step 6: Heuristic fixes
def fix_relation(row):
    rel = row["Relation"]
    subj, obj = row["Subject"], row["Object"]

    # Example fixes
    if rel == "shortstop":
        return "play shortstop for"
    if rel == "shadow":
        return "appointed"
    if rel == "back":
        return "support"
    if rel == "jump":
        return "increase"
    return rel

df["Relation"] = df.apply(fix_relation, axis=1)

# Reset index
df = df.reset_index(drop=True)

# Save cleaned triples
df.to_csv("triples_cleaned.csv", index=False)

print("Cleaned triples saved -> triples_cleaned.csv")
print(f"Final count: {len(df)} triples")
