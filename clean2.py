import pandas as pd
import re

# CONFIG
INPUT_FILE = "cleaned_triples.csv"   # change to your filename
OUTPUT_FILE = "newtriples.csv"

# Relations that are too vague for presentation
VAGUE_RELATIONS = {"say", "tell", "have", "send", "point"}

# Simple name normalization rules
def normalize_entity(entity: str) -> str:
    if not isinstance(entity, str):
        return ""
    entity = entity.strip()
    # Remove honorifics / prefixes
    entity = re.sub(r"^(Mr|Mrs|Ms|Dr)\.?\s+", "", entity)
    # Collapse multiple spaces
    entity = re.sub(r"\s+", " ", entity)
    return entity

# CLEANING FUNCTION
def clean_triples(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing values
    df = df.dropna(subset=["Subject", "Relation", "Object"])
    
    # Normalize entities
    df["Subject"] = df["Subject"].apply(normalize_entity)
    df["Object"] = df["Object"].apply(normalize_entity)
    
    # Remove self-loops (Subject == Object)
    df = df[df["Subject"].str.lower() != df["Object"].str.lower()]
    
    # Remove vague relations
    df = df[~df["Relation"].str.lower().isin(VAGUE_RELATIONS)]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["Subject", "Relation", "Object"])
    
    # Remove cases where subject/object accidentally merged with relation
    bad_pattern = r"\b(say|tell|have|offer|create)\b"
    df = df[~df["Subject"].str.contains(bad_pattern, case=False, na=False)]
    df = df[~df["Object"].str.contains(bad_pattern, case=False, na=False)]
    
    # Reset index
    df = df.reset_index(drop=True)
    return df

# MAIN
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(INPUT_FILE)
    
    print("Before cleaning:", len(df), "triples")
    
    # Clean
    clean_df = clean_triples(df)
    
    print("After cleaning:", len(clean_df), "triples")
    
    # Save
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned triples saved to {OUTPUT_FILE}")
