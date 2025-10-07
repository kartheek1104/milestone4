import pandas as pd
import re

# Load your triples CSV
df = pd.read_csv("triples.csv")  # Replace with your file path
print("Original triples:", len(df))

# 1. Remove duplicates
df = df.drop_duplicates()
print("After removing duplicates:", len(df))

# 2. Remove self-loops (subject = object)
df = df[df['Subject'].str.strip().str.lower() != df['Object'].str.strip().str.lower()]
print("After removing self-loops:", len(df))

# 3. Clean entity names (remove extra spaces, trailing punctuation)
def clean_entity(e):
    e = e.strip()  # remove leading/trailing spaces
    e = re.sub(r"[-–—]+$", "", e)  # remove trailing dashes
    e = re.sub(r"^\W+|\W+$", "", e)  # remove non-word characters at start/end
    return e

df['Subject'] = df['Subject'].apply(clean_entity)
df['Object'] = df['Object'].apply(clean_entity)

# 4. Normalize relations (basic lowercase)
df['Relation'] = df['Relation'].str.strip().str.lower()

# 5. Optional: Remove triples with empty fields
df = df[(df['Subject'] != "") & (df['Object'] != "") & (df['Relation'] != "")]
print("After cleaning empty fields:", len(df))

# Save cleaned triples
df.to_csv("cleaned_triples.csv", index=False)
print("Cleaned triples saved to cleaned_triples.csv")