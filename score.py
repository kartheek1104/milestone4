import pandas as pd

# Load your cleaned triples file
df = pd.read_csv("triples_cleaned.csv")

# Function: a triple is Great if it’s meaningful
def is_great(row):
    subj, rel, obj = str(row["Subject"]).strip(), str(row["Relation"]).strip(), str(row["Object"]).strip()
    
    if not subj or not rel or not obj:
        return "Weak"
    if rel in {",", ".", ";", ":"}:
        return "Weak"
    if len(subj) <= 1 or len(obj) <= 1 or len(rel) <= 1:
        return "Weak"
    
    return "Great"

# Apply classification
df["Quality"] = df.apply(is_great, axis=1)

# Calculate percentage
proportions = df["Quality"].value_counts(normalize=True) * 100
proportions = proportions.round(2)

print("Quality distribution (%):")
print(proportions)

# Save labeled file for presentation
df.to_csv("triples_presentation_ready.csv", index=False)
print("\nSaved -> triples_presentation_ready.csv")

