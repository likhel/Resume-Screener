import pandas as pd

# Load your dataset
df = pd.read_csv("data/job_description/job_title_des.csv")

# Drop irrelevant or empty rows
df = df.dropna(subset=['Job Title', 'Job Description'])

# Filter out rows that don't look like actual job titles
df = df[df['Job Title'].str.len() > 3]

# Optional: Remove duplicates
df = df.drop_duplicates(subset=['Job Title', 'Job Description'])

# Reset index
df.reset_index(drop=True, inplace=True)

print(f" Cleaned dataset: {len(df)} usable job descriptions.")
print(df.head())

df.to_csv("data/job_description/job_data_cleaned.csv", index=False)
