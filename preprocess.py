import pandas as pd
import re

# Load data
df = pd.read_csv('data/fake_job_postings.csv')

# Step 1 - Clean HTML tags from text
def clean_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'<[^>]+>', '', text)  # remove any HTML tags
    text = re.sub(r'\s+', ' ', text)     # remove extra spaces
    return text.strip()

# Step 2 - Combine all useful columns into one text
def combine_columns(row):
    title = clean_text(row['title'])
    profile = clean_text(row['company_profile'])
    description = clean_text(row['description'])
    requirements = clean_text(row['requirements'])
    return f"{title} {profile} {description} {requirements}"

print("Cleaning and combining columns...")
df['combined_text'] = df.apply(combine_columns, axis=1)

# Step 3 - Keep only what we need
df_clean = df[['combined_text', 'fraudulent']].copy()

# Step 4 - Drop any empty rows
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['combined_text'].str.strip() != '']

# Step 5 - Check the result
print("Clean dataset size:", len(df_clean))
print("Fake count:", df_clean['fraudulent'].sum())
print("Real count:", (df_clean['fraudulent'] == 0).sum())
print()
print("=== SAMPLE COMBINED TEXT (first fake) ===")
first_fake = df_clean[df_clean['fraudulent'] == 1].iloc[0]
print(first_fake['combined_text'][:400])

# Step 6 - Save cleaned data
df_clean.to_csv('data/cleaned_jobs.csv', index=False)
print("\nSaved to data/cleaned_jobs.csv")