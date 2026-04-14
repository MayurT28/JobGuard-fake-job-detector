import pandas as pd

df = pd.read_csv('data/fake_job_postings.csv')

# How big is the data?
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print()

# What are the column names?
print("Columns:")
print(df.columns.tolist())
print()

# How many fake vs real?
print("Fake vs Real count:")
print(df['fraudulent'].value_counts())
print()

# Show one fake posting
print("=== ONE FAKE POSTING ===")
fake = df[df['fraudulent'] == 1].iloc[0]
print("Title:", fake['title'])
print("Description:", fake['description'])

# How much data is missing?
print("=== MISSING DATA ===")
print(df.isnull().sum())
print()

# Which columns do fake postings miss more?
print("=== MISSING IN FAKE POSTINGS ===")
fake_df = df[df['fraudulent'] == 1]
real_df = df[df['fraudulent'] == 0]
print("Fake - missing salary_range:", fake_df['salary_range'].isnull().sum(), "out of", len(fake_df))
print("Real - missing salary_range:", real_df['salary_range'].isnull().sum(), "out of", len(real_df))
print()

# Read 3 fake postings manually
print("=== 3 FAKE POSTINGS - READ CAREFULLY ===")
for i, row in fake_df.head(3).iterrows():
    print(f"\n--- Fake #{i} ---")
    print("Title:", row['title'])
    print("Salary:", row['salary_range'])
    print("Description:", str(row['description'])[:300])
    print()