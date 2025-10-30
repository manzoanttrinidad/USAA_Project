import pandas as pd

# Load the data
url = pd.read_csv("extracted_urls.csv")
terms_df = pd.read_csv("fruad_terms.csv")

# Assume terms are in a column named "terms" (adjust if different)
terms = terms_df['terms'].dropna().tolist()

# Case-insensitive filtering function
def contains_term(url):
    url_lower = str(url).lower()
    return any(term.lower() in url_lower for term in terms)

# Filter
filtered_df = url[url['extracted'].apply(contains_term)]

# Save results
filtered_df.to_csv("filtered_url.csv", index=False)


