import pandas as pd
import os

print(f"Current directory: {os.getcwd()}")

# Read the CSV
df = pd.read_csv('datasets/Images/results.csv', sep='|')
print(f"✅ Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Column names mein spaces hain, is liye strip karein
df.columns = df.columns.str.strip()

# Group by image_name and combine comments
def combine_comments(group):
    comments = group['comment'].tolist()
    return ' '.join(comments[:3])

image_descriptions = df.groupby('image_name').apply(combine_comments).reset_index()
image_descriptions.columns = ['image_name', 'description']

# Create the format: image_path|caption|category
image_descriptions['image_path'] = 'datasets/Images/' + image_descriptions['image_name']
image_descriptions['caption'] = image_descriptions['description']
image_descriptions['category'] = 'image'

# Save in pipe-separated format
os.makedirs('data/raw', exist_ok=True)
image_descriptions[['image_path', 'caption', 'category']].to_csv(
    'data/raw/dataset.txt', sep='|', index=False, header=False
)

print(f'✅ Dataset created! Total: {len(image_descriptions)} images')
print('\nFirst 3 samples:')
for i in range(min(3, len(image_descriptions))):
    print(f"{i+1}. {image_descriptions.iloc[i]['image_name']}")
    print(f"   {image_descriptions.iloc[i]['description'][:100]}...")