import os
from sklearn.datasets import fetch_20newsgroups

data_dir = 'text_dataset'
os.makedirs(data_dir, exist_ok=True)

newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), download_if_missing=True)


for idx, text in enumerate(newsgroups_data.data):
    file_path = os.path.join(data_dir, f'doc_{idx}.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

print(f"Downloaded and saved {len(newsgroups_data.data)} documents to {data_dir}")
