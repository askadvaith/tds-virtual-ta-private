# import os
# import json
# import numpy as np
# import re
# from pathlib import Path
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# import google.generativeai as genai

# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# class DocumentChunker:
#     def __init__(self, chunk_size: int = 512, overlap: int = 50):
#         self.chunk_size = chunk_size
#         self.overlap = overlap

#     def chunk_markdown(self, content: str, file_path: str) -> List[Dict[str, Any]]:
#         chunks = []
#         sections = re.split(r'\n(?=#{1,6}\s)', content)
#         for section in sections:
#             if not section.strip():
#                 continue
#             header_match = re.match(r'^(#{1,6})\s+(.+?)(?:\n|$)', section)
#             header = header_match.group(2) if header_match else ""
#             section_content = section[header_match.end():].strip() if header_match else section
#             paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
#             current_chunk = ""
#             for para in paragraphs:
#                 if len(current_chunk) + len(para) <= self.chunk_size:
#                     current_chunk += para + "\n\n"
#                 else:
#                     if current_chunk:
#                         chunks.append({
#                             'content': current_chunk.strip(),
#                             'source': file_path,
#                             'type': 'course_material',
#                             'header': header,
#                             'metadata': {'file_path': file_path}
#                         })
#                     current_chunk = para + "\n\n"
#             if current_chunk:
#                 chunks.append({
#                     'content': current_chunk.strip(),
#                     'source': file_path,
#                     'type': 'course_material',
#                     'header': header,
#                     'metadata': {'file_path': file_path}
#                 })
#         return chunks

#     def chunk_forum_post(self, post: Dict[str, Any]) -> List[Dict[str, Any]]:
#         content = post['content']
#         if len(content) <= self.chunk_size:
#             return [{
#                 'content': content,
#                 'source': post['post_url'],
#                 'type': 'forum_post',
#                 'metadata': {
#                     'topic_id': post['topic_id'],
#                     'post_number': post['post_number'],
#                     'username': post['username'],
#                     'created_at': post['created_at'],
#                     'post_url': post['post_url'],
#                     'has_images': len(post.get('images_base64', [])) > 0
#                 }
#             }]
#         chunks = []
#         sentences = content.split('. ')
#         current_chunk = ""
#         for sentence in sentences:
#             if len(current_chunk) + len(sentence) <= self.chunk_size:
#                 current_chunk += sentence + ". "
#             else:
#                 if current_chunk:
#                     chunks.append({
#                         'content': current_chunk.strip(),
#                         'source': post['post_url'],
#                         'type': 'forum_post',
#                         'metadata': {
#                             'topic_id': post['topic_id'],
#                             'post_number': post['post_number'],
#                             'username': post['username'],
#                             'created_at': post['created_at'],
#                             'post_url': post['post_url'],
#                             'has_images': len(post.get('images_base64', [])) > 0
#                         }
#                     })
#                 current_chunk = sentence + ". "
#         if current_chunk:
#             chunks.append({
#                 'content': current_chunk.strip(),
#                 'source': post['post_url'],
#                 'type': 'forum_post',
#                 'metadata': {
#                     'topic_id': post['topic_id'],
#                     'post_number': post['post_number'],
#                     'username': post['username'],
#                     'created_at': post['created_at'],
#                     'post_url': post['post_url'],
#                     'has_images': len(post.get('images_base64', [])) > 0
#                 }
#             })
#         return chunks

# class EmbeddingManager:
#     def __init__(self):
#         pass  # No client initialization needed for google.generativeai

#     def generate_embeddings(self, texts: List[str]) -> np.ndarray:
#         embeddings = []
#         for text in texts:
#             try:
#                 response = genai.embed_content(
#                     model="models/gemini-embedding-exp-03-07",
#                     content=text
#                 )
#                 if 'embedding' not in response or response['embedding'] is None:
#                     print(f"Warning: No embedding returned for text: {text[:60]}...")
#                 else:
#                     print(f"Embedding generated for text: {text[:60]}...")
#             except Exception as e:
#                 print(f"Error generating embedding for text: {text[:60]}... Exception: {e}")
#                 response = {'embedding': None}
#             embeddings.append(response['embedding'])
#         return np.array(embeddings)

#     def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], embeddings_file: str, metadata_file: str):
#         np.savez_compressed(embeddings_file, embeddings=embeddings)
#         with open(metadata_file, 'w', encoding='utf-8') as f:
#             json.dump(chunks, f, indent=2, ensure_ascii=False)

# def generate_and_save_embeddings(data_dir="data", embeddings_dir="embeddings"):
#     chunker = DocumentChunker()
#     embedding_manager = EmbeddingManager()

#     data_dir = Path(data_dir)
#     embeddings_dir = Path(embeddings_dir)
#     embeddings_dir.mkdir(exist_ok=True)

#     all_chunks = []
#     course_content_dir = data_dir / "tools-in-data-science-public"
#     if course_content_dir.exists():
#         for md_file in course_content_dir.rglob("*.md"):
#             with open(md_file, 'r', encoding='utf-8') as f:
#                 content = f.read()
#             chunks = chunker.chunk_markdown(content, str(md_file))
#             all_chunks.extend(chunks)

#     forum_posts_file = data_dir / "tds_discourse_posts.json"
#     if forum_posts_file.exists():
#         with open(forum_posts_file, 'r', encoding='utf-8') as f:
#             posts = json.load(f)
#         for post in posts:
#             chunks = chunker.chunk_forum_post(post)
#             all_chunks.extend(chunks)

#     texts = [chunk['content'] for chunk in all_chunks]
#     embeddings = embedding_manager.generate_embeddings(texts)
#     embeddings_file = embeddings_dir / "tds_embeddings.npz"
#     metadata_file = embeddings_dir / "tds_metadata.json"

#     embedding_manager.save_embeddings(embeddings, all_chunks, str(embeddings_file), str(metadata_file))

# if __name__ == "__main__":
#     generate_and_save_embeddings()

# import os
# import json
# import numpy as np
# import re
# from pathlib import Path
# from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer

# class DocumentChunker:
#     def __init__(self, chunk_size: int = 512, overlap: int = 50):
#         self.chunk_size = chunk_size
#         self.overlap = overlap

#     def chunk_markdown(self, content: str, file_path: str) -> List[Dict[str, Any]]:
#         chunks = []
#         sections = re.split(r'\n(?=#{1,6}\s)', content)
#         for section in sections:
#             if not section.strip():
#                 continue
#             header_match = re.match(r'^(#{1,6})\s+(.+?)(?:\n|$)', section)
#             header = header_match.group(2) if header_match else ""
#             section_content = section[header_match.end():].strip() if header_match else section
#             paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
#             current_chunk = ""
#             for para in paragraphs:
#                 if len(current_chunk) + len(para) <= self.chunk_size:
#                     current_chunk += para + "\n\n"
#                 else:
#                     if current_chunk:
#                         chunks.append({
#                             'content': current_chunk.strip(),
#                             'source': file_path,
#                             'type': 'course_material',
#                             'header': header,
#                             'metadata': {'file_path': file_path}
#                         })
#                     current_chunk = para + "\n\n"
#             if current_chunk:
#                 chunks.append({
#                     'content': current_chunk.strip(),
#                     'source': file_path,
#                     'type': 'course_material',
#                     'header': header,
#                     'metadata': {'file_path': file_path}
#                 })
#         return chunks

#     def chunk_forum_post(self, post: Dict[str, Any]) -> List[Dict[str, Any]]:
#         content = post['content']
#         if len(content) <= self.chunk_size:
#             return [{
#                 'content': content,
#                 'source': post['post_url'],
#                 'type': 'forum_post',
#                 'metadata': {
#                     'topic_id': post['topic_id'],
#                     'post_number': post['post_number'],
#                     'username': post['username'],
#                     'created_at': post['created_at'],
#                     'post_url': post['post_url'],
#                     'has_images': len(post.get('images_base64', [])) > 0
#                 }
#             }]
#         chunks = []
#         sentences = content.split('. ')
#         current_chunk = ""
#         for sentence in sentences:
#             if len(current_chunk) + len(sentence) <= self.chunk_size:
#                 current_chunk += sentence + ". "
#             else:
#                 if current_chunk:
#                     chunks.append({
#                         'content': current_chunk.strip(),
#                         'source': post['post_url'],
#                         'type': 'forum_post',
#                         'metadata': {
#                             'topic_id': post['topic_id'],
#                             'post_number': post['post_number'],
#                             'username': post['username'],
#                             'created_at': post['created_at'],
#                             'post_url': post['post_url'],
#                             'has_images': len(post.get('images_base64', [])) > 0
#                         }
#                     })
#                 current_chunk = sentence + ". "
#         if current_chunk:
#             chunks.append({
#                 'content': current_chunk.strip(),
#                 'source': post['post_url'],
#                 'type': 'forum_post',
#                 'metadata': {
#                     'topic_id': post['topic_id'],
#                     'post_number': post['post_number'],
#                     'username': post['username'],
#                     'created_at': post['created_at'],
#                     'post_url': post['post_url'],
#                     'has_images': len(post.get('images_base64', [])) > 0
#                 }
#             })
#         return chunks

# class EmbeddingManager:
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)

#     def generate_embeddings(self, texts: List[str]) -> np.ndarray:
#         return self.model.encode(texts, show_progress_bar=True)

#     def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], embeddings_file: str, metadata_file: str):
#         np.savez_compressed(embeddings_file, embeddings=embeddings)
#         with open(metadata_file, 'w', encoding='utf-8') as f:
#             json.dump(chunks, f, indent=2, ensure_ascii=False)

# def generate_and_save_embeddings(data_dir="data", embeddings_dir="embeddings"):
#     chunker = DocumentChunker()
#     embedding_manager = EmbeddingManager()

#     data_dir = Path(data_dir)
#     embeddings_dir = Path(embeddings_dir)
#     embeddings_dir.mkdir(exist_ok=True)

#     all_chunks = []
#     course_content_dir = data_dir / "tools-in-data-science-public"
#     if course_content_dir.exists():
#         for md_file in course_content_dir.rglob("*.md"):
#             with open(md_file, 'r', encoding='utf-8') as f:
#                 content = f.read()
#             chunks = chunker.chunk_markdown(content, str(md_file))
#             all_chunks.extend(chunks)

#     forum_posts_file = data_dir / "tds_discourse_posts.json"
#     if forum_posts_file.exists():
#         with open(forum_posts_file, 'r', encoding='utf-8') as f:
#             posts = json.load(f)
#         for post in posts:
#             chunks = chunker.chunk_forum_post(post)
#             all_chunks.extend(chunks)

#     texts = [chunk['content'] for chunk in all_chunks]
#     embeddings = embedding_manager.generate_embeddings(texts)
#     embeddings_file = embeddings_dir / "tds_embeddings.npz"
#     metadata_file = embeddings_dir / "tds_metadata.json"

#     embedding_manager.save_embeddings(embeddings, all_chunks, str(embeddings_file), str(metadata_file))

# if __name__ == "__main__":
#     generate_and_save_embeddings()
import os
import json
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_markdown(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        chunks = []
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        for section in sections:
            if not section.strip():
                continue
            header_match = re.match(r'^(#{1,6})\s+(.+?)(?:\n|$)', section)
            header = header_match.group(2) if header_match else ""
            section_content = section[header_match.end():].strip() if header_match else section
            paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            'content': current_chunk.strip(),
                            'source': file_path,
                            'type': 'course_material',
                            'header': header,
                            'metadata': {'file_path': file_path}
                        })
                    current_chunk = para + "\n\n"
            if current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'source': file_path,
                    'type': 'course_material',
                    'header': header,
                    'metadata': {'file_path': file_path}
                })
        return chunks

    def chunk_forum_post(self, post: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = post['content']
        if len(content) <= self.chunk_size:
            return [{
                'content': content,
                'source': post['post_url'],
                'type': 'forum_post',
                'metadata': {
                    'topic_id': post['topic_id'],
                    'post_number': post['post_number'],
                    'username': post['username'],
                    'created_at': post['created_at'],
                    'post_url': post['post_url'],
                    'has_images': len(post.get('images_base64', [])) > 0
                }
            }]
        chunks = []
        sentences = content.split('. ')
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'source': post['post_url'],
                        'type': 'forum_post',
                        'metadata': {
                            'topic_id': post['topic_id'],
                            'post_number': post['post_number'],
                            'username': post['username'],
                            'created_at': post['created_at'],
                            'post_url': post['post_url'],
                            'has_images': len(post.get('images_base64', [])) > 0
                        }
                    })
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'source': post['post_url'],
                'type': 'forum_post',
                'metadata': {
                    'topic_id': post['topic_id'],
                    'post_number': post['post_number'],
                    'username': post['username'],
                    'created_at': post['created_at'],
                    'post_url': post['post_url'],
                    'has_images': len(post.get('images_base64', [])) > 0
                }
            })
        return chunks

class EmbeddingManager:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.api_url = "https://aipipe.org/openai/v1/embeddings"
        self.headers = {
            "Authorization": AIPIPE_TOKEN,
            "Content-Type": "application/json"
        }

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model_name,
                "input": text
            }
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                # OpenAI-compatible embedding API returns a list under 'data'
                print(f"Requesting embedding for text: {text[:60]}...")
                print(f"Payload: {payload}")
                print(f"Response status code: {response.status_code}")
                print(f"Response JSON: {data}")
                if "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
                    print("Embedding found, appending to list.")
                    embeddings.append(data["data"][0]["embedding"])
                else:
                    print(f"Warning: No embedding returned for text: {text[:60]}...")
                    embeddings.append([0.0])  # fallback
            except Exception as e:
                print(f"Error generating embedding for text: {text[:60]}... Exception: {e}")
                embeddings.append([0.0])  # fallback
        return np.array(embeddings)

    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], embeddings_file: str, metadata_file: str):
        np.savez_compressed(embeddings_file, embeddings=embeddings)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

def generate_and_save_embeddings(data_dir="data", embeddings_dir="embeddings"):
    print("Starting embedding generation...")
    chunker = DocumentChunker()
    embedding_manager = EmbeddingManager()

    data_dir = Path(data_dir)
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(exist_ok=True)

    all_chunks = []
    course_content_dir = data_dir / "tools-in-data-science-public"
    if course_content_dir.exists():
        for md_file in course_content_dir.rglob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = chunker.chunk_markdown(content, str(md_file))
            all_chunks.extend(chunks)

    forum_posts_file = data_dir / "tds_discourse_posts.json"
    if forum_posts_file.exists():
        with open(forum_posts_file, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        for post in posts:
            chunks = chunker.chunk_forum_post(post)
            all_chunks.extend(chunks)

    print(f"Total chunks to embed: {len(all_chunks)}")
    texts = [chunk['content'] for chunk in all_chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    embeddings_file = embeddings_dir / "tds_embeddings.npz"
    metadata_file = embeddings_dir / "tds_metadata.json"

    embedding_manager.save_embeddings(embeddings, all_chunks, str(embeddings_file), str(metadata_file))
    
    



if __name__ == "__main__":
    generate_and_save_embeddings()