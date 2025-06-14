# # TDS Virtual Teaching Assistant
# # Complete implementation with embedding generation, storage, and API server

# import os
# import json
# import numpy as np
# import base64
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# import re
# from pathlib import Path
# import hashlib
# import requests 

# # Core dependencies
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import markdown
# from PIL import Image
# import io

# # Web framework and utilities
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import google.generativeai as genai
# from werkzeug.exceptions import BadRequest

# # env variables
# from dotenv import load_dotenv
# load_dotenv()

# class DocumentChunker:
#     """Handles chunking of markdown and forum post content"""
    
#     def __init__(self, chunk_size: int = 512, overlap: int = 50):
#         self.chunk_size = chunk_size
#         self.overlap = overlap
    
#     def chunk_markdown(self, content: str, file_path: str) -> List[Dict[str, Any]]:
#         """Chunk markdown content by sections and paragraphs"""
#         chunks = []
        
#         # Split by headers first
#         sections = re.split(r'\n(?=#{1,6}\s)', content)
        
#         for section in sections:
#             if not section.strip():
#                 continue
                
#             # Extract header if present
#             header_match = re.match(r'^(#{1,6})\s+(.+?)(?:\n|$)', section)
#             header = header_match.group(2) if header_match else ""
            
#             # Remove header from content for chunking
#             section_content = section
#             if header_match:
#                 section_content = section[header_match.end():].strip()
            
#             # Split content into paragraphs
#             paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
            
#             # Create chunks from paragraphs
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
            
#             # Add remaining content
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
#         """Chunk forum post content"""
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
        
#         # Split long posts
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
#     """Manages embedding generation and storage"""
    
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#         self.model_name = model_name
    
#     def generate_embeddings(self, texts: List[str]) -> np.ndarray:
#         """Generate embeddings for a list of texts"""
#         return self.model.encode(texts, show_progress_bar=True)
    
#     def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], 
#                        embeddings_file: str, metadata_file: str):
#         """Save embeddings and metadata to files"""
#         # Save embeddings as compressed numpy array
#         np.savez_compressed(embeddings_file, embeddings=embeddings)
        
#         # Save metadata as JSON
#         with open(metadata_file, 'w', encoding='utf-8') as f:
#             json.dump(chunks, f, indent=2, ensure_ascii=False)
    
#     def load_embeddings(self, embeddings_file: str, metadata_file: str):
#         """Load embeddings and metadata from files"""
#         embeddings_data = np.load(embeddings_file)
#         embeddings = embeddings_data['embeddings']
        
#         with open(metadata_file, 'r', encoding='utf-8') as f:
#             chunks = json.load(f)
        
#         return embeddings, chunks

# class TDSVirtualTA:
#     """Main Virtual Teaching Assistant class"""
    
#     def __init__(self, data_dir: str = "data", embeddings_dir: str = "embeddings"):
#         self.data_dir = Path(data_dir)
#         self.embeddings_dir = Path(embeddings_dir)
#         self.embeddings_dir.mkdir(exist_ok=True)
        
#         self.chunker = DocumentChunker()
#         self.embedding_manager = EmbeddingManager()
        
#         # Files for storing embeddings and metadata
#         self.embeddings_file = self.embeddings_dir / "tds_embeddings.npz"
#         self.metadata_file = self.embeddings_dir / "tds_metadata.json"
        
#         # Load or generate embeddings
#         self.embeddings, self.chunks = self._load_or_generate_embeddings()
        
#         # --- AI model setup ---
#         self.aipipe_token = os.getenv("AIPIPE_TOKEN")
#         self.openai_model = "gpt-4.1-mini"  # You can change to any available model

#     def _load_or_generate_embeddings(self):
#         """Load existing embeddings or generate new ones"""
#         if self.embeddings_file.exists() and self.metadata_file.exists():
#             print("Loading existing embeddings...")
#             return self.embedding_manager.load_embeddings(
#                 str(self.embeddings_file), str(self.metadata_file)
#             )
#         else:
#             print("Generating new embeddings...")
#             return self._generate_embeddings()
    
#     def _generate_embeddings(self):
#         """Generate embeddings for all course content"""
#         all_chunks = []
        
#         # Process course materials
#         course_content_dir = self.data_dir / "tools-in-data-science-public"
#         if course_content_dir.exists():
#             for md_file in course_content_dir.rglob("*.md"):
#                 print(f"Processing: {md_file}")
#                 try:
#                     with open(md_file, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                     chunks = self.chunker.chunk_markdown(content, str(md_file))
#                     all_chunks.extend(chunks)
#                 except Exception as e:
#                     print(f"Error processing {md_file}: {e}")
        
#         # Process forum posts
#         forum_posts_file = self.data_dir / "tds_discourse_posts.json"
#         if forum_posts_file.exists():
#             print(f"Processing forum posts: {forum_posts_file}")
#             try:
#                 with open(forum_posts_file, 'r', encoding='utf-8') as f:
#                     posts = json.load(f)
                
#                 for post in posts:
#                     chunks = self.chunker.chunk_forum_post(post)
#                     all_chunks.extend(chunks)
#             except Exception as e:
#                 print(f"Error processing forum posts: {e}")
        
#         # Generate embeddings
#         texts = [chunk['content'] for chunk in all_chunks]
#         embeddings = self.embedding_manager.generate_embeddings(texts)
        
#         # Save embeddings and metadata
#         self.embedding_manager.save_embeddings(
#             embeddings, all_chunks, str(self.embeddings_file), str(self.metadata_file)
#         )
        
#         return embeddings, all_chunks
    
#     def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Search for similar chunks using cosine similarity"""
#         query_embedding = self.embedding_manager.model.encode([query])
        
#         # Calculate similarities
#         similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
#         # Get top-k most similar chunks
#         top_indices = np.argsort(similarities)[::-1][:top_k]
        
#         results = []
#         for idx in top_indices:
#             chunk = self.chunks[idx].copy()
#             chunk['similarity'] = float(similarities[idx])
#             results.append(chunk)
        
#         return results
    
#     def generate_response(self, question: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
#         """Generate response using RAG"""
#         # List of faculty/TA usernames
#         authoritative_users = {
#             "s.anand", "carlton", "Jivraj", "devam_07", "PulkitMangal",
#             "22f3001517", "SahuUtkarsh03", "deepbist", "23f1001171"
#         }

#         relevant_chunks = self.search_similar_chunks(question, top_k=8)

#         # Prioritize authoritative posts
#         def is_authoritative(chunk):
#             return (
#                 chunk['type'] == 'forum_post' and
#                 chunk.get('metadata', {}).get('username') in authoritative_users
#             )

#         # Sort: authoritative first, then by similarity
#         relevant_chunks.sort(key=lambda c: (not is_authoritative(c), -c.get('similarity', 0)))

#         # Prepare context
#         context = ""
#         for i, chunk in enumerate(relevant_chunks):
#             context += f"Context {i+1}:\n"
#             context += f"Source: {chunk['source']}\n"
#             context += f"Content: {chunk['content']}\n\n"
        
#         image_context = ""
#         if image_base64:
#             image_context = "\n\nThe user has also provided an image with their question. Please consider the image content when formulating your response."
        
#         prompt = f"""Act like a Virtual Teaching Assistant for the "Tools in Data Science" course. You are responsible for helping students by answering their questions accurately and concisely using only the context provided from official course materials and forum discussions.

# Your task is to evaluate and respond to a student’s question, which may include an image. Carefully analyze the question and any image it contains. Your answer must strictly follow the guidelines below.

# Objective:
# Provide a practical, well-reasoned, and context-based answer that supports the student's learning and aligns with the official course content.

# Step-by-step instructions:
# 1. Begin by examining whether the student has uploaded an image. If yes, extract the core problem from the image. Identify the question being asked, and propose the most efficient and practical method to solve it. Clearly state your reasoning process.
# 2. Next, read the written question provided by the student. Do not assume their statements are correct—validate everything only against the provided context.
# 3. Cross-reference the question (and image, if applicable) with the provided context from course materials and forum posts. Ensure all information in your answer is drawn directly from that context.
# 4. If the context lacks enough information to answer the question accurately, explicitly state: "The context provided does not contain sufficient information to answer this question reliably."
# 5. Focus on providing practical guidance that the student can apply. If a specific method or formula is implied in the context, walk the student through its application.
# 6. If your answer refers to a specific course material or forum post, cite it using brief labels (e.g., "As seen in Module 3 notes" or "Forum reply on k-means clustering from Week 5").
# 7. Use concise, student-friendly language and avoid generic or speculative responses. Talk directly to the student as if you are their personal TA.
# 8. Follow all recommendations given in the course materials. For example, the course material suggests using Podman over Docker, so your answer should reflect that preference.
# 9. NOTE: The bonus marks for assignments in the course are added to the total, not the maximum. Scores are typically shown out of 100. For example, if a student gets a bonus point for an assignment out of 10, their score is given as 11/10 or 110 on their dashboard.

# IMPORTANT:
# – Do not treat student-provided information as factual unless it is confirmed by the course context.
# – Do not guess or extrapolate beyond the context.
# – Prioritize helping the student learn how to approach the problem practically.

# Student Question: {question}{image_context}

# Relevant Context from Course Materials and Forum:
# {context}

# Take a deep breath and work through this problem step by step.
# """

#         try:
#             # --- Use OpenAI via AIPipe ---
#             headers = {
#                 "Authorization": self.aipipe_token,
#                 "Content-Type": "application/json"
#             }
#             image_content = []
#             if image_base64:
#                 image_content.append({
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/png;base64,{image_base64}"
#                     }
#                 })

#             messages = [
#                 {
#                     "role": "user",
#                     "content": [{"type": "text", "text": prompt}] + image_content
#                 }
#             ]

#             data = {
#                 "model": self.openai_model,
#                 "messages": messages
#             }

#             resp = requests.post(
#             "https://aipipe.org/openai/v1/chat/completions",
#             headers=headers,
#             json=data,
#             timeout=30
#             )
#             resp.raise_for_status()
#             resp_json = resp.json()
#             # Extract answer from output
#             answer = ""
#             choices = resp_json.get("choices", [])
#             if choices:
#                 for choice in choices:
#                     message = choice.get("message", {})
#                     if message.get("role") == "assistant":
#                         answer += message.get("content", "")

#             if not answer:
#                 answer = "Sorry, I could not generate a response at this time."

#             # --- Link extraction  ---
#             links = []
#             for chunk in relevant_chunks[:5]:
#                 if chunk['similarity'] > 0.3:
#                     if chunk['type'] == 'forum_post':
#                         metadata = chunk.get('metadata', {})
#                         topic_id = metadata.get('topic_id')
#                         post_number = metadata.get('post_number')
#                         slug = metadata.get('slug') or chunk.get('slug')
#                         print(f"DEBUG: Processing forum_post with topic_id={topic_id}, post_number={post_number}, slug={slug}")
#                         if slug and topic_id and post_number:
#                             url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{post_number}"
#                         else:
#                             url = chunk['source']
#                         is_auth = metadata.get('username') in authoritative_users
#                         links.append({
#                             "url": url,
#                             "text": f"{'Authoritative: ' if is_auth else ''}Forum discussion - {chunk['content'][:100]}..."
#                         })
#                     elif chunk['type'] == 'course_material':
#                         file_path = chunk['metadata'].get('file_path', '')
#                         print("DEBUG: File path for course material:", file_path)
#                         md_name = Path(file_path).stem if file_path else 'content'
#                         links.append({
#                             "url": f"https://tds.s-anand.net/#/{md_name}",
#                             "text": f"Course material: {chunk.get('header', 'Content section')}"
#                         })
#             # Remove duplicate URLs
#             seen_urls = set()
#             unique_links = []
#             for link in links:
#                 if link['url'] not in seen_urls:
#                     unique_links.append(link)
#                     seen_urls.add(link['url'])
            
#             return {
#                 "answer": answer,
#                 "links": unique_links[:5]  # Limit to 5 links
#             }
            
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return {
#                 "answer": "I apologize, but I encountered an error while processing your question. Please try again or contact the course staff for assistance.",
#                 "links": []
#             }

# # Flask API Server
# app = Flask(__name__)
# CORS(app)

# # Initialize Virtual TA
# print("Initializing TDS Virtual TA...")
# virtual_ta = TDSVirtualTA()
# print("Virtual TA ready!")

# @app.route('/api/', methods=['POST'])
# def handle_question():
#     """Handle student questions via API"""
#     try:
#         data = request.get_json()
        
#         if not data or 'question' not in data:
#             return jsonify({"error": "Question is required"}), 400
        
#         question = data['question']
#         image_base64 = data.get('image')
        
#         # Generate response
#         response = virtual_ta.generate_response(question, image_base64)
        
#         return jsonify(response)
        
#     except Exception as e:
#         print(f"Error handling request: {e}")
#         return jsonify({
#             "error": "Internal server error",
#             "answer": "I apologize, but I encountered an error processing your request.",
#             "links": []
#         }), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({"status": "healthy", "message": "TDS Virtual TA is running"})

# if __name__ == "__main__":
#     # For local development
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

import os
import json
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class TDSVirtualTA:
    def __init__(self, embeddings_dir: str = "embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_file = self.embeddings_dir / "tds_embeddings.npz"
        self.metadata_file = self.embeddings_dir / "tds_metadata.json"
        self.embeddings = np.load(self.embeddings_file)['embeddings']
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        self.aipipe_token = os.getenv("AIPIPE_TOKEN")
        self.openai_model = "gpt-4.1-mini"

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        client = genai.Client()
        response = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            content=query
        )
        query_embedding = np.array([response['embedding']])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            results.append(chunk)
        return results


    def generate_response(self, question: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        authoritative_users = {
            "s.anand", "carlton", "Jivraj", "devam_07", "PulkitMangal",
            "22f3001517", "SahuUtkarsh03", "deepbist", "23f1001171"
        }
        relevant_chunks = self.search_similar_chunks(question, top_k=8)
        def is_authoritative(chunk):
            return chunk['type'] == 'forum_post' and chunk.get('metadata', {}).get('username') in authoritative_users
        relevant_chunks.sort(key=lambda c: (not is_authoritative(c), -c.get('similarity', 0)))
        context = ""
        for i, chunk in enumerate(relevant_chunks):
            context += f"Context {i+1}:\nSource: {chunk['source']}\nContent: {chunk['content']}\n\n"
        image_context = "\n\nThe user has also provided an image with their question. Please consider the image content when formulating your response." if image_base64 else ""
        prompt = f"""Act like a Virtual Teaching Assistant for the "Tools in Data Science" course. You are responsible for helping students by answering their questions accurately and concisely using only the context provided from official course materials and forum discussions.

Your task is to evaluate and respond to a student’s question, which may include an image. Carefully analyze the question and any image it contains. Your answer must strictly follow the guidelines below.

Objective:
Provide a practical, well-reasoned, and context-based answer that supports the student's learning and aligns with the official course content.

Step-by-step instructions:
1. Begin by examining whether the student has uploaded an image. If yes, extract the core problem from the image. Identify the question being asked, and propose the most efficient and practical method to solve it. Clearly state your reasoning process.
2. Next, read the written question provided by the student. Do not assume their statements are correct—validate everything only against the provided context.
3. Cross-reference the question (and image, if applicable) with the provided context from course materials and forum posts. Ensure all information in your answer is drawn directly from that context.
4. If the context lacks enough information to answer the question accurately, explicitly state: "The context provided does not contain sufficient information to answer this question reliably."
5. Focus on providing practical guidance that the student can apply. If a specific method or formula is implied in the context, walk the student through its application.
6. If your answer refers to a specific course material or forum post, cite it using brief labels (e.g., "As seen in Module 3 notes" or "Forum reply on k-means clustering from Week 5").
7. Use concise, student-friendly language and avoid generic or speculative responses. Talk directly to the student as if you are their personal TA.
8. Follow all recommendations given in the course materials. For example, the course material suggests using Podman over Docker, so your answer should reflect that preference.
9. NOTE: The bonus marks for assignments in the course are added to the total, not the maximum. Scores are typically shown out of 100. For example, if a student gets a bonus point for an assignment out of 10, their score is given as 11/10 or 110 on their dashboard.

IMPORTANT:
– Do not treat student-provided information as factual unless it is confirmed by the course context.
– Do not guess or extrapolate beyond the context.
– Prioritize helping the student learn how to approach the problem practically.

Student Question: {question}{image_context}

Relevant Context from Course Materials and Forum:
{context}

Take a deep breath and work through this problem step by step.
"""
        try:
            headers = {
                "Authorization": self.aipipe_token,
                "Content-Type": "application/json"
            }
            image_content = []
            if image_base64:
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + image_content}]
            data = {"model": self.openai_model, "messages": messages}
            resp = requests.post("https://aipipe.org/openai/v1/chat/completions", headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            answer = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            links = []
            for chunk in relevant_chunks[:5]:
                if chunk['similarity'] > 0.3:
                    if chunk['type'] == 'forum_post':
                        md = chunk.get('metadata', {})
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{md.get('slug')}/{md.get('topic_id')}/{md.get('post_number')}"
                        is_auth = md.get('username') in authoritative_users
                        links.append({"url": url, "text": f"{'Authoritative: ' if is_auth else ''}Forum discussion - {chunk['content'][:100]}..."})
                    elif chunk['type'] == 'course_material':
                        fp = chunk['metadata'].get('file_path', '')
                        md_name = Path(fp).stem if fp else 'content'
                        links.append({"url": f"https://tds.s-anand.net/#/{md_name}", "text": f"Course material: {chunk.get('header', 'Content section')}"})
            seen_urls, unique_links = set(), []
            for link in links:
                if link['url'] not in seen_urls:
                    unique_links.append(link)
                    seen_urls.add(link['url'])
            return {"answer": answer or "Sorry, I could not generate a response.", "links": unique_links[:5]}
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"answer": "I apologize, an error occurred.", "links": []}

app = Flask(__name__)
CORS(app)
virtual_ta = TDSVirtualTA()

@app.route('/api/', methods=['POST'])
def handle_question():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        response = virtual_ta.generate_response(data['question'], data.get('image'))
        return jsonify(response)
    except Exception as e:
        print(f"Error handling request: {e}")
        return jsonify({"error": "Internal server error", "answer": "An error occurred.", "links": []}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "TDS Virtual TA is running"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)