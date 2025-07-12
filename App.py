from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
import torch
import re
from Main_FL import ModelABSQ
from flask import render_template,Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (e.g., from your HTML frontend)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Parse incoming JSON
        data = request.get_json()
        paragraph = data.get("textContent", "")
        query = data.get("userQuery", "")
        compression = data.get("compressionRange", {"min": 30, "max": 70})
        
        # Convert compression to summary length (simple logic: avg compression => length)
        avg_ratio = (int(compression["min"]) + int(compression["max"])) / 2
        total_words = len(paragraph.split())
        target_summary_words = max(20, int(total_words * (100 - avg_ratio) / 100))  # at least 20 words

        # Approximate length values (Pegasus uses tokens, not words, but this is close enough for now)
        Max_length = min(200, target_summary_words)
        Min_length = max(30, int(Max_length * 0.5))

        # Generate summary using your function
        summary = ModelABSQ.modelABSQ(paragraph, query, Max_length, Min_length)

        return jsonify({
            "success": True,
            "summary": summary,
            "selected_indices": [],  # Optional: if you want to return which sentences were used
            "statistics": {
                "original_sentences": len(paragraph.split('.')),
                "original_length": len(paragraph),
                "summary_sentences": len(summary.split('.')),
                "summary_length": len(summary),
                "compression_achieved": round(100 - (len(summary) / len(paragraph) * 100), 2),
                "query_used": bool(query.strip())
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)