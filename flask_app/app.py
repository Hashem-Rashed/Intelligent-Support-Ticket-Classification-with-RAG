import os
import requests
import pandas as pd
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from werkzeug.utils import secure_filename
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['SESSION_TYPE'] = 'filesystem'

# Use environment variable for API base URL, fallback to localhost:8000
API_BASE = os.environ.get('API_BASE_URL', 'http://localhost:8000/api/v1')
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_api_status():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

@app.context_processor
def inject_globals():
    return dict(api_status=get_api_status(), api_base=API_BASE)

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
    return render_template('index.html', history=session['history'])

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text', '').strip()
    if not text:
        flash('Please enter ticket text.', 'warning')
        return redirect(url_for('index'))
    
    model_type = request.form.get('model_type', 'ensemble')
    try:
        resp = requests.post(
            f"{API_BASE}/classify",
            json={"text": text, "model_type": model_type, "return_details": True},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Save to history
        history_entry = {
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'category': data.get('category'),
            'confidence': data.get('confidence'),
            'model': model_type,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        history = session.get('history', [])
        history.insert(0, history_entry)
        session['history'] = history[:10]
        session.modified = True
        
        return render_template('index.html',
                               result=data,
                               input_text=text,
                               model_type=model_type,
                               history=session['history'])
    except requests.exceptions.ConnectionError:
        flash('Cannot connect to classification API. Make sure the backend is running (python -m src.api.main).', 'danger')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Classification error: {e}", 'danger')
        return redirect(url_for('index'))

@app.route('/rag')
def rag():
    return render_template('rag.html')

@app.route('/explain', methods=['POST'])
def explain():
    text = request.form.get('text', '').strip()
    model_type = request.form.get('model_type', 'ensemble')
    if not text:
        flash('Please enter ticket text.', 'warning')
        return redirect(url_for('rag'))
    
    try:
        resp = requests.post(
            f"{API_BASE}/rag/explain",
            json={"text": text, "model_type": model_type, "return_details": True},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return render_template('rag.html',
                               result=data,
                               input_text=text,
                               model_type=model_type)
    except requests.exceptions.ConnectionError:
        flash('Cannot connect to RAG API. Ensure the backend is running.', 'danger')
        return redirect(url_for('rag'))
    except Exception as e:
        flash(f"RAG error: {e}", 'danger')
        return redirect(url_for('rag'))

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/batch/upload', methods=['POST'])
def batch_upload():
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('batch'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('batch'))
    
    if not allowed_file(file.filename):
        flash('Only CSV files allowed', 'danger')
        return redirect(url_for('batch'))
    
    model_type = request.form.get('model_type', 'ensemble')
    
    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            flash('CSV must contain a "text" column', 'danger')
            return redirect(url_for('batch'))
        
        results = []
        for i, row in df.iterrows():
            text = str(row['text']).strip()
            if not text:
                results.append({"text": text, "category": "ERROR", "confidence": 0.0})
                continue
            try:
                resp = requests.post(
                    f"{API_BASE}/classify",
                    json={"text": text, "model_type": model_type, "return_details": True},
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                results.append({
                    "text": text,
                    "category": data.get('category', 'Unknown'),
                    "confidence": data.get('confidence', 0.0)
                })
            except Exception as e:
                results.append({"text": text, "category": "ERROR", "confidence": 0.0})
        
        result_df = pd.DataFrame(results)
        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name="classification_results.csv",
            mimetype="text/csv"
        )
    except Exception as e:
        flash(f"Batch processing error: {e}", 'danger')
        return redirect(url_for('batch'))

@app.route('/clear_history')
def clear_history():
    session['history'] = []
    flash('History cleared.', 'info')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    status = get_api_status()
    return jsonify({"status": "ok" if status else "degraded", "api": "up" if status else "down"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)