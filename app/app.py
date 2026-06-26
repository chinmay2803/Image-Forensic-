from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from pipelines.pipeline_manager import analyze_image
from PIL import Image
import numpy as np
from flask import send_file
from utils.pdf_generator import generate_forensic_report
from datetime import datetime
import json

# ==============================
# Configuration
# ==============================
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
# Helper functions
# ==============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_visuals(visuals, base_filename):
    """
    Save numpy array visuals as images in UPLOAD_FOLDER.
    Returns dict of URLs.
    """
    urls = {}
    for key, arr in visuals.items():
        if arr is None:
            continue
        
        # Special handling for masks - save as grayscale
        if key == 'mask':
            # Check if already scaled to 0-255
            if arr.max() <= 1:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            
            # Save as grayscale (mode 'L')
            img = Image.fromarray(arr, mode='L')
        else:
            # RGB images
            img = Image.fromarray(arr.astype(np.uint8))
        
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_{key}.png")
        img.save(save_path)
        urls[key] = url_for('static', filename=f"uploads/{base_filename}_{key}.png")
    
    return urls

# ==============================
# Routes
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if file and allowed_file(file.filename):
            # Unique filename
            filename = f"{int(np.random.randint(1e6))}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Mode from dropdown
            mode = request.form.get('mode', 'dis25k').lower()

            # Run analysis
            try:
                analysis = analyze_image(filepath, mode=mode, show=False)
            except Exception as e:
                return render_template('index.html', error=f"Error running analysis: {str(e)}")

            # Initialize defaults
            metadata = {}
            qtables = {}

            # Extract results for template
            if mode in ['autosplice', 'dis25k']:
                result = analysis["label"]
                techniques = analysis.get("triggered_techniques", [])
                visuals = save_visuals(analysis.get("visuals", {}), filename.split('.')[0])
                metadata = analysis.get("metadata", {})
                qtables = analysis.get("qtables", {})
                
            elif mode == 'both':
                # Merge summary
                result = f"AutoSplice: {analysis['autosplice']['label']}, DIS25k: {analysis['dis25k']['label']}"
                techniques = list(set(
                    analysis['autosplice'].get("triggered_techniques", []) +
                    analysis['dis25k'].get("triggered_techniques", [])
                ))
                # Save visuals for both pipelines
                visuals = {}
                visuals.update(save_visuals(analysis['autosplice'].get("visuals", {}), filename.split('.')[0] + "_autosplice"))
                visuals.update(save_visuals(analysis['dis25k'].get("visuals", {}), filename.split('.')[0] + "_dis25k"))
                
                # Get metadata from one pipeline
                metadata = analysis['dis25k'].get("metadata", {}) or analysis['autosplice'].get("metadata", {})
                qtables = analysis['dis25k'].get("qtables", {}) or analysis['autosplice'].get("qtables", {})
                
            else:
                result = "Unknown"
                techniques = []
                visuals = {}

            return render_template(
                'result.html',
                image_url=url_for('static', filename=f'uploads/{filename}'),
                result=result,
                techniques=techniques,
                visuals=visuals,
                metadata=metadata,
                qtables=qtables
            )

    return render_template('index.html')


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Extract image path from URL
        image_filename = data.get('image_url', '').split('/')[-1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        result = data.get('result', '')
        techniques = data.get('techniques', [])
        metadata = data.get('metadata', {})
        qtables = data.get('qtables', {})
        visuals = data.get('visuals', {})
        
        # Convert visual URLs to actual file paths
        visuals_paths = {}
        for key, url in visuals.items():
            filename = url.split('/')[-1]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                visuals_paths[key] = file_path
        
        # Generate PDF buffer
        pdf_buffer = generate_forensic_report(
            image_path, result, techniques, 
            metadata, qtables, visuals_paths
        )
        
        # CRITICAL: Ensure buffer pointer is at start
        pdf_buffer.seek(0)
        
        # Generate unique filename
        pdf_filename = f'forensic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        # Send file with proper headers
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=pdf_filename
        )
        
    except Exception as e:
        print(f"PDF Generation Error: {str(e)}")  # Debug log
        return {'error': str(e)}, 500
# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
