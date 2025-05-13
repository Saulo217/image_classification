from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from use_trained_model import classify_image
import imghdr

app = Flask(__name__)

# Configurações
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Arquivo excede o limite de 20 MB'}), 413

@app.route('/classificacao', methods=['POST'])
def classify_upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Arquivo não enviado no formulário'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Arquivo não selecionado'}), 400
    
    if file and allowed_file(file.filename):
        # Verifica se é realmente uma imagem
        header_bytes = file.read(512)
        file.seek(0)  # Volta ao início do arquivo após leitura para verificar tipo
        file_type = imghdr.what(None, header_bytes)
        if file_type not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'O arquivo enviado não é uma imagem válida'}), 400

        try:
            classification_result, prediction = classify_image(file)
            print(classification_result, prediction)
            return jsonify({
                'category': prediction,
                'detalhes': classification_result
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

if __name__ == '__main__':
    app.run(debug=True)

