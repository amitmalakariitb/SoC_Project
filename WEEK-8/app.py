import os

import torch
from flask import Flask, request, render_template, send_from_directory
from torchvision.transforms import functional as F
from PIL import Image
from generator import Generator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(in_channels=3)
generator.load_state_dict(torch.load(
    '../WEEK-5/generator_model.pth', map_location=device))
generator.eval()
generator.to(device)


def upscale_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_image = generator(image)
    generated_image = generated_image.squeeze(0).cpu()
    return F.to_pil_image(generated_image)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:

        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        low_res_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        low_res_image = Image.open(low_res_path)
        high_res_image = upscale_image(low_res_image)

        high_res_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        high_res_image.save(high_res_path)

        return render_template('result.html', low_res_path=low_res_path, high_res_path=high_res_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    app.run()
