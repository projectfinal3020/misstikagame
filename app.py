from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from pydub import AudioSegment
import librosa 
from tensorflow.keras.models import load_model


app = Flask(__name__)

model= load_model('my_model.h5')

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected." 


@app.route('/pred_alphabet', methods=['POST'])
def convert_m4a_to_wav():
    # Check if the request contains the M4A file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    m4a_file = request.files['file']
    
    m4a_path = 'temp.m4a'
    wav_path = 'temp.wav'

    try:
        # Save the M4A file
        m4a_file.save(m4a_path)

        # Convert M4A to WAV
        audio = AudioSegment.from_file(m4a_path, format='m4a')
        audio.export(wav_path, format='wav')

        # Load the WAV file using librosa
        y, sr = librosa.load(wav_path)
        X = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
        X=np.array(X)
        X=np.expand_dims(X,-1)
        #prediksi
        predictions = np.argmax(model.predict(np.array(X)), axis = -1)
        # print(predictions)
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        # Remove temporary WAV file
        os.remove(wav_path)

        # Return the audio data and sample rate
        return jsonify({'code':200,
            'result': labels[predictions[0]]})

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        # Remove temporary M4A file
        os.remove(m4a_path)
    

@app.route('/pred_image/<int:item_id>', methods=['POST'])

def image_similarity(item_id):
    img_file = request.files.get('img_file')
    if item_id == 1:
        img00 = cv2.imread('Gambar_Yang_Benar/Apel.png', cv2.IMREAD_COLOR)
        
    img_bytes = img_file.read() # Read the file content as bytes
    img_np = np.frombuffer(img_bytes, np.uint8) # Convert bytes to numpy array
    img01 = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # Decode the image
    
    orb_similarity = orb_sim(img00, img01)
    if orb_similarity >0.8:
        return jsonify ({'code':200,
            'result': 'Yeay, kamu benar'})
    else:
        return jsonify ({'code':400,
            'result': 'Ayo, belajar lagi!'})
    
def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    #perform matches. 
    matches = bf.match(desc_a, desc_b)
    #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]  
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


if __name__ == '__main__':
    app.run(debug=True)