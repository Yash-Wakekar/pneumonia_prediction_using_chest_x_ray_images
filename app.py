import os
from flask import Flask, flash, request, redirect,render_template
# from flask import *
from werkzeug.utils import secure_filename
import keras
import numpy as np
import operator as op
from keras.models import load_model

dirname = os.path.dirname(__file__)
relative_path = 'static/uploads'
UPLOAD_FOLDER = os.path.join(dirname, relative_path) #if u get error might be issue of path check it
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'hxcd1234ghfd'
model1 = load_model('final_model_resnet.h5')
model2 = load_model('final_model_densnet.h5')
model3 = load_model('final_model_vgg_final.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def homepage():
    return render_template('imgupload.html')


def ml_model(file_path):
    predictions=[]


    test_image = keras.utils.load_img(UPLOAD_FOLDER+"//"+file_path,target_size=(224, 224))
    test_image = keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result1 = model1.predict(test_image)
    # print(result1)
    result2 = model2.predict(test_image)
    # print(result2)
    result3 = model3.predict(test_image)
    # print(result3)
    confidence_s1 = np.squeeze(result1) * 100
    confidence_s2 = np.squeeze(result2) * 100
    confidence_s3 = np.squeeze(result3) * 100
    if (confidence_s1 > 95 and confidence_s1 <= 100):
        resnet_model_result = 1
    else:
        resnet_model_result = 0

    if (confidence_s2 > 99.99 and confidence_s2 <= 100):
        desnet_model_result = 1
    else:
        desnet_model_result = 0
    if (confidence_s3 > 80 and confidence_s3 <= 100):
        vgg_model_result = 1
    else:
        vgg_model_result = 0

    predictions.append(resnet_model_result)
    predictions.append(desnet_model_result)
    predictions.append(vgg_model_result)
    # print(predictions)
    occurenece_one=op.countOf(predictions, 1)
    occurenece_zero = op.countOf(predictions, 0)
    # print(occurenece_one,occurenece_zero)
    if(confidence_s3==100.0):
        return 1
    else:
        if(occurenece_one>occurenece_zero):
            return 1
        else:
            return 0

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # print(file)
        # if user does not select file and submit an empty part without filename

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction_result=ml_model(filename)
            # print(prediction_result)

    return render_template('page2.html',prediction_result=prediction_result)


@app.route('/searchdoctor', methods=["POST", "GET"])
def searchdoctor():

    if request.method == 'POST':
        loactionofuser = request.form['search']
        loc_link = "https://www.google.com/search?q=pneumonia+doctor+near+"+ loactionofuser +"+&sxsrf=ALiCzsY95eOfaDJY6TcXVBvKDdxCr2LAQQ%3A1666277803909&ei=q2FRY7WQN9GcseMPrqqUkAw&ved=0ahUKEwj1keXEiO_6AhVRTmwGHS4VBcIQ4dUDCA8&uact=5&oq=pneumonia+doctor+near+"+loactionofuser+"+&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECc6CAgAEKIEELADSgQIQRgBSgQIRhgAUO0tWNgzYPo6aAJwAHgAgAGiAYgBpwWSAQMwLjWYAQCgAQHIAQPAAQE&sclient=gws-wiz"
    return redirect(loc_link)

if __name__ =='__main__':
    app.run(debug = True)