# Important imports
from unittest import result
from app import app
from flask import request, render_template, url_for
from keras import models
import numpy as np
from PIL import Image
import string
import random
import os
import cv2

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Loading model
model = models.load_model('app/static/model/aur_1.h5')

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

	# Execute if request is get
	if request.method == "GET":
		full_filename =  'images/white_bg.jpg'
		return render_template("index.html", full_filename = full_filename)

	# Execute if reuqest is post
	if request.method == "POST":

		# Generating unique image name
		letters = string.ascii_lowercase
		name = ''.join(random.choice(letters) for i in range(10)) + '.jpg'
		full_filename =  'uploads/' + name
		print(full_filename)

		# Reading, resizing, saving and preprocessing image for predicition 
		image_upload = request.files['image_upload']
		print(type(image_upload))
		imagename = image_upload.filename
		image = Image.open(image_upload)
		image11=np.array(image)
		print('length',len(image11))
		#image1 = cv2.imread('C:/Users/user/Pictures/Camera Roll/d.jpg')
		#print('opencv-->',type(image1))
		#image1 = cv2.resize(image1,(256,192))
		#image = image.resize((192,256))
		img = cv2.resize(image11,(256, 192))
		print('image-->',type(img))
		image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))
		#image_arr = np.array(image.convert('RGB'))
		#image_arr.shape = (1,192,256,3)

		# Predicting output
		#predi=model.predict(np.array([img]).astype(np.float32))
        #print(predi)
		result=model.predict(np.array([img]).astype(np.float32))
		print(result)
		#print('opencv--->',result1)
		#result=model.predict(image_arr)
		#result = model.predict(np.array([image]).astype(np.float32))
		print('image-->',result)
		ind = np.argmax(result)
		classes = ['Normal','Allergic you need use our Radiant Plump Soap']
		print(ind)

		# Returning template, filename, extracted text
		return render_template('index.html', full_filename = full_filename, pred = classes[ind])

# Main function
if __name__ == '__main__':
    app.run(debug=True)
