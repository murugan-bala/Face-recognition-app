import os
import re
import json
#import cv2
import glob
import sys
import math
import flask 
import datetime
import requests
import logging
import base64
import numpy as np
import face_recognition
from flask import Flask , render_template
from flask import request
from flask import jsonify
#from waitress import serve
from flask_cors import CORS
from PIL import Image, ImageDraw

global out
app = Flask(__name__)
CORS(app)
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/verifyPhoto',methods = ['POST','GET'])
def face_detection():
    global out
    global error_code
    out=0
    try:
        print("verifyPhoto.face_detection: {}".format("Entered service method"),flush=True)
        if request.method == 'POST':
             print("Hello")
             #file = request.files['file']
             req_json=request.json
             #print("face_detection.req_json: {}".format(req_json),flush=True)
             print(type(req_json))
             if 'img1' in request.json:
                req_json=request.json
                '''encod_img1=req_json['img2'][22:].encode()
                print()
                print(encod_img1)'''
                encod_img1=req_json['img1'][22:].encode()
                #print(req_json['img2'][22:])  #iV
                encod_img2=req_json['img2'][22:].encode()
                #print(encod_img2)
                image1_decode = base64.decodebytes(encod_img1)
                with open('image1_decode.jpg', 'wb') as image1_result:
                    image1_result.write(image1_decode)
                    
                image2_decode = base64.decodebytes(encod_img2)
                with open('image2_decode.jpg', 'wb') as image2_result:
                    image2_result.write(image2_decode)
                    
                muru_image = face_recognition.load_image_file("image1_decode.jpg")
                print()
                print(face_recognition.face_encodings(muru_image))
                
                if len(muru_image)>0:
                    
                    muru_face_encoding = face_recognition.face_encodings(muru_image)[0]
                    # Create arrays of known face encodings and their names
                    known_face_encodings = [
                        muru_face_encoding   
                    ]
                    known_face_names = [
                        "Matched"   
                    ]

                    # Load an image with an unknown face
                    unknown_image = face_recognition.load_image_file("image2_decode.jpg")  #amu , sug , unknownface, tamizh,

                    # Find all the faces and face encodings in the unknown image
                    face_locations = face_recognition.face_locations(unknown_image,number_of_times_to_upsample=2)   #face_locations = face_recognition.face_locations(image, model="cnn")
                    print(face_locations)
                    print("There are ",len(face_locations),"people in this image")
                    no_face=len(face_locations)
                    print("Total number of faces : ",no_face)
                    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                    
                    if len(face_locations)==0:
                        print("No Faces Detected..")
                        result="No Faces Detected"
                        error_code=4
                        face_distances="no face detected"
                    elif len(face_locations)>1:
                        print("More than One Face detected....")
                        result="More than One Face detected"
                        error_code=3
                        face_distances="more than one face detected"
                    elif len(face_locations)==1:
                        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
                        # See http://pillow.readthedocs.io/ for more about PIL/Pillow
                        pil_image = Image.fromarray(unknown_image)
                        # Create a Pillow ImageDraw Draw instance to draw with
                        draw = ImageDraw.Draw(pil_image)
                        result=""
                        # Loop through each face found in the unknown image
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            print(matches)  # [True, False]
                            name = "Unknown"

                            # If a match was found in known_face_encodings, just use the first one.
                            # if True in matches:
                            # first_match_index = matches.index(true)
                            # name = known_face_names[first_match_index]

                            # Or instead, use the known face with the smallest distance to the new face
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            print(face_distances)  # [0.56227993 0.69760146]
                            #no_face=len(face_distances)
                            print(len(face_distances))
                            if face_distances[0]<0.60:
                                print("Photo Matched.....!!!")
                                result="Photo Matched"
                                face_distances=face_distances[0]
                                out=face_distance_to_conf(face_distances,face_match_threshold=0.6)
                                out=out*100
                                print("Accuracy : ",out)
                                error_code=1
                            else:
                                face_distances=face_distances[0]
                                print("Not Matched ")
                                result="Not Matched"
                                error_code=2
                            best_match_index = np.argmin(face_distances)
                            print(best_match_index)  # is 0
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                            # Draw a box around the face using the Pillow module
                            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                            # Draw a label with a name below the face
                            text_width, text_height = draw.textsize(name)
                            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
                        # Remove the drawing library from memory as per the Pillow docs
                        del draw

                        # Display the resulting image
                        #pil_image.show()

                        # You can also save a copy of the new image to disk if you want by uncommenting this line
                        pil_image.save("image_with_boxes.jpg")
                else:
                    
                    result="No photo detected in the image"
                    error_code=4
                    #out=0
                    
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Oops!",sys.exc_info()[0],"occured.")
        print("Error occured..")
        result="Not Matched"
        out=0
        no_face=0
        face_distances="error occured"
        error_code=0

    return jsonify({'response' : result,'Threshold_value':0.60,'Facial_distance':face_distances,'no_of_faces':no_face,'error_code':error_code})
    #return jsonify({'res' : "hello"})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5030,debug=True)
    #serve(app, host='0.0.0.0', port=5030)


