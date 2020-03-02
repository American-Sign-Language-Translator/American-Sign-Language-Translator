# ASL TRANSLATOR
Udacity 

BACKGROUND

ASL Translator is a gesture recognition app that uses previously trained models 
to recognize and translate the American Sign Language(ASL). The translator 
model uses the SD3 framework with a MobileNet V3 backbone. The ASL translator is 
triggered by a face detection model using the WIDER FACE detection benchmark. 
The face detect network features a default MobileNet backbone that includes 
depth-wise convolutions.

INSTALLATION/CONFIGURATION

Install vlc media player( https://www.videolan.org/vlc/ )

All the files in the manifest should be copied to the same folder. Openvino 
environment variables should be sourced and the translator should be run with a 
python3 interpreter. The translator has been tested with Openvino 2020.1 and 
Python3.6 but should work with any Python3 or recent Openvino version. 
The audible portion currently requires VLC media player to be installed and 
an internet connection. To capture the ASL gestures for translation the camera 
device id 0 (default). The ASL translator relies on the following python libraries…

a. OpenCV 4.0.1

b. numpy1.18

c. gTTS 2.1.1

d. os

e. sys

f. argsparser


OPERATION INSTRUCTIONS

1.) In a terminal, change to the directory where all the files are located and run ‘python3 ASL.py’ (-v to set the video location 0 is default, -d to set the device, ‘CPU’ is default)

2.) The app loops through a face detection model until a face is detected in view of the camera.

3.) Once detection occurs, the sign language recognition model takes over. The user is cued to begin signing visually (the camera LED will blink once) and audibly (user will hear a 'beep').

4.) The translator will recognize gestures from the MSASL-100 dataset (see 
signs.py for the full list of recognized signs).

5.) The user is allowed 1 second per sign and can continue to sign until 
the desired message is complete at which point the user should gesture the sign for 
‘READ’ to have the message audibly translated.

6a.) Once the message is played back the user may step out of view from the camera and the app will wait until another face is detected before returning to step 3 or...

b.) the users may sign another message or...

c.) sign ‘READ’ again to close the app.

MANIFEST

7 files need to be in the same folder.

a.)  ASL.py

b.)  asl-recognition-0003.bin

c.)  asl-recognition-0003.xml

d.)  head-pose-estimation-adas-0001.bin

e,)  head-pose-estimation-adas-0001.xml

f.)  inference.py

g.)  signs.py


Info

Openvino is an Intel open source project and allows the user to create an app 
using previously trained AI models by creating an intermediate representation 
of the model suitable to be deploy in an application at the edge of the network. 
Below are the pretrained models used to enable this project.

face-detection-adas-0003 (Multimedia Laboratory, Department of Information 
Engineering, The Chinese University of Hong Kong  WIDER FACE)  The network 
features a default MobileNet backbone that includes depth-wise convolutions. 
Over 32,000 images were chosen and almost 400,000 faces were labeled with a 
high degree of variability in scale, pose, illumination and occlusion. The 
WIDER FACE dataset is organized based on 61 event classes. For each event class,
we randomly select 40%/10%/50% data as training, validation and testing sets.

asl-recognition-0003 model uses the SD3 framework with a MobileNet V3 backbone. 
The model uses the MS-ASL100 (Microsoft American Sign Language 100) data set to 
translate 100 words to intermediate representations of those signs. 


gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with 
Google Translate text-to-speech API

