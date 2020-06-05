# ASL TRANSLATOR

REQUIRED Openvino ver 2020.1

Udacity 
There is a short demonstration video in the repository, the full length video is a recording of the desktop
as I was recording the above demo

ASL_win.py is untested but is intended to add support for windows

BACKGROUND

An estimated 1 in 1000 babies are born deaf and it is the #1 birth defect in the US. 
According to the World Health Organization, 360,000,000 people worldwide suffer from disabling
hearing loss. Each of these people has family, friends, acquaintences and coworkers that do not 
use sign language and have great difficulty communicating with them. There is an overwhelming 
need for a method for these people to communicate. 

A gesture recognition tool is an ideal solution for this issue. Being able to open an app on 
your phone and communicate with people who are hard of hearing would not only provide relief 
for those suffering from deafness but it would also provide 360 million people a method to 
contribute in a meaningful way without being held back by not getting their ideas and their 
message across to the hearing community. The ASL Translator is a demonstration of one possible 
method for resolving this issue.

ASL Translator is a gesture recognition app that uses previously trained models 
to recognize and translate the American Sign Language(ASL). The translator 
model uses the SD3 framework with a MobileNet V3 backbone. The ASL translator is 
triggered by a face detection model using the WIDER FACE detection benchmark. 
The face detect network features a default MobileNet backbone that includes 
depth-wise convolutions.

ADDONS

Ideally the translator would use facial recognition as an upgrade to just face 
detection. This way the user could pick someone out of a crowd to communicate with. The app 
would be more useful when installed on a phone, allowing the user to be able to translate 
on the spot as opposed to finding a computer. The user could open an app on their 
phone, point the camera and start translating. Most obviously there needs to be a 
a complete dictionary of translations so that a complete conversation could be held.
Also, the translator could serve as a ASL teacher. Signs.py already contains links to see a youtube
video of the word the user is wanting to sign but a database somewhere of a complete dictionary
of words that would be reachable at any time and would not change would work best.


INSTALLATION/CONFIGURATION

REQUIRED Openvino ver 2020.1 is required and it is assumed the setup is complete and all requirements have been installed and the test samples have been run.

Install vlc media player( https://www.videolan.org/vlc/ ) for audible translation.

All the files in the manifest should be copied to the same folder. Openvino 
environment variables should be sourced and the translator should be run with a 
python3 interpreter. The translator has been tested with and requires Openvino 2020.1 and 
Python3.6 to be installed. 
The audible portion currently requires VLC media player to be installed and 
an internet connection. To capture the ASL gestures for translation the camera 
device id 0 (default). The ASL translator relies on the following python dependencies, (pip3 install)…

a. OpenCV 4.0.1

b. numpy1.18

c. gTTS 2.1.1

d. os

e. sys

f. argsparser

g. winsound (windows only)


OPERATION INSTRUCTIONS
Linux
1.) In a terminal, move to the directory where all the files are located, run source setupvars to setup openvino enviornment variables and run ‘python3 ASL.py’ (-v to set the video location, 0 is default, -d to set the device, ‘CPU’ is default)
Currently only CPU is supported for device

2.) The app loops through a face detection model until a face is detected in view of the camera.

3.) Once detection occurs, the sign language recognition model takes over. The user is cued to begin signing visually (the camera LED will blink once) and audibly (user will hear a 'beep').

4.) The translator will recognize gestures from the MSASL-100 dataset (see 
signs.py for the full list of recognized signs and youtube links to see how to sign them).

5.) The user is allowed 1 second per sign and can continue to sign until 
the desired message is complete at which point the user should gesture the sign for 
‘READ’ to have the message audibly translated.

6a.) Once the message is played back the user may step out of view from the camera and the app will wait until another face is detected before returning to step 3 or...

b.) the users may sign another message or...

c.) sign ‘READ’ again to close the app.

MANIFEST

7 files need to be in the same folder.

a.)  ASL.py (update ASL_linux.py or ASL_win.py, not both)

b.)  asl-recognition-0003.bin

c.)  asl-recognition-0003.xml

d.)  Face-detection-adas-0001.bin

e,)  face-detection-adas-0001.xml

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
they randomly selected 40%/10%/50% data as training, validation and testing sets.

asl-recognition-0003 model uses the SD3 framework with a MobileNet V3 backbone. 
The model uses the MS-ASL100 (Microsoft American Sign Language 100) data set to 
translate 100 words to intermediate representations of those signs. 


gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with 
Google Translate text-to-speech API

