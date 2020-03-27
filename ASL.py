import numpy as np
import os
import cv2
import argparse
from inference import Network
from gtts import gTTS
from signs import MASL


def get_args():
    parser = argparse.ArgumentParser("Translate signs to audio/text")
    v_desc = "The location of video input, default is 0"
    d_desc = "The device type, default is 'CPU'"
    
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    optional.add_argument("-v", help=v_desc, default= int(0))
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()
    
    return args

### active waits for a face to be detected then returns True
def active():
### Load intermediate representation of face detection model 
    args = get_args()
    model = 'face-detection-adas-0001.xml'
    weights = 'face-detection-adas-0001.bin'
    net = cv2.dnn.readNet(model, weights)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
### Start video, unless args.v has been changed, the onboard camera (at 0) is started
    cap = cv2.VideoCapture(int(args.v))
    while cap.isOpened():
        ret,frame = cap.read()
        if frame is None:
            raise Exception('Cannot find camera, current location is 0')
### process network model output - bounding box with at least 50% confidence 
        blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            if confidence < 0.5:
                continue
            if confidence > 0.5:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0,255), 4)
### display video 'q' to exit and release the camera
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
            cap.release()        
            return True
            

### sign_in cues the user through an audible beep to start signing then preprocesses 16 frames/sec for inference
def sign_in():
    args = get_args()
    c = -1
    os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 1080))
    height, width = 224,224
    frame_list = [ ]
    frame_array = [ ]
### The camera is again started, this causes the led to blink and cue the user to start signing
    cap = cv2.VideoCapture(int(0))
    while cap.isOpened():
        flag, frame = cap.read()
        hit, wit, col = np.shape(frame)
        if not flag:
            break
        if c< 16:
            frame_list.append(frame) 
            c+=1
        if cv2.waitKey(1) & 0xFF == ord('q') or c == 15:
            break
### frame_list contains 16 frames that make up the word to be translated
    for i in frame_list:
        sframe = cv2.resize(i, (224, 224))
        frame_array.append(sframe)
    imarray = np.asarray(frame_array)
    imarray = np.transpose(imarray, (3,0,1,2))
    imarray = imarray.reshape(1, 3, 16, height, width)
    return imarray
    
 
### infer creates the intermediate representation of the asl model and processes the inference to return
### a code for the translated word
def infer(imarray):
    model = "asl-recognition-0003.xml"
    inet = Network.net(model)
    exec_net = Network.load_model(inet, imarray, 'CPU')
    input_blob = next(iter(exec_net.inputs))
    input_layer = inet.inputs[input_blob].shape
    ##### asynchronous inference
    asy_net = Network.async_inference(exec_net, imarray, input_blob)
    output_blob = next(iter(exec_net.outputs))
    enc_net = exec_net.requests[0].outputs[output_blob]
    ###### synch - uncomment the 2 lines below and comment out the 3 lines below async to switch to synchronous inference
    #syn_net = Network.inf_(exec_net, imarray, input_blob)
    #enc_net = Network.extract_output(syn_net, exec_net)
    ###### continue
    code = (np.argmax(enc_net))
    return code
    

### txtplay uses gTTS to create an mp3 of the asl translated phrase
def txtPlay(tx):
    try:
        tts = gTTS(text=str(tx), lang='en')
        tts.save("tx.mp3")
        os.system("cvlc -q --play-and-exit tx.mp3")
        main()
    except:
        exit()    


### decode uses the translated code number and compares it to a list of dictionaries in signs.py
def decode(key):
    for j in MASL:
        if j['label'] == key:
### print the translation to the screen
            print(j['org_text'])
            tx = j['org_text']
            return tx

### manages the execution            
def main():
    go = 0
    phrase = [' ']
    activate = active()
    while go != str():
        if activate is False:
            continue
        elif activate is True:
            imarray = sign_in()
            inf = infer(imarray)
            sgn = decode(inf)
### Signing the word 'READ' will read back all signed words 
            if sgn != 'READ':
                phrase.append(sgn)
### Signing the word 'READ' again will exit
            elif sgn == 'READ':
                replay = " ".join(phrase)
                print(replay)
                txtPlay(replay)
                cap.release()
                exit()

    

if __name__ == "__main__":
    main()


