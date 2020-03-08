import numpy as np
import os
import cv2
import argparse
from inference import Network
from gtts import gTTS
from signs import MASL
import winsound


    
def get_args():
    parser = argparse.ArgumentParser("Translate ASL to audio/text")
    v_desc = "The location of video input, default is 0"
    d_desc = "The device type, default is 'CPU'"
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-v", help=v_desc, default= int(0))
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()
    return args

### active waits for a face to be detected then returns True for the asl 
### recognition portion of the app. 
def active():
    args = get_args()
    xfile = 'face-detection-adas-0001.xml'
    bfile = 'face-detection-adas-0001.bin'
### face detection models from the open model zoo are converted into IR using
### opencv
    net = cv2.dnn.readNetFromModelOptimizer(xfile, bfile)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    cap = cv2.VideoCapture(int(args.v))
    while cap.isOpened():
        ret,frame = cap.read()
        if frame is None:
            raise Exception('Cannot find camera, current location is 0')
### blob uses opencv to create the output from the IR, the image is also resized 
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
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0,255), 2)
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
    winsound.Beep(1080, 200)
    height, width = 224,224
    frame_list = [ ]
    frame_array = [ ]
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
    for i in frame_list:
        sframe = cv2.resize(i, (224, 224))
        frame_array.append(sframe)
    imarray = np.asarray(frame_array)
    imarray = np.transpose(imarray, (3,0,1,2))
    imarray = imarray.reshape(1, 3, 16, height, width)
    return imarray
    
 
### infer executes inference using the intermediate representation of the asl 
### model with Openvino and input video. It processes the input and returns a 
### code for the translated word
def infer(imarray):
    model = "asl-recognition-0003.xml"
    inet = Network.net(model)
    exec_net = Network.load_model(inet, imarray, 'CPU')
    input_blob = next(iter(exec_net.inputs))
    input_layer = inet.inputs[input_blob].shape
    ##### asynch
    asy_net = Network.async_inference(exec_net, imarray, input_blob)
    output_blob = next(iter(exec_net.outputs))
    enc_net = exec_net.requests[0].outputs[output_blob]
    ###### synch
    #syn_net = Network.inf_(exec_net, imarray, input_blob)
    #enc_net = Network.extract_output(syn_net, exec_net)
    code = (np.argmax(enc_net))
    return code
    

### txtplay uses gTTS to create an mp3 of the asl translated phrase
def txtPlay(tx):
    try:
        tts = gTTS(text=str(tx), lang='en')
        tts.save("tx.mp3")
        os.system("cvlc -q --play-and-exit tx.mp3")
    #os.system('play -nq -t alsa synth {} sine {}'.format(2, 0))
        main()
    except:
        exit()    


### decode uses the translated code number and compares it to a list of 
### dictionaries in signs.py
def decode(key):
    for j in MASL:
        if j['label'] == key:
            print(j['org_text'])
            tx = j['org_text']
            return tx

### manages the execution            
def main():
    
    go = 0
    phrase = [' ']
    while go != str():
        activate = active()
        if activate is False:
            continue
        elif activate is True:
            
            imarray = sign_in()
            inf = infer(imarray)
            sgn = decode(inf) 
            if sgn != 'READ':
                phrase.append(sgn)
            elif sgn == 'READ':
                replay = " ".join(phrase)
                print(replay)
                txtPlay(replay)
                cap.release()
                exit()

    

if __name__ == "__main__":
    main()


