from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import os
import cv2
from utils import face_preprocess
import sklearn
import configparser
import time

from nets.mtcnn_model import P_Net, R_Net, O_Net
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
import tensorflow.compat.v1 as tf
from convert_trt import engine as eng
from utils import inference as inf
import tensorrt as trt 

tf.disable_v2_behavior()

def init_runtime():
    TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    return trt_runtime


def load_mtcnn(conf):
    # load mtcnn model
    MODEL_PATH = conf.get("MTCNN", "MODEL_PATH")
    MIN_FACE_SIZE = int(conf.get("MTCNN", "MIN_FACE_SIZE"))
    STEPS_THRESHOLD = [float(i)  for i in conf.get("MTCNN", "STEPS_THRESHOLD").split(",")]

    detectors = [None, None, None]
    prefix = [MODEL_PATH + "/PNet_landmark/PNet",
              MODEL_PATH + "/RNet_landmark/RNet",
              MODEL_PATH + "/ONet_landmark/ONet"]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=MIN_FACE_SIZE, threshold=STEPS_THRESHOLD)

    return mtcnn_detector

def load_mobilefacenet(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def load_faces(faces_dir, mtcnn_detector, engine = None):
    face_db = []
    with tf.Graph().as_default():
        with engine:
            inputs_placeholder = tf.get_default_graph()#.get_tensor_by_name("input:0")
            for root, dirs, files in os.walk(faces_dir):
                files.remove('.gitignore')
                if('.DS_Store' in files): # macos
                    files.remove('.DS_Store')
                for file in files:
                    input_image = cv2.imread(os.path.join(root, file))
                    faces, landmarks = mtcnn_detector.detect(input_image)
                    bbox = faces[0,:4]
                    points = landmarks[0,:].reshape((5, 2))
                    nimg = face_preprocess.preprocess(input_image, bbox, points, image_size='112,112')

                    nimg = nimg - 127.5
                    nimg = nimg * 0.0078125
                    name = file.split(".")[0]

                    input_image = np.expand_dims(nimg,axis=0)

                    feed_dict = {inputs_placeholder: input_image}
                    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, len(faces), trt.float32)
                    emb_array = inf.do_inference(engine, feed_dict[inputs_placeholder], h_input, d_input, h_output, d_output, stream, batch_size=len(faces))
                    embedding = sklearn.preprocessing.normalize(emb_array).flatten()

                    face_db.append({
                        "name": name,
                        "feature": embedding
                    })
    return face_db

def feature_compare(feature1, feature2, threshold):
    # dist = np.sum(np.square(feature1- feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def draw_rect(faces, names, sims, image):
    for i, face in enumerate(faces):
        prob = '%.2f' % sims[i]
        label = "{}, {}".format(names[i], prob)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x, y = int(face[0]), int(face[1])
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)

    conf = configparser.ConfigParser()
    conf.read("config/main.cfg")

    mtcnn_detector = load_mtcnn(conf)
    ENGINE_PATH = conf.get("MOBILEFACENET","ENGINE_PATH")
    VERIFICATION_THRESHOLD = float(conf.get("MOBILEFACENET", "VERIFICATION_THRESHOLD"))
    FACE_DB_PATH = conf.get("MOBILEFACENET", "FACE_DB_PATH")
    trt_runtime = init_runtime()
    engine =  eng.load_engine(trt_runtime, ENGINE_PATH)
    faces_db = load_faces(FACE_DB_PATH, mtcnn_detector, engine = engine)

    with tf.Graph().as_default():
        with engine:
            inputs_placeholder = tf.get_default_graph()
            while True:
                ret, frame = cap.read()
                # fps calculation
                fps = 0
                prev_frame_time = time.time()
                new_frame_time = 0
                
                if ret:
                    faces,landmarks = mtcnn_detector.detect(frame)
                    if faces.shape[0] is not 0:
                        input_images = np.zeros((faces.shape[0], 112,112,3), dtype= np.float16)
                        for i, face in enumerate(faces):
                            if round(faces[i, 4], 6) > 0.95:
                                bbox = faces[i,0:4]
                                points = landmarks[i,:].reshape((5,2))
                                nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                                # cv2.imshow("face", nimg)
                                nimg = nimg - 127.5
                                nimg = nimg * 0.0078125
                                # input_image = np.expand_dims(nimg, axis=0)
                                input_images[i,:] = nimg
                        feed_dict = {inputs_placeholder: input_images}
                        h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, len(faces), trt.float32)
                        emb_arrays = inf.do_inference(engine, feed_dict[inputs_placeholder], h_input, d_input, h_output, d_output, stream, batch_size=len(faces)) # , HEIGHT, WIDTH

                        emb_arrays = sklearn.preprocessing.normalize(emb_arrays)
                        names = []
                        sims = []
                        for i, embedding in enumerate(emb_arrays):
                            embedding = embedding.flatten()
                            temp_dict = {}
                            for com_face in faces_db:
                                ret, sim = feature_compare(embedding, com_face["feature"], 0.6)
                                temp_dict[com_face["name"]] = sim
                            dict = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
                            if dict[0][1] > VERIFICATION_THRESHOLD:
                                name = dict[0][0]
                                sim = dict[0][1]
                            else:
                                name = "unknown"
                                sim = 0
                            names.append(name)
                            sims.append(sim)
                            x1, y1, x2, y2 = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                            x1 = max(int(x1), 0)
                            y1 = max(int(y1), 0)
                            x2 = min(int(x2), frame.shape[1])
                            y2 = min(int(y2), frame.shape[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        draw_rect(faces, names, sims, frame)
                        # fps
                        new_frame_time = time.time()
                        fps = 1/(new_frame_time-prev_frame_time)
                        prev_frame_time = new_frame_time

                    # Display FPS on the frame
                    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

if __name__ == '__main__':
    main()