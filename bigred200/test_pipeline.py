import sys
print('Step 1: importing os')
sys.stdout.flush()
import os
os.environ['OMP_NUM_THREADS'] = '1'

print('Step 2: importing numpy')
sys.stdout.flush()
import numpy as np

print('Step 3: importing cv2')
sys.stdout.flush()
import cv2

print('Step 4: importing MTCNN')
sys.stdout.flush()
from pymtcnn import MTCNN

print('Step 5: importing CLNF')
sys.stdout.flush()
from pyclnf import CLNF

print('Step 6: importing CalcParams')
sys.stdout.flush()
from pyfaceau.alignment.calc_params import CalcParams

print('Step 7: importing PDMParser')
sys.stdout.flush()
from pyfaceau.features.pdm import PDMParser

print('All imports successful!')
sys.stdout.flush()

print('Loading models...')
sys.stdout.flush()
detector = MTCNN()
print('  MTCNN loaded')
sys.stdout.flush()

clnf = CLNF(model_dir='pyclnf/pyclnf/models')
print('  CLNF loaded')
sys.stdout.flush()

pdm_parser = PDMParser('pyfaceau/weights/In-the-wild_aligned_PDM_68.txt')
print('  PDMParser loaded')
sys.stdout.flush()

calc_params = CalcParams(pdm_parser)
print('  CalcParams loaded')
sys.stdout.flush()

print('Loading test frame...')
sys.stdout.flush()
cap = cv2.VideoCapture('S Data/Normal Cohort/IMG_0942.MOV')
ret, frame = cap.read()
cap.release()
print(f'  Frame loaded: {ret}, shape: {frame.shape if ret else None}')
sys.stdout.flush()

print('Running detection...')
sys.stdout.flush()
bboxes, _ = detector.detect(frame)
print(f'  Found {len(bboxes) if bboxes is not None else 0} faces')
sys.stdout.flush()

print('Running CLNF...')
sys.stdout.flush()
landmarks, info = clnf.fit(frame, bboxes[0][:4])
print(f'  Landmarks shape: {landmarks.shape}')
sys.stdout.flush()

print('Running CalcParams...')
sys.stdout.flush()
global_params, local_params = calc_params.calc_params(landmarks)
print(f'  Pose: rx={global_params[1]:.3f}, ry={global_params[2]:.3f}, rz={global_params[3]:.3f}')
sys.stdout.flush()

print('SUCCESS - Pipeline works!')
