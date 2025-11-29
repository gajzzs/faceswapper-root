from typing import Any, List, Callable
import cv2
import numpy as np
import onnxruntime
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.HYPERSWAP-FACE-SWAPPER'

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/hyperswap_1a_256.onnx')
            # Create ONNX session directly with proper configuration
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 3
            
            FACE_SWAPPER = onnxruntime.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def prepare_source_embedding(source_face: Face) -> np.ndarray:
    """Prepare source embedding for hyperswap model"""
    # Use normalized embedding for hyperswap
    source_embedding = source_face.normed_embedding.reshape((1, -1))
    return source_embedding.astype(np.float32)

def prepare_crop_frame(crop_frame: np.ndarray) -> np.ndarray:
    """Prepare crop frame for hyperswap model"""
    # Hyperswap uses specific normalization
    crop_frame = crop_frame[:, :, ::-1] / 255.0  # BGR to RGB and normalize
    crop_frame = (crop_frame - 0.5) / 0.5  # Normalize to [-1, 1]
    crop_frame = crop_frame.transpose(2, 0, 1)  # HWC to CHW
    crop_frame = np.expand_dims(crop_frame, axis=0).astype(np.float32)
    return crop_frame

def normalize_output_frame(output_frame: np.ndarray) -> np.ndarray:
    """Normalize output frame from hyperswap model"""
    output_frame = output_frame.transpose(1, 2, 0)  # CHW to HWC
    output_frame = output_frame * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    output_frame = np.clip(output_frame, 0, 1)
    output_frame = output_frame[:, :, ::-1] * 255  # RGB to BGR and scale
    return output_frame.astype(np.uint8)

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Swap face using hyperswap model"""
    face_swapper = get_face_swapper()
    
    # Get face landmarks and crop the face region
    kps = target_face.kps
    
    # Create a simple crop around the face
    x1, y1, x2, y2 = target_face.bbox.astype(int)
    
    # Expand bbox slightly
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(temp_frame.shape[1], x2 + margin)
    y2 = min(temp_frame.shape[0], y2 + margin)
    
    # Crop and resize to 256x256
    crop_frame = temp_frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame, (256, 256))
    
    # Prepare inputs
    source_embedding = prepare_source_embedding(source_face)
    target_input = prepare_crop_frame(crop_frame)
    
    # Run inference
    inputs = {
        'source': source_embedding,
        'target': target_input
    }
    
    try:
        output = face_swapper.run(None, inputs)[0][0]
        output_frame = normalize_output_frame(output)
        
        # Resize back and paste
        output_frame = cv2.resize(output_frame, (x2-x1, y2-y1))
        result_frame = temp_frame.copy()
        result_frame[y1:y2, x1:x2] = output_frame
        
        return result_frame
    except Exception as e:
        print(f"Hyperswap inference error: {e}")
        return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)