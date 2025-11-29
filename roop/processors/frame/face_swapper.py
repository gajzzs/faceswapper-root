from typing import Any, List, Callable
import cv2
import insightface
import onnxruntime
import numpy as np
import threading
import os

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_choice = getattr(roop.globals, 'face_swap_model', 'hyperswap')
            
            if model_choice == 'hyperswap':
                hyperswap_path = resolve_relative_path('../models/hyperswap_1a_256.onnx')
                if os.path.exists(hyperswap_path):
                    try:
                        session_options = onnxruntime.SessionOptions()
                        session_options.log_severity_level = 3
                        FACE_SWAPPER = onnxruntime.InferenceSession(
                            hyperswap_path, 
                            sess_options=session_options,
                            providers=roop.globals.execution_providers
                        )
                        FACE_SWAPPER.model_type = 'hyperswap'
                        print(f"Loaded hyperswap model: {hyperswap_path}")
                    except Exception as e:
                        print(f"Failed to load hyperswap: {e}")
                        return None
                else:
                    print(f"Hyperswap model not found: {hyperswap_path}")
                    return None
            else:  # inswapper
                inswapper_path = resolve_relative_path('../models/inswapper_128.onnx')
                if os.path.exists(inswapper_path):
                    FACE_SWAPPER = insightface.model_zoo.get_model(inswapper_path, providers=roop.globals.execution_providers)
                    FACE_SWAPPER.model_type = 'inswapper'
                    print(f"Loaded inswapper model: {inswapper_path}")
                else:
                    print(f"Inswapper model not found: {inswapper_path}")
                    return None
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
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


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()
    
    if hasattr(face_swapper, 'model_type') and face_swapper.model_type == 'hyperswap':
        return swap_face_hyperswap(source_face, target_face, temp_frame, face_swapper)
    else:
        return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def swap_face_hyperswap(source_face: Face, target_face: Face, temp_frame: Frame, face_swapper) -> Frame:
    """Swap face using hyperswap model with proper face alignment"""
    try:
        from insightface.utils import face_align
        
        # Use proper face alignment like FaceFusion
        kps = target_face.kps
        
        # Align face to 256x256 using landmarks
        aligned_face = face_align.norm_crop(temp_frame, landmark=kps, image_size=256, mode='arcface')
        
        # Prepare source embedding (hyperswap uses normalized embedding)
        source_embedding = source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        
        # Prepare target frame with proper normalization
        target_input = aligned_face.astype(np.float32)
        target_input = target_input[:, :, ::-1] / 255.0  # BGR to RGB and normalize
        target_input = (target_input - 0.5) / 0.5  # Normalize to [-1, 1] 
        target_input = target_input.transpose(2, 0, 1)  # HWC to CHW
        target_input = np.expand_dims(target_input, axis=0)
        
        # Run inference
        inputs = {
            'source': source_embedding,
            'target': target_input
        }
        
        output = face_swapper.run(None, inputs)[0][0]
        
        # Normalize output properly
        output_frame = output.transpose(1, 2, 0)  # CHW to HWC
        output_frame = output_frame * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
        output_frame = np.clip(output_frame, 0, 1)
        output_frame = (output_frame[:, :, ::-1] * 255).astype(np.uint8)  # RGB to BGR
        
        # Create transformation matrix for proper alignment back
        M = cv2.getAffineTransform(
            np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]], dtype=np.float32),
            kps[:3].astype(np.float32)
        )
        
        # Warp the swapped face back to original position
        result_frame = temp_frame.copy()
        warped_face = cv2.warpAffine(output_frame, M, (temp_frame.shape[1], temp_frame.shape[0]))
        
        # Create a mask for blending
        mask = np.zeros((temp_frame.shape[0], temp_frame.shape[1]), dtype=np.uint8)
        hull = cv2.convexHull(kps.astype(int))
        cv2.fillPoly(mask, [hull], 255)
        
        # Apply Gaussian blur to mask for smooth blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        
        # Blend the faces
        result_frame = result_frame.astype(np.float32)
        warped_face = warped_face.astype(np.float32)
        result_frame = result_frame * (1 - mask) + warped_face * mask
        result_frame = result_frame.astype(np.uint8)
        
        return result_frame
        
    except Exception as e:
        print(f"Hyperswap inference error: {e}")
        # Fallback to simple crop method if alignment fails
        return swap_face_hyperswap_simple(source_face, target_face, temp_frame, face_swapper)


def swap_face_hyperswap_simple(source_face: Face, target_face: Face, temp_frame: Frame, face_swapper) -> Frame:
    """Simple hyperswap fallback method"""
    try:
        # Get face bbox with better margins
        x1, y1, x2, y2 = target_face.bbox.astype(int)
        
        # Calculate face size and adjust margin accordingly
        face_width = x2 - x1
        face_height = y2 - y1
        margin = int(min(face_width, face_height) * 0.3)  # 30% margin
        
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(temp_frame.shape[1], x2 + margin)
        y2 = min(temp_frame.shape[0], y2 + margin)
        
        # Crop and resize to 256x256
        crop_frame = temp_frame[y1:y2, x1:x2]
        original_size = (x2-x1, y2-y1)
        crop_frame = cv2.resize(crop_frame, (256, 256))
        
        # Prepare inputs
        source_embedding = source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        target_input = crop_frame[:, :, ::-1].astype(np.float32) / 255.0
        target_input = (target_input - 0.5) / 0.5
        target_input = target_input.transpose(2, 0, 1)
        target_input = np.expand_dims(target_input, axis=0)
        
        # Run inference
        output = face_swapper.run(None, {'source': source_embedding, 'target': target_input})[0][0]
        
        # Process output
        output_frame = output.transpose(1, 2, 0)
        output_frame = output_frame * 0.5 + 0.5
        output_frame = np.clip(output_frame, 0, 1)
        output_frame = (output_frame[:, :, ::-1] * 255).astype(np.uint8)
        
        # Resize back and blend
        output_frame = cv2.resize(output_frame, original_size)
        
        # Create smooth mask for blending
        mask = np.ones((original_size[1], original_size[0]), dtype=np.float32)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / mask.max()
        mask = np.expand_dims(mask, axis=2)
        
        # Blend with original
        result_frame = temp_frame.copy().astype(np.float32)
        crop_region = result_frame[y1:y2, x1:x2]
        blended = crop_region * (1 - mask) + output_frame.astype(np.float32) * mask
        result_frame[y1:y2, x1:x2] = blended
        
        return result_frame.astype(np.uint8)
        
    except Exception as e:
        print(f"Simple hyperswap error: {e}")



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
