import os
import sys
import webbrowser
import gradio as gr
from typing import Callable
import cv2
import roop.globals
import roop.metadata
from roop.face_analyser import get_many_faces, get_one_face
from roop.face_reference import clear_face_reference
from roop.utilities import is_image, is_video, resolve_relative_path

def init(start: Callable[[], None], destroy: Callable[[], None]) -> None:
    with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}') as ui:
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(label="Source Face", type="filepath")
                target_file = gr.File(label="Target File (Image or Video)", type="filepath")
                
                with gr.Group():
                    gr.Markdown("### Processing Options")
                    face_swapper = gr.Checkbox(label="Face Swapper", value=True)
                    face_enhancer = gr.Checkbox(label="Face Enhancer (GFPGAN)", value=False)
                    
                with gr.Group():
                    keep_fps = gr.Checkbox(label="Keep FPS", value=roop.globals.keep_fps)
                    skip_audio = gr.Checkbox(label="Skip Audio", value=roop.globals.skip_audio)
                    many_faces = gr.Checkbox(label="Many Faces", value=roop.globals.many_faces)
                    keep_frames = gr.Checkbox(label="Keep Frames", value=roop.globals.keep_frames)
                
                with gr.Group():
                    gr.Markdown("### Output")
                    output_path = gr.Textbox(
                        label="Output Path",
                        value="./output/",
                        info="Directory or full path for output file"
                    )
                
                with gr.Group():
                    gr.Markdown("### Face Selection")
                    face_count_display = gr.Textbox(label="Detected Faces", value="0", interactive=False)
                    face_position_slider = gr.Slider(
                        minimum=0, 
                        maximum=10, 
                        step=1, 
                        value=roop.globals.reference_face_position,
                        label="Select Face Position",
                        info="0 = First face detected"
                    )
                    detect_faces_btn = gr.Button("Detect Faces in Target", size="sm")
                
                with gr.Accordion("Advanced Settings", open=False):
                    reference_frame_number = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        step=1,
                        value=roop.globals.reference_frame_number,
                        label="Reference Frame Number",
                        info="Which video frame to use for face detection (0 = first frame)"
                    )
                    
                    similar_face_distance = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=roop.globals.similar_face_distance,
                        label="Face Similarity Threshold",
                        info="Lower = stricter matching (default: 0.85)"
                    )
                    
                    gr.Markdown("**Output Settings**")
                    
                    output_video_quality = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=roop.globals.output_video_quality,
                        label="Output Video Quality",
                        info="0 = best quality, 100 = worst (default: 35)"
                    )
                    
                    temp_frame_format = gr.Radio(
                        choices=["png", "jpg"],
                        value=roop.globals.temp_frame_format,
                        label="Temp Frame Format",
                        info="Format for extracted frames"
                    )
                    
                    temp_frame_quality = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=roop.globals.temp_frame_quality,
                        label="Temp Frame Quality",
                        info="0 = best (only for JPG)"
                    )
                    
                    gr.Markdown("**Processing Settings**")
                    
                    execution_threads = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=roop.globals.execution_threads,
                        label="Execution Threads",
                        info="Number of threads for processing"
                    )
                    
                    execution_batch_size = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=roop.globals.execution_batch_size,
                        label="Batch Size",
                        info="Number of frames per batch"
                    )
                    
                    gr.Markdown("**Hardware Acceleration**")
                    
                    execution_provider = gr.CheckboxGroup(
                        choices=["cpu", "cuda", "coreml", "mps", "directml"],
                        value=["cpu"],  # Default to CPU
                        label="Execution Providers",
                        info="Select hardware accelerators (order matters: first = highest priority)"
                    )
                    
                    gr.Markdown("**Memory Management**")
                    
                    max_memory = gr.Slider(
                        minimum=1,
                        maximum=64,
                        step=1,
                        value=roop.globals.max_memory if roop.globals.max_memory else 0,
                        label="Max Memory (GB)",
                        info="Limit RAM usage (0 = no limit)"
                    )
                
                start_button = gr.Button("Start", variant="primary")
                destroy_button = gr.Button("Destroy")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Face Detection"):
                        preview_image = gr.Image(label="Detected Faces")
                    
                    with gr.Tab("Result Preview"):
                        result_preview_image = gr.Image(label="Result Preview")
                        preview_frame_slider = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=0,
                            label="Preview Frame",
                            info="Select frame to preview"
                        )
                        preview_btn = gr.Button("Generate Preview", size="sm")
                    
                    with gr.Tab("Output"):
                        output_video = gr.Video(label="Processed Output")
                        output_image = gr.Image(label="Processed Output")
                
                status_label = gr.Label(value="Ready")
                
        def on_source_change(path):
            roop.globals.source_path = path
            
        def on_target_change(path):
            roop.globals.target_path = path
            clear_face_reference()
            
        def detect_faces(target_path, frame_number):
            if target_path is None:
                return "No target file selected", None
            
            try:
                # Load the image/first frame
                if is_image(target_path):
                    frame = cv2.imread(target_path)
                    timestamp_info = ""
                elif is_video(target_path):
                    capture = cv2.VideoCapture(target_path)
                    
                    # Get FPS to calculate timestamp
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        timestamp_seconds = int(frame_number) / fps
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        timestamp_info = f" (at {minutes:02d}:{seconds:02d})"
                    else:
                        timestamp_info = ""
                    
                    # Seek to the specified frame
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                    ret, frame = capture.read()
                    capture.release()
                    if not ret:
                        return f"Failed to read frame {int(frame_number)} from video", None
                else:
                    return "Invalid file type", None
                
                # Detect faces
                faces = get_many_faces(frame)
                face_count = len(faces) if faces else 0
                
                # Draw bounding boxes and labels on the frame
                preview_frame = frame.copy()
                for idx, face in enumerate(faces or []):
                    # Get bounding box coordinates
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    # Draw rectangle around face
                    cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Add label with face index
                    label = f"Face {idx}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(preview_frame, 
                                (x1, y1 - text_height - 10), 
                                (x1 + text_width + 10, y1), 
                                (0, 255, 0), -1)
                    
                    # Draw text
                    cv2.putText(preview_frame, label, (x1 + 5, y1 - 5), 
                              font, font_scale, (0, 0, 0), thickness)
                
                # Convert BGR to RGB for Gradio
                preview_frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                
                frame_info = f"Frame {int(frame_number)}{timestamp_info}: " if is_video(target_path) else ""
                return f"{frame_info}{face_count} face(s) detected", preview_frame_rgb
            except Exception as e:
                return f"Error: {str(e)}", None
        
        def on_face_position_change(position):
            roop.globals.reference_face_position = int(position)
            clear_face_reference()
            
        def on_settings_change(k_fps, s_audio, m_faces, k_frames):
            roop.globals.keep_fps = k_fps
            roop.globals.skip_audio = s_audio
            roop.globals.many_faces = m_faces
            roop.globals.keep_frames = k_frames
        
        def update_frame_processors(swapper, enhancer):
            processors = []
            if swapper:
                processors.append('face_swapper')
            if enhancer:
                processors.append('face_enhancer')
            roop.globals.frame_processors = processors if processors else ['face_swapper']
        
        def update_output_path(path):
            # Handle output path
            if os.path.isdir(path):
                roop.globals.output_path = path
            else:
                # If it's a file path, use it directly
                roop.globals.output_path = path
        
        def generate_preview(target_path, source_path, frame_num, use_swapper, use_enhancer):
            if not target_path or not source_path:
                return None
            
            try:
                from roop.capturer import get_video_frame
                from roop.processors.frame.core import get_frame_processors_modules
                import cv2
                
                # Get frame
                temp_frame = get_video_frame(target_path, int(frame_num))
                if temp_frame is None:
                    return None
                
                # Get source face
                source_face = get_one_face(cv2.imread(source_path))
                if source_face is None:
                    return None
                
                # Get reference face
                reference_frame = get_video_frame(target_path, roop.globals.reference_frame_number)
                reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
                
                # Update frame processors temporarily
                original_processors = roop.globals.frame_processors[:]
                processors = []
                if use_swapper:
                    processors.append('face_swapper')
                if use_enhancer:
                    processors.append('face_enhancer')
                roop.globals.frame_processors = processors if processors else ['face_swapper']
                
                # Process frame
                for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                    temp_frame = frame_processor.process_frame(source_face, reference_face, temp_frame)
                
                # Restore original processors
                roop.globals.frame_processors = original_processors
                
                # Convert to RGB
                from PIL import Image
                temp_frame_rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                return temp_frame_rgb
            except Exception as e:
                print(f"Preview error: {str(e)}")
                return None

        def on_start(output_path_value):
            if roop.globals.source_path and roop.globals.target_path:
                # Ensure output path is set from UI
                if not roop.globals.output_path:
                    roop.globals.output_path = output_path_value
                
                # Determine output path
                output_location = roop.globals.output_path
                
                if not output_location:
                    return "Please select an output path.", None, None
                
                if os.path.isdir(output_location):
                    # If output_path is a directory, generate filename
                    target_name, target_ext = os.path.splitext(os.path.basename(roop.globals.target_path))
                    output_file = os.path.join(output_location, f"{target_name}_output{target_ext}")
                else:
                    # If output_path is a full path, use it directly
                    output_file = output_location
                
                roop.globals.output_path = output_file
                
                status_label.value = "Processing..."
                start()
                
                # Return output for display
                if is_video(roop.globals.target_path):
                    return f"Processing complete! Output: {output_file}", output_file, None
                else:
                    return f"Processing complete! Output: {output_file}", None, output_file
            else:
                return "Please select source and target files.", None, None
        
        def on_destroy():
            destroy()

        source_image.change(on_source_change, inputs=[source_image])
        target_file.change(on_target_change, inputs=[target_file])
        
        detect_faces_btn.click(detect_faces, inputs=[target_file, reference_frame_number], outputs=[face_count_display, preview_image])
        face_position_slider.change(on_face_position_change, inputs=[face_position_slider])
        
        keep_fps.change(lambda x: setattr(roop.globals, 'keep_fps', x), inputs=[keep_fps])
        skip_audio.change(lambda x: setattr(roop.globals, 'skip_audio', x), inputs=[skip_audio])
        many_faces.change(lambda x: setattr(roop.globals, 'many_faces', x), inputs=[many_faces])
        keep_frames.change(lambda x: setattr(roop.globals, 'keep_frames', x), inputs=[keep_frames])
        
        # Frame processors
        face_swapper.change(update_frame_processors, inputs=[face_swapper, face_enhancer])
        face_enhancer.change(update_frame_processors, inputs=[face_swapper, face_enhancer])
        
        # Output path
        output_path.change(update_output_path, inputs=[output_path])
        
        # Preview
        preview_btn.click(
            generate_preview, 
            inputs=[target_file, source_image, preview_frame_slider, face_swapper, face_enhancer],
            outputs=[result_preview_image]
        )
        
        # Advanced settings handlers
        reference_frame_number.change(lambda x: setattr(roop.globals, 'reference_frame_number', int(x)), inputs=[reference_frame_number])
        similar_face_distance.change(lambda x: setattr(roop.globals, 'similar_face_distance', x), inputs=[similar_face_distance])
        output_video_quality.change(lambda x: setattr(roop.globals, 'output_video_quality', int(x)), inputs=[output_video_quality])
        temp_frame_format.change(lambda x: setattr(roop.globals, 'temp_frame_format', x), inputs=[temp_frame_format])
        temp_frame_quality.change(lambda x: setattr(roop.globals, 'temp_frame_quality', int(x)), inputs=[temp_frame_quality])
        execution_threads.change(lambda x: setattr(roop.globals, 'execution_threads', int(x)), inputs=[execution_threads])
        execution_batch_size.change(lambda x: setattr(roop.globals, 'execution_batch_size', int(x)), inputs=[execution_batch_size])
        max_memory.change(lambda x: setattr(roop.globals, 'max_memory', int(x) if x > 0 else None), inputs=[max_memory])
        
        def update_execution_providers(providers):
            # Import here to avoid circular dependency
            import onnxruntime
            # Decode providers (e.g., 'cuda' -> 'CUDAExecutionProvider')
            decoded = []
            provider_map = {
                'cpu': 'CPUExecutionProvider',
                'cuda': 'CUDAExecutionProvider',
                'coreml': 'CoreMLExecutionProvider',
                'mps': 'MPSExecutionProvider',
                'directml': 'DmlExecutionProvider'
            }
            available = onnxruntime.get_available_providers()
            for p in providers:
                full_name = provider_map.get(p.lower())
                if full_name and full_name in available:
                    decoded.append(full_name)
            
            # Always fallback to CPU if nothing is available
            if not decoded:
                decoded = ['CPUExecutionProvider']
            
            roop.globals.execution_providers = decoded
            return f"Using: {', '.join(decoded)}"
        
        execution_provider.change(update_execution_providers, inputs=[execution_provider], outputs=[status_label])

        start_button.click(on_start, inputs=[output_path], outputs=[status_label, output_video, output_image])
        destroy_button.click(on_destroy)

    ui.launch(inbrowser=False, server_name="0.0.0.0", share=roop.globals.share)

def update_status(message: str) -> None:
    print(f"[UI] {message}")
