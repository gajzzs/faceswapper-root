import argparse
import cv2
import roop.globals
from roop.face_analyser import get_many_faces

def main():
    parser = argparse.ArgumentParser(description="Extract face details from an image using Roop's face analyser.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    args = parser.parse_args()

    # Initialize roop globals for execution providers
    roop.globals.execution_providers = ['cpu'] # For consistency, using CPU

    # Load the image
    frame = cv2.imread(args.image_path)
    if frame is None:
        print(f"Error: Could not read image from {args.image_path}")
        return

    # Get many faces
    many_faces = get_many_faces(frame)

    if many_faces:
        print(f"Found {len(many_faces)} face(s) in {args.image_path}:")
        for i, face in enumerate(many_faces):
            print(f"\n--- Face {i+1} ---")
            print(f"  Bounding Box (bbox): {face.bbox.astype(int).tolist()}")
            print(f"  Keypoints (kps): {face.kps.astype(int).tolist()}")
            if hasattr(face, 'gender'):
                print(f"  Gender: {face.gender}")
            if hasattr(face, 'age'):
                print(f"  Age: {face.age}")
            # You can add more attributes if needed, e.g., 'det_score', 'embedding'
            # print(f"  Detection Score: {face.det_score}")
            # print(f"  Embedding (first 5 values): {face.embedding[:5].tolist()}...")
    else:
        print(f"No faces found in {args.image_path}.")

if __name__ == '__main__':
    main()

