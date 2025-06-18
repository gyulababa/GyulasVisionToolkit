import argparse
import cv2
import os
import yaml
from scr.utils.pipeline_builder import build_pipeline
from scr.coreclasses.video.video_frame_grabber import VideoFrameGrabber

def run_image_mode(pipeline, input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Failed to load image: {input_path}")
        return

    results, visual_img = pipeline.process(img)

    for idx, res in enumerate(results):
        print(f"[{idx+1}] Emotion: {res['emotion']}")
        if "age" in res and "gender" in res:
            print(f"    Age: {res['age']} | Gender: {res['gender']}")

    cv2.imwrite(output_path, visual_img)
    print(f"✅ Output saved to {output_path}")

def run_video_mode(pipeline, args):
    os.makedirs(args.output, exist_ok=True)

    grabber = VideoFrameGrabber(
        video_path=args.input,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        batch_size=args.batch_size,
        batch_skip=args.batch_skip,
        queue_size=args.queue_size
    )

    for batch_idx, frame_batch in enumerate(grabber):
        for frame_idx, frame in enumerate(frame_batch):
            results, visual_img = pipeline.process(frame)

            out_path = os.path.join(args.output, f"frame_{batch_idx:05d}_{frame_idx}.jpg")
            cv2.imwrite(out_path, visual_img)
            print(f"✅ Saved: {out_path}")

def load_run_settings(path="scr/configs/run_settings.yaml"):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --- Future scaffold: Folder mode ---
# def run_folder_mode(pipeline, input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
#             run_image_mode(pipeline, input_path, output_path)

# --- Future scaffold: Stream mode ---
# def run_stream_mode(pipeline, stream_source=0):
#     cap = cv2.VideoCapture(stream_source)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results, visual_img = pipeline.process(frame)
#         cv2.imshow("Live Stream", visual_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Face Emotion Detection CLI")

    parser.add_argument('--mode', choices=['image', 'video'], default='image',
                        help="Mode: image or video")

    parser.add_argument('--input', required=True, help="Input file (image or video)")
    parser.add_argument('--output', default="test_output/", help="Output folder or file")
    parser.add_argument('--config', default="config_default.yaml", help="Config file inside configs/main/")

    # Video frame grabber parameters
    parser.add_argument('--skip_frames', type=int, default=0, help="Skip N frames after each frame")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of frames per batch")
    parser.add_argument('--batch_skip', type=int, default=0, help="Skip N batches after each batch")
    parser.add_argument('--start_frame', type=int, default=0, help="Start from frame number")
    parser.add_argument('--max_frames', type=int, default=None, help="Limit total frames processed")
    parser.add_argument('--queue_size', type=int, default=32, help="Prefetch queue size")


    # --- Future flags ---
    # parser.add_argument('--preview', action='store_true', help="Show live preview window")
    # parser.add_argument('--save_faces', action='store_true', help="Save cropped face images")
    # parser.add_argument('--skip', type=int, default=0, help="Frame skip interval (video)")
    # parser.add_argument('--max_frames', type=int, default=0, help="Limit frames processed")
    # --- Future flags scaffold ---
    # parser.add_argument('--stream_url', help="Stream source for live mode")
    # parser.add_argument('--benchmark', action='store_true', help="Run inference benchmark mode")

    args = parser.parse_args()

    # Build pipeline from config
    pipeline = build_pipeline(args.config)

    if args.mode == "image":
        input_filename = os.path.basename(args.input)
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"processed_{input_filename}")

        run_image_mode(pipeline, args.input, output_path)

    elif args.mode == "video":
        run_video_mode(pipeline, args)

if __name__ == "__main__":
    main()
