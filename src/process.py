import os
import difflib
import json
import torch
import numpy as np
import cv2
import logging
import yaml
from pathlib import Path
from ultralytics import YOLOv10

# Get the current directory path
current_directory = os.getcwd()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get settings from YAML
RUNNINGINDOCKER = os.getenv("RUNNINGINDOCKER", False)
RESULTS_FILE_OUTPUT_PATH = config.get("results_file_output_path", "surgical-tools.json")
VIDEO_TESTING_PATH = config.get("video_testing_path", "vid_1_short.mp4")
IOU_SCORE = config.get("iou_score", 0.5)
MODEL_FOLDER = config.get("model_folder", "models")
CONF_SCORE = config.get("conf_score", 0.5)

# Load all model paths from the folder
model_paths = [str(p) for p in Path(MODEL_FOLDER).rglob("*.pt")]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Execute in Docker" if RUNNINGINDOCKER else "[NOT] Execute in Docker")

def list_files(startpath: str) -> None:
    """
    Recursively lists all files and directories starting from the given path.

    Args:
        startpath (str): The root directory to start listing files from.
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        logging.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logging.info(f"{subindent}{f}")

class VideoLoader:
    """
    Base class for loading video files.
    """

    def load(self, fname: str) -> dict:
        """
        Loads a video file.

        Args:
            fname (str): The path to the video file.

        Returns:
            dict: A dictionary containing the path of the loaded video.

        Raises:
            IOError: If the file cannot be loaded.
        """
        path = Path(fname)
        if not path.is_file() or not str(path).endswith('mp4'):
            raise IOError(f"Could not load {fname} using {self.__class__.__qualname__}.")
        return {"path": fname}

class SurgToolLocDet(VideoLoader):
    """
    Class for surgical tool localization and detection in video frames.
    """

    def __init__(self):
        """
        Initializes the SurgToolLocDet class, sets up paths, loads the models, 
        and prepares output directories.
        """
        self._case_results = []
        self._input_path = Path("/input/") if RUNNINGINDOCKER else Path("../test/")
        self._output_path = Path("/output/") if RUNNINGINDOCKER else Path("../output/")
        self._output_file = self._output_path / RESULTS_FILE_OUTPUT_PATH
        self.models = [YOLOv10(model_path).to(device) for model_path in model_paths]
        
        # List of possible tools (reference for class matching)
        self.tool_list = [
            "needle_driver", "monopolar_curved_scissor", "force_bipolar", "clip_applier",
            "tip_up_fenestrated_grasper", "cadiere_forceps", "bipolar_forceps", "vessel_sealer",
            "suction_irrigator", "bipolar_dissector", "prograsp_forceps", "stapler",
            "permanent_cautery_hook_spatula", "grasping_retractor"
        ]
        
        # Mapping model classes to known tool classes
        self.model_classes = []
        for model in self.models:
            model_class = [
                str(classes).lower().replace('-', '_').replace(' ', '_') 
                for classes in model.names.values()
            ]
            model_class = [difflib.get_close_matches(tool, self.tool_list)[0] for tool in model_class]
            self.model_classes.append(model_class)

        # Create the directory for saving predicted frames if it doesn't exist
        self.predicted_dir = self._output_path / "predicted"
        self.predicted_dir.mkdir(parents=True, exist_ok=True)

    def process_case(self, *, case: str) -> None:
        """
        Processes a video case, making predictions for each frame.

        Args:
            case (str): The filename of the video case to process.
        """
        video_loaded = self.load(fname=os.path.join(self._input_path, case))
        scored_candidates = self.predict(video_loaded["path"])
        self._case_results.append(
            dict(
                type="Multiple 2D bounding boxes",
                boxes=scored_candidates,
                version={"major": 1, "minor": 0}
            )
        )

    def save(self) -> None:
        """
        Saves the prediction results to a JSON file.
        """
        with open(self._output_file, "w") as f:
            json.dump(self._case_results[0], f, default=self.convert_to_serializable)

    @staticmethod
    def convert_to_serializable(obj: object) -> float:
        """
        Converts non-serializable objects to serializable types.

        Args:
            obj (object): The object to convert.

        Returns:
            float: A serializable representation of the object.

        Raises:
            TypeError: If the object cannot be converted.
        """
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    @staticmethod
    def is_black_frame(frame: np.ndarray, threshold: int = 30) -> bool:
        """
        Determines if a frame is black based on the mean pixel value.

        Args:
            frame (np.ndarray): The video frame to check.
            threshold (int): The threshold below which a frame is considered black.

        Returns:
            bool: True if the frame is black, False otherwise.
        """
        return np.mean(frame) < threshold

    def nms(self, boxes, scores, iou_threshold):
        """
        Performs Non-Maximum Suppression (NMS) on bounding boxes.

        Args:
            boxes (list): List of bounding boxes.
            scores (list): List of confidence scores for each bounding box.
            iou_threshold (float): The IoU threshold for NMS.

        Returns:
            list: Indices of the boxes to keep after NMS.
        """
        keep = []
        idxs = np.argsort(scores)[::-1]

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)

            if len(idxs) == 1:
                break

            ious = []
            for j in idxs[1:]:
                x1 = max(boxes[i][0], boxes[j][0])
                y1 = max(boxes[i][1], boxes[j][1])
                x2 = min(boxes[i][2], boxes[j][2])
                y2 = min(boxes[i][3], boxes[j][3])

                inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                box1_area = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1)
                box2_area = (boxes[j][2] - boxes[j][0] + 1) * (boxes[j][3] - boxes[j][1] + 1)

                iou = inter_area / float(box1_area + box2_area - inter_area)
                ious.append(iou)

            ious = np.array(ious)
            idxs = idxs[1:][ious < iou_threshold]

        return keep


    def predict(self, fname: str) -> list:
        """
        Predicts bounding boxes for tools in each frame of the video.

        Args:
            fname (str): The path to the video file.

        Returns:
            list: A list of prediction results for each frame.
        """
        cap = cv2.VideoCapture(fname)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_frames_predicted_outputs = []
        for fid in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue

            if self.is_black_frame(frame):
                continue  # Skip black frames

            all_boxes = []
            all_scores = []
            all_class_ids = []
            for model_idx, model in enumerate(self.models):
                results = model.predict(frame, imgsz=[512, 640], conf=CONF_SCORE)
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        score = box.conf.cpu().item()
                        class_id = int(box.cls.cpu().item())
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(score)
                        all_class_ids.append((model_idx, class_id))

            if len(all_boxes) > 0:
                all_boxes = np.array(all_boxes)
                all_scores = np.array(all_scores)

                # Apply Non-Maximum Suppression (NMS)
                keep_indices = self.nms(all_boxes, all_scores, iou_threshold=IOU_SCORE)
                
                frame_predictions = []
                for idx in keep_indices:
                    x1, y1, x2, y2 = all_boxes[idx]
                    score = all_scores[idx]
                    model_idx, class_id = all_class_ids[idx]
                    classes = self.model_classes[model_idx][class_id]

                    bbox = [
                        [x1, y1, 0.5],
                        [x2, y1, 0.5],
                        [x2, y2, 0.5],
                        [x1, y2, 0.5]
                    ]
                    prediction_dict = {
                        "corners": bbox,
                        "name": f"slice_nr_{fid}_{classes}",
                        "probability": float(score)
                    }
                    frame_predictions.append(prediction_dict)

                filtered_predictions = frame_predictions
                all_frames_predicted_outputs.extend(filtered_predictions)

                if not RUNNINGINDOCKER:
                    # Draw bounding box and label on the frame
                    for prediction in filtered_predictions:
                        x1, y1, x2, y2 = prediction["corners"][0][0], prediction["corners"][0][1], \
                                         prediction["corners"][2][0], prediction["corners"][2][1]
                        classes = prediction["name"]
                        score = prediction["probability"]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{classes} {score:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                if not RUNNINGINDOCKER:
                    # Save the frame as an image in the 'predicted' directory
                    frame_output_path = self.predicted_dir / f"frame_{fid:04d}.png"
                    cv2.imwrite(str(frame_output_path), frame)

        cap.release()
        return all_frames_predicted_outputs

if __name__ == "__main__":
    logging.info(f"Directory structure of {current_directory}:")
    list_files(current_directory)
    detector = SurgToolLocDet()
    detector.process_case(case=VIDEO_TESTING_PATH)
    detector.save()
