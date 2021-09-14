import typer
from pathlib import Path
from typing import Optional
import time
import cv2 as cv
from detection_service import FireDetector

detector = FireDetector()


def start_live_detection(confidence: int = 40):
    typer.echo("[+] Starting live camera...")
    cap = cv.VideoCapture(0)
    while True:
        ret, image_np = cap.read()
        proc_image, detections = detector.detect_from_array(
            image_np, confidence)
        if detections:
            # FIRE IS DETECTED
            # CALL SUITABLE API FOR SMS/CALL ETC
            for detection in detections:
                typer.echo("Probability: {} | BOX: {}".format(
                    detection["percentage_probability"],
                    detection["box_points"]))

        cv.imshow('FireDetection', proc_image)
        time.sleep(0.05)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            cap.release()
            raise typer.Abort()


def main(
        input_path: Optional[Path] = typer.Option(None, help='Input path'),
        confidence: int = typer.Option(40, help="Minimum probability")):
    if input_path is None:
        typer.echo("[+] Loading Model...")
        detector.load_image_detector()
        start_live_detection(confidence)
    elif input_path.is_file():
        start = time.time()
        if (
            any(input_path.name.endswith(e) for e in [".png", ".jpg", ".jpeg"])
        ):
            typer.echo("[+] Loading Model...")
            detector.load_image_detector()
            typer.echo("[+] Running detection on image")
            output_path = input_path.stem+"-processed.png"
            detector.detect_from_image(
                str(input_path), str(output_path), confidence=confidence)
        elif input_path.name.endswith(".mp4"):
            typer.echo("[+] Loading Model...")
            detector.load_video_detector()
            typer.echo("[+] Running detection on video")
            output_path = input_path.stem+"-processed"
            detector.detect_from_video(
                str(input_path), str(output_path), confidence=confidence)
        else:
            typer.echo("[!] Invalid File Type")
            raise typer.Abort()
        typer.echo("[-] Completed in  {} seconds".format(time.time()-start))
    else:
        typer.echo("[!] The input file doesn't exist")
        raise typer.Abort()


if __name__ == "__main__":
    typer.run(main)
