/**
 * Webcam head tracking (plan 9.9) with MediaPipe's BlazeFace detector,
 * loaded on demand from a CDN (nothing is bundled or uploaded; the video
 * never leaves the browser).
 *
 * Geometry: a pinhole camera with horizontal field of view `fovDeg` has
 * focal length f = (W/2) / tan(fov/2) pixels. An adult face is ~0.15 m
 * wide, so the distance is d = f * 0.15 / w_px, and the lateral offset of
 * the face centre at pixel u is x = d * (u - W/2) / f. The camera sits
 * above the screen centre, which is the midpoint of the laptop speakers.
 */

export interface HeadPose {
  x: number; // metres, + = to the user's right as seen by the camera
  y: number; // metres from the screen plane
  confidence: number;
}

const TASKS_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

interface Detector {
  detectForVideo(v: HTMLVideoElement, t: number): { detections: { boundingBox?: { originX: number; width: number; originY: number; height: number }; categories?: { score: number }[] }[] };
  close(): void;
}

export function poseFromBox(box: { originX: number; width: number }, videoWidth: number, fovDeg = 62, faceWidthM = 0.15): HeadPose {
  const f = videoWidth / 2 / Math.tan((fovDeg * Math.PI) / 360);
  const d = (f * faceWidthM) / Math.max(1, box.width);
  const u = box.originX + box.width / 2;
  // Mirror: camera image is left-right flipped relative to the user.
  const x = (-d * (u - videoWidth / 2)) / f;
  return { x, y: d, confidence: 1 };
}

export class HeadTracker {
  private detector: Detector | null = null;
  private stream: MediaStream | null = null;
  private raf = 0;
  video: HTMLVideoElement | null = null;
  onPose: (p: HeadPose | null) => void = () => {};

  async start(video: HTMLVideoElement): Promise<void> {
    this.video = video;
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = this.stream;
    video.muted = true;
    await video.play();
    const vision = await import(/* @vite-ignore */ `${TASKS_URL}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${TASKS_URL}/wasm`);
    this.detector = (await vision.FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.5,
    })) as Detector;
    const loop = () => {
      if (!this.detector || !this.video) return;
      if (this.video.readyState >= 2) {
        const res = this.detector.detectForVideo(this.video, performance.now());
        const det = res.detections[0];
        if (det?.boundingBox) {
          const pose = poseFromBox(det.boundingBox, this.video.videoWidth);
          pose.confidence = det.categories?.[0]?.score ?? 1;
          this.onPose(pose);
        } else this.onPose(null);
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    cancelAnimationFrame(this.raf);
    this.detector?.close();
    this.detector = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}
