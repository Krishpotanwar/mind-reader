"""Detection rate of MediaPipe HandLandmarker on the TF rock-paper-scissors CGI images."""
import sys, time, pathlib, collections
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

model, root = sys.argv[1], pathlib.Path(sys.argv[2])
opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_hands=1)
det = vision.HandLandmarker.create_from_options(opts)
found, total, t0 = collections.Counter(), collections.Counter(), time.time()
for p in sorted(root.rglob("*.png")):
    label = p.parent.name
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    total[label] += 1
    found[label] += bool(res.hand_landmarks)
n = sum(total.values())
print({k: f"{found[k]}/{total[k]}" for k in total}, f"overall {sum(found.values())}/{n} = {sum(found.values())/n:.1%}", f"{(time.time()-t0)/n*1000:.0f} ms/img")
