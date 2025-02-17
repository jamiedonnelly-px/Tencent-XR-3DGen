import cv2
import os
import numpy as np

import pathlib

# load full


dump_path = "tripo/legs"
pathlib.Path(dump_path).mkdir(exist_ok=True)

img_path = "tripo/ref_image.png"
ref_image = cv2.imread( img_path , -1 )
h, w, _ = ref_image.shape

start = None
end = None

def click(event, x, y, flags, param):
    # grab references to the global variables
    global start, end
    if event == cv2.EVENT_LBUTTONDOWN:
        start = [x, y]
    if event == cv2.EVENT_MBUTTONDOWN:
        end = [x, y]





cv2.namedWindow("correspondence", 0)
cv2.resizeWindow("correspondence", w, h)
cv2.setMouseCallback("correspondence", click)

# cv2.setMouseCallback('depth', click)

# cv2.namedWindow( "selector" , 0 )
# cv2.resizeWindow("selector", width*2, height)
# cv2.setMouseCallback("selector", click)


while True:

    correspondence = np.zeros_like(ref_image[..., :3])


    if start != None and  end!= None:
        cv2.rectangle(correspondence, start, end, color=(255,0,0), thickness=2)


    correspondence = correspondence + ref_image[..., :3]

    cv2.imshow("correspondence", correspondence)



    # print pairs
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):

        img_path = os.path.join(dump_path, "img.png")
        crop = ref_image[  start[1]: end[1], start[0]: end[0] ]
        cv2.imwrite( img_path, crop)
        array = np.asarray( [start, end], dtype=int)
        np.savetxt(os.path.join(dump_path, "crop.txt"), array)

    # elif key == ord('b'):  # Roll back one correspondence (to correct mis-click)
    #     if len(pairs) > 0:
    #         pairs = pairs[:-1]
    #         print("pairs", pairs)
    # elif key == ord('d'):
    #     break

cv2.destroyAllWindows()