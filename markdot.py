import cv2, os, math
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog
import io
import base64
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from matplotlib.colors import ListedColormap

RUN_MODE = "flask"  # Change to "batch" for image Processing & "flask" for API server.


# Disable interactive mode
plt.ioff()
# Use 'Agg' backend for non-GUI rendering
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes
app.secret_key = 'secret_key'  # Required for session management
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

df1=pd.read_csv("position.csv")
thetal=df1["theta"].to_numpy()
rl=df1["r"].to_numpy()
cntrl=df1["count"].to_numpy()

#Function to remove grid surrounding contours.
def trax_noise4(image, labimg2, lower, upper):
    threshval=0
    h,w=image.shape[:2]
    labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    labimg[:, :, 0] = 0

    #turqoise color part mask
    lower_lab = np.array([0, 80, 99])
    upper_lab = np.array([0, 115, 130])
    maskc = cv2.inRange(labimg, lower_lab, upper_lab)
    maskc2=cv2.bitwise_not(maskc)

    mask1 = cv2.inRange(labimg2, lower, upper)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(maskc, connectivity=8)
    minx=[stats[i,0] for i in range(1,num_labels)]; miny=[stats[i,1] for i in range(1,num_labels)]
    maxx=[stats[i,0]+stats[i,2] for i in range(1,num_labels)]; maxy=[stats[i,1]+stats[i,3] for i in range(1,num_labels)]
    try:
        minvalx=min(minx); minvaly=min(miny); maxvalx=max(maxx); maxvaly=max(maxy)
        threshval=(maxvalx-minvalx)*(maxvaly-minvaly)
        width, height = w, h
        maskg1 = np.zeros((height, width), dtype=np.uint8)
        maskg1[minvaly:maxvaly, minvalx:maxvalx] = 255

        alist=[stats[i,4] for i in range(1,num_labels)]
        thresh1=max(alist)
        alist2=sorted(alist, reverse=True)
        id1=alist.index(alist2[0]); id2=alist.index(alist2[1])

        maxvalx=maxx[id1]; minvalx=minx[id1]; maxvaly=maxy[id1]; minvaly=miny[id1]
        delx = round(0.15*(maxvalx-minvalx)); dely=round(0.15*(maxvaly-minvaly))
        maskg2 = np.zeros((height, width), dtype=np.uint8)
        maskg2[minvaly:maxvaly, minvalx:maxvalx] = 255
        # maskg2[minvaly+dely:maxvaly-dely, minvalx+delx:maxvalx-delx] = 255
        maskg2=cv2.bitwise_not(maskg2)

        maxvalx=maxx[id2]; minvalx=minx[id2]; maxvaly=maxy[id2]; minvaly=miny[id2]
        delx = round(0.15*(maxvalx-minvalx)); dely=round(0.15*(maxvaly-minvaly))
        maskg3 = np.zeros((height, width), dtype=np.uint8)
        maskg3[minvaly:maxvaly, minvalx:maxvalx] = 255
        # maskg3[minvaly+dely:maxvaly-dely, minvalx+delx:maxvalx-delx] = 255
        maskg3=cv2.bitwise_not(maskg3)

        #masking to get only sticker part
        # mask1=cv2.bitwise_and(mask1, maskg1)
        maska1=cv2.bitwise_and(mask1, maskc2)
        #masking to get only grid part
        # mask1=cv2.bitwise_and(mask1, maskg2)
        # mask1=cv2.bitwise_and(mask1, maskg3)
        # mask1=cv2.bitwise_and(mask1, cv2.bitwise_not(cv2.inRange(labimg, (0,120,135), (0,128,155))))
    except:
        pass
    # cv2.imshow("Contours2", mask1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return threshval, mask1

#Identify sticker bounding box, crop, resize and rotate.
def imgready(image):
    h, w = image.shape[:2]
    labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    labimg[:, :, 0] = 0

    # turquoise color mask
    lower_lab = np.array([0, 80, 99])
    upper_lab = np.array([0, 115, 130])
    maskc = cv2.inRange(labimg, lower_lab, upper_lab)
    contours, _ = cv2.findContours(maskc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ❌ If turquoise not found, return resized image and False
    if not contours:
        print("❌ TraxIt not detected!")
        return cv2.resize(image, (1000, 1000)), False

    try:
        clist = [cnt.reshape(-1, 2) for cnt in contours]
        aval = [cv2.contourArea(cnt) for cnt in contours]
        coords = clist[aval.index(max(aval))]
        cx = [pt[0] for pt in coords]
        cy = [pt[1] for pt in coords]

        minx = min(cx)
        maxx = max(cx)
        miny = min(cy)
        maxy = max(cy)

        if abs(maxx - minx) > abs(maxy - miny):
            ymin = cy[cx.index(minx)]
            ymax = cy[cx.index(maxx)]
            xmin = minx
            xmax = maxx
        else:
            xmin = cx[cy.index(miny)]
            xmax = cx[cy.index(maxy)]
            ymin = miny
            ymax = maxy

        angrot = math.degrees(math.atan2((ymax - ymin), (xmax - xmin)))
        angle = -(90 - angrot) if angrot < 90 else (angrot - 90)

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale=1.0)
        image = cv2.warpAffine(image, M, (w, h))

        labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        labimg[:, :, 0] = 0
        maskc = cv2.inRange(labimg, lower_lab, upper_lab)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(maskc, connectivity=8)

        if num_labels <= 2:
            print("❌ TraxIt not detected after rotation!")
            return cv2.resize(image, (1000, 1000)), False

        stats = stats[1:]  # ignore background
        alist = [s[4] for s in stats]
        id1, id2 = sorted(range(len(alist)), key=lambda i: alist[i], reverse=True)[:2]

        minvalx = min(stats[id1][0], stats[id2][0])
        maxvalx = max(stats[id1][0] + stats[id1][2], stats[id2][0] + stats[id2][2])
        minvaly = min(stats[id1][1], stats[id2][1])
        maxvaly = max(stats[id1][1] + stats[id1][3], stats[id2][1] + stats[id2][3])

        # ✅ CHECK IF IMAGE IS TOO FAR
        sticker_area = (maxvalx - minvalx) * (maxvaly - minvaly)
        image_area = h * w
        area_ratio = sticker_area / image_area

        if area_ratio < 0.05:
            print(f"❌ Image too far! Sticker too small ({area_ratio*100:.2f}%).")
            return cv2.resize(image, (1000, 1000)), "too_far"


        image = image[minvaly:maxvaly, minvalx:maxvalx]
        image = cv2.resize(image, (1000, 1000))
    except:
        image = cv2.resize(image, (1000, 1000))
        return image, False

    return image, True

def process_image():
    # Open a file dialog to select an image
    Tk().withdraw()  # Hide the root window
    global image
    global fl
    while True:
        file_path = filedialog.askopenfilename(
            title="Select Image File", 
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            print("No file selected. Exiting.")
            exit()
        # print("filepath:", file_path)
        fl=file_path.split("/")[-1]
        # Load and enhance brightness for dim images
        
        image = Image.open(file_path)
        image = np.array(image)

def imgready2(image):
    lower_lab = np.array([0, 0, 0])
    upper_lab = np.array([70, 70, 70])
    maskc = cv2.inRange(image, lower_lab, upper_lab)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(maskc, connectivity=8)

    alist = [stats[i, 4] for i in range(1, num_labels)]
    if not alist:
        print("⚠️ No dark-gray regions found for cropping. Skipping imgready2.")
        return image  # Fallback to original image

    alist2 = sorted(alist, reverse=True)
    id1 = alist.index(alist2[0])

    minx = stats[id1 + 1, 0]
    miny = stats[id1 + 1, 1]
    maxx = stats[id1 + 1, 0] + stats[id1 + 1, 2]
    maxy = stats[id1 + 1, 1] + stats[id1 + 1, 3]

    image = image[miny:maxy, minx:maxx]
    h, w = image.shape[:2]

    if w > h:
        image = cv2.resize(image, (812, 404))
        angle = -90
    else:
        image = cv2.resize(image, (404, 812))
        angle = 0

    if angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (812, 404) if angle == -90 else (w, h))

    return image


# Function to check if an image is blurry
def is_blurry(image, threshold=100):
    """
    Checks if an image is blurry using the Laplacian Variance method.
    :param image: Input image (BGR format)
    :param threshold: Blur threshold (higher = more sensitive)
    :return: (True if blurry, Laplacian variance value)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    # print(variance)
    return variance < threshold, variance

#def detect_specular_reflection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    brightness = np.sqrt(
        0.241 * np.mean(image[:, :, 2]) ** 2 +
        0.691 * np.mean(image[:, :, 1]) ** 2 +
        0.068 * np.mean(image[:, :, 0]) ** 2
    )

    Ts = 40
    kv = 2.5
    Tv = brightness * kv

    if np.sum(v == 255) > (image.shape[0] * image.shape[1] // 3):
        Ts = 35
        Tv = 240

    mask = np.where((s < Ts) & (v > Tv), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel, iterations=1)

    return mask_clean   

def detect_specular_reflection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # ✅ Slightly more tolerant thresholds
    mask = cv2.inRange(hsv, (0, 0, 220), (180, 85, 255))  # S < 80, V > 230

    # ✅ Better kernel for glare blob merging
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # ✅ Allow smaller glare patches (≥ 80px instead of 200px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    final_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 30:  # reduced threshold to detect more valid glare
            final_mask[labels == i] = 255

    return final_mask


def filtered_contours(centx, centy):
    centx=np.array(centx); centy=np.array(centy)
    if len(centx)>1:
        refpoint=np.array([centx.max(),centy.max()])
        # refpoint = np.array([359,758])
        filtvals = np.array([np.linalg.norm(refpoint - np.array([centx[i], centy[i]])) for i in range(len(centx))])
        
        centx = centx[filtvals>50]
        centy = centy[filtvals>50]
        filtvals = filtvals[filtvals>50]
        filtvals2 = np.sort(filtvals)

        ids=[np.where(filtvals==ele)[0][0] for ele in filtvals2]
        centx = np.array([centx[ele] for ele in ids])
        centy = np.array([centy[ele] for ele in ids])

        delfiltvals = np.array([abs(filtvals2[i+1]-filtvals2[i]) for i in range(len(filtvals2)-1)])
        idxx=np.where(delfiltvals>2.25*delfiltvals.mean())[0]
        for ele in np.sort(idxx)[::-1]:
            centx=np.delete(centx,ele+1)
            centy=np.delete(centy,ele+1)

        refpoint2=np.array([centx[np.where(centy==centy.min())[0][0]],centy.min()])
        refpoint3=np.array([centx.min(), centy[np.where(centx==centx.min())[0][0]]])

        radval2=np.linalg.norm(refpoint - refpoint2)
        radval3=np.linalg.norm(refpoint - refpoint3)
        theta2=round(math.degrees(math.atan2(abs(refpoint2[1]-refpoint[1]), abs(refpoint2[0]-refpoint[0]))),2)
        theta3=round(math.degrees(math.atan2(abs(refpoint3[1]-refpoint[1]), abs(refpoint3[0]-refpoint[0]))),2)

        if theta2>90:theta2=89
        if theta3>90:theta3=89

        distl1=[np.linalg.norm(np.array([radval2, theta2]) - np.array([rl[i],thetal[i]])) for i in range(len(rl))]
        distl2=[np.linalg.norm(np.array([radval3, theta3]) - np.array([rl[i],thetal[i]])) for i in range(len(rl))]
        cntr1=cntrl[distl1.index(min(distl1))]
        cntr2=cntrl[distl2.index(min(distl2))]
        out=[cntr1,cntr2]
        maxcntr=max(out)
        sel=[(round(radval2),round(theta2)), (round(radval3),round(theta3))][out.index(maxcntr)]
        # maxcntr+=1
    else:
        maxcntr=len(centx)
        sel=(359,758)
    return maxcntr,sel


#Filter mask contours and provide count.
def getcount(mask1):
    h,w=mask1.shape[:2]
    # mask1=mask1[round(0.1*h):round(h*0.965), 0:w]

    # cv2.imshow("separated", cv2.resize(mask1, (800,800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    threshdist=28
    areathresh=1963

    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntr=0; cx=[]; cy=[]
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Avoid division by zero
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        # 
        if (circularity > 0.2) and (area<areathresh*2.85) and (area>areathresh*0.3):
            if area>areathresh*2:
                cntr+=2
            else:
                cntr+=1
            M = cv2.moments(cnt)

            # Avoid division by zero
            if M["m00"] == 0:
                continue

            # Centroid coordinates
            cx.append(int(M["m10"] / M["m00"]))
            cy.append(int(M["m01"] / M["m00"]))
            # print(alist1, len(alist1))
    
    return cntr, cx, cy

    # process_image()
    # rawp="D:/Nayan/Traxit/TraxIt_IRL/752025"
    # allf=os.listdir(rawp)
    # for fl in allf:
    #     pathin=rawp+"/"+fl
    #     image=cv2.imread(pathin)
    
def run_temperature_analysis(image):
    # 🔍 Check if image is blurry
    
    imagec=image

    h,w=image.shape[:2]
    labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(labimg)
    lvald = len(l[(l<50) & (l>20)])*100/h/w
    lvall = len(l[l>200])*100/h/w
    if lvald>30 and lvall<5:
        appstr="Dark room"
    else:
        appstr=""

    image, traxit_status = imgready(image)

    image = imgready2(image)

    is_blur, blur_value = is_blurry(image, threshold=5)
    if is_blur:
        blur_msg = f"Blurry (Level: {blur_value:.2f})"
        print(f"⚠️ {blur_msg}")
        return {
        "temperature": "N/A",
        "error": "Image is too blurry. Retake image.",
        "blur": blur_msg,
        "image_base64": ""
        }   
    else:
        blur_msg = f"Clear (Level: {blur_value:.2f})"
        print(f"✅ {blur_msg}")

    if traxit_status == "too_far":
        print("⚠️ Skipping: Image taken from too far.")
        return {
            "temperature": "N/A",
            "error": "Image taken from too far. Retake image.",
            "blur": blur_msg,
            "image_base64": "",
            "log": "Image too far"
        }

    if traxit_status is False:
        print("⚠️ Skipping: TraxIt not detected.")
        return {
            "temperature": "N/A",
            "error": "TraxIt not detected",
            "blur": blur_msg,
            "image_base64": "",
            "log": "TraxIt not detected"
        }
    
    mask = detect_specular_reflection(image)
    reflection_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    reflection_ratio = round(reflection_pixels / total_pixels*100,1)

    # Optional visualization (can be removed if not running interactively)
    # if RUN_MODE == "batch":
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #     ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     ax[0].set_title("Original Image")
    #     ax[0].axis("off")

    #     ax[1].imshow(mask, cmap='gray')
    #     ax[1].set_title("Reflection Mask")
    #     ax[1].axis("off")
    #     plt.tight_layout()
    #     plt.show()

    final_temp = "NA"
    img_base64 = ""
    reflection_msg = ""

    if reflection_ratio < 10 and appstr=="":

        #create mask1 for green shade
        lower_lab = np.array([0, 70, 120])
        upper_lab = np.array([0, 118, 160])

        labimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        labimg2[:, :, 0] = 0
        threshval, mask1 = trax_noise4(image, labimg2, lower_lab, upper_lab)
        alist1=getcount(mask1)[0]

        #create mask11 for green shade3
        lower_lab = np.array([0, 70, 120])
        upper_lab = np.array([0, 100, 160])

        threshval, mask12 = trax_noise4(image, labimg2, lower_lab, upper_lab)
        alist12=getcount(mask12)[0]
        if (alist12-alist1)>1:
            alist1=alist12; mask1=mask12

        #create mask11 for green shade2
        lower_lab = np.array([0, 70, 120])
        upper_lab = np.array([0, 123, 150])

        threshval, mask11 = trax_noise4(image, labimg2, lower_lab, upper_lab)
        alist11=getcount(mask11)[0]
        if (alist11-alist1)>2:
            mask1=mask11

        #create mask2 for yellow 
        lower_lab = np.array([0, 70, 150])
        upper_lab = np.array([0, 120, 185])
        threshval, mask2=trax_noise4(image, labimg2, lower_lab, upper_lab)

        #create mask3 for orange
        lower_lab = np.array([0, 115, 134])
        upper_lab = np.array([0, 145, 160])
        threshval, mask3=trax_noise4(image, labimg2, lower_lab, upper_lab)

        #create mask3 for orange2
        lower_lab = np.array([0, 122, 130])
        upper_lab = np.array([0, 126, 135])
        threshval, masko2=trax_noise4(image, labimg2, lower_lab, upper_lab)

        #create single mask green
        lower_lab = np.array([0, 0, 138])
        upper_lab = np.array([0, 0, 170])
        labimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        labimg2[:, :, 0] = 0; labimg2[:, :, 1] = 0
        threshval, mask4=trax_noise4(image, labimg2, lower_lab, upper_lab)

        #create single mask yellow
        lower_lab = np.array([0, 70, 0])
        upper_lab = np.array([0, 107, 0])
        labimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        labimg2[:, :, 0] = 0; labimg2[:, :, 2] = 0
        threshval, mask5=trax_noise4(image, labimg2, lower_lab, upper_lab)

        maskall=cv2.bitwise_or(cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3)), masko2)
        mskl=[maskall, cv2.bitwise_or(cv2.bitwise_or(mask2, mask3), masko2), cv2.bitwise_or(cv2.bitwise_or(mask1, mask3), masko2), mask1, mask4, mask5]
        # lighted_count=max([len(getcount(maskall)), len(alist1)+len(alist3), len(alist2)+len(alist3)])

        lighted_count=[getcount(ele)[0] for ele in mskl]
        idxx = lighted_count.index(max(lighted_count))
        fmask=mskl[idxx]

        lcntr,cx,cy = getcount(fmask)
        
        # ==================================================================
        # == THIS IS THE NEW CODE TO DRAW RED DOTS ON THE COUNTED POINTS  ==
        # ==================================================================
        for i in range(len(cx)):
            # Draw a filled red circle with a radius of 10 pixels
            cv2.circle(image, (cx[i], cy[i]), 10, (0, 0, 255), -1)
        # ==================================================================

        maxcntr1,sel = filtered_contours(cx,cy)
        maxcntr=max([lcntr, maxcntr1])
        # brfg=bright_detect(fmask)
        # NEW - HSV based reflection mask
        # NEW - HSV based reflection mask

        print("✅ No significant reflection detected.")
        reflection_msg = "N/A"

        #Temperature calculation
        tempout=round(104.8-maxcntr*0.2,1)

        # cv2.imshow("Contours", image)
        # cv2.imshow("Contours2", cv2.resize(cv2.bitwise_or(mask1, mask3), (800,800)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  

        # print("&&&&&&&&&&&&&&&&&&&&", lighted_count)
        #selecting temperature and printing on output image and saving the image
        fmask=cv2.hconcat([image, cv2.merge([fmask, fmask, fmask])])
        image = imagec
        h,w=image.shape[:2]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 2
        font_color = (130, 30, 219)
        thickness = 3

        if tempout < 94.0:
            final_temp = "Below 94.0"
        else:
            final_temp = str(tempout)
        # Draw text on image
        image_annotated = cv2.putText(fmask, f"T:{final_temp} {maxcntr1}", (int(w/35), int(h/24)), font, font_scale, font_color, thickness)

        # Encode to PNG
        _, img_encoded = cv2.imencode('.png', image_annotated)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    if appstr!="":
        print(f"⚠️ Dark image detected! Retake image. ({lvald:.2f}% of image)")
        reflection_msg = f"Darkness detected ({lvald:.2f}%)"

    if reflection_ratio>10:
        print(f"⚠️ Reflection detected! Retake image. ({reflection_ratio:.2f}% of image)")
        reflection_msg = f"Reflection detected ({reflection_ratio:.2f}%)"
        
    return {
    "temperature": final_temp,
    "error": reflection_msg,
    "blur": blur_msg,
    "image_base64": img_base64,
    "image": image_annotated 
    }

@app.route('/', methods=['POST'])
def index():
    if 'file' not in request.files or not request.files['file']:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    image = cv2.imread(file_path)

    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    # ✅ Resize image to 2000x1000 (Width x Height)
    image = cv2.resize(image, (800, 600))
    
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    result = run_temperature_analysis(image)

    # ✅ Save the processed image
    output_path = os.path.join("outputs", f"{os.path.splitext(filename)[0]}_output.png")
    os.makedirs("outputs", exist_ok=True)
    if "image" in result and result["image"] is not None:
        cv2.imwrite(output_path, result["image"])

    # ✅ Return JSON without the raw image to keep response light
    result.pop("image", None)
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'running', 'message': 'Server is up'}), 200

if __name__ == '__main__':
    if RUN_MODE == "flask":
        app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)


    elif RUN_MODE == "batch":
        from tkinter import Tk, filedialog
        Tk().withdraw()
        rawp = filedialog.askdirectory(title="Select Folder with Images")
        #allf = os.listdir(rawp)
        supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.BMP')
        allf = [f for f in os.listdir(rawp) if f.lower().endswith(supported_ext)]

        for fl in allf:
            pathin = os.path.join(rawp, fl)
            image = cv2.imread(pathin)
            if image is None:
                print(f"Skipping {fl} (could not read image)")
                continue

            result = run_temperature_analysis(image)

            # ✅ Skip and notify if TraxIt not detected
            if 'TraxIt not detected' in result.get('log', ''):
                print(f"{fl} ⚠️ TraxIt not detected. Retake image.")
                continue

            # ✅ Skip and notify if too much reflection
            if result is None or 'Reflection detected' in result.get('error', ''):
                print(f"{fl} ⚠️ Image is reflective. Retake image.")
                continue

            # ✅ Log result
            print(f"{fl} ➤ Temp: {result['temperature']} | Error: {result['error']} | Blur: {result['blur']}")

            # ✅ Save output image
            img_output_path = os.path.join("outputs", f"{os.path.splitext(fl)[0]}_output.png")
            if not os.path.exists("outputs"):
                os.makedirs("outputs")

            # print("#########################", result["image_base64"])
            if result["image_base64"] != "":
                with open(img_output_path, "wb") as f:
                    f.write(base64.b64decode(result["image_base64"]))