from paddleocr import PaddleOCR
import cv2
import  numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import spacy
from spellchecker import SpellChecker
import yaml

def draw_ocr_box_txt(image,boxes,txts,scores=None,drop_score=0.5,font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        tuple_box = tuple(map(tuple, box))
        draw_left.polygon(tuple_box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    ret, frame = cap.read()
    w, h = frame.shape[1], frame.shape[0]
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps ,w, h

save_result = True
with open("param.yaml") as file:
    param = yaml.load(file , Loader = yaml.FullLoader)

lang = param["language"]
video_path = param["img_path"]

frames ,fps ,width ,height = get_video_frames(video_path)

video_writer = cv2.VideoWriter("output.avi",0,5,(2*width ,height))

spell = SpellChecker()
# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang=lang) # The model file will be downloaded automatically when executed for the first time

for i in range(len(frames)):
    if i % int(fps/2) == 0:
        result = ocr.ocr(frames[i])
        # Recognition and detection can be performed separately through parameter control
        # result = ocr.ocr(img_path, det=False)  Only perform recognition
        # result = ocr.ocr(img_path, rec=False)  Only perform detection
        #for line in result:
        #    print(line)

        # Visualization
        from PIL import Image
        #image = Image.open(Image_path).convert('RGB')
        image = Image.fromarray(frames[i])
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        misspelled = spell.unknown(txts)

        for i in range(len(txts)):
            if txts[i] in misspelled:
                txts[i] = spell.correction(txts[i])

        im_show = draw_ocr_box_txt(image, boxes, txts, scores, font_path='./latin.ttf')
        video_writer.write(im_show)
        cv2.imshow("image", im_show)
        if cv2.waitKey(10) & 0xFF == 27:
            break

cv2.destroyAllWindows()
video_writer.release()
'''
if save_result:
    im_show = Image.fromarray(im_show)
    im_show.save('result3.jpg') 
'''
