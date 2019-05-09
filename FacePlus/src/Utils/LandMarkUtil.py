'''
Created on 2018年7月7日

@author: IL MARE
'''
import requests
import json
import os

landMarkPoints=["mouth_left_corner","contour_right1","right_eyebrow_upper_left_quarter",
                "nose_left","left_eyebrow_upper_right_quarter","contour_right8","left_eye_top",
                "right_eyebrow_upper_middle","contour_chin","contour_right9","nose_contour_left2",
                "contour_right4","left_eye_lower_right_quarter","left_eyebrow_lower_middle",
                "right_eye_right_corner","left_eye_pupil","left_eye_bottom",
                "nose_contour_lower_middle","right_eye_upper_right_quarter",
                "nose_contour_left1","left_eye_right_corner","nose_contour_right2",
                "nose_contour_left3","right_eye_bottom","contour_left2","right_eye_center",
                "left_eye_left_corner","mouth_upper_lip_bottom","contour_right5",
                "contour_left7","mouth_lower_lip_bottom","nose_right",
                "mouth_lower_lip_left_contour2","left_eyebrow_lower_left_quarter",
                "contour_left5","mouth_upper_lip_top","right_eyebrow_lower_right_quarter",
                "mouth_upper_lip_right_contour3","mouth_lower_lip_left_contour1",
                "right_eyebrow_upper_right_quarter","right_eyebrow_right_corner",
                "left_eyebrow_right_corner","left_eyebrow_upper_middle",
                "right_eyebrow_lower_middle","mouth_upper_lip_left_contour3",
                "nose_tip","contour_left8","mouth_lower_lip_right_contour1",
                "left_eye_center","mouth_lower_lip_right_contour2",
                "mouth_lower_lip_right_contour3","nose_contour_right3",
                "right_eye_top","contour_left1","contour_right2","contour_right3",
                "right_eye_lower_right_quarter","right_eyebrow_lower_left_quarter",
                "mouth_upper_lip_right_contour1","contour_left3","mouth_lower_lip_top",
                "right_eye_upper_left_quarter","contour_right6","mouth_upper_lip_left_contour2",
                "right_eye_pupil","contour_left6","right_eye_lower_left_quarter",
                "left_eye_upper_right_quarter","right_eye_left_corner","mouth_right_corner",
                "contour_left4","left_eyebrow_lower_right_quarter","mouth_upper_lip_left_contour1",
                "left_eyebrow_left_corner","nose_contour_right1","contour_left9",
                "left_eye_upper_left_quarter","left_eyebrow_upper_left_quarter",
                "right_eyebrow_left_corner","contour_right7","mouth_upper_lip_right_contour2",
                "left_eye_lower_left_quarter","mouth_lower_lip_left_contour3"]

def generateLandMark(imagePath):
    try:
        Url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
        parameters = {
            "api_key": "DVTXIboHVgISfgkHXD77WfX2q609WFfe",
            "api_secret": "DRTArnHV2GHACBT1qXK3gip6Ub7Wn8UH",
            "return_landmark": 1
            }
        for filename in os.listdir(imagePath):
            if filename.endswith("jpg"):
                if os.path.exists(r"{0}{1}.txt".format(imagePath, filename[0 : 10])):
                    continue
                files = {
                    "image_file": open("{0}{1}".format(imagePath, filename), "rb")
                    }
                resp = requests.post(Url, timeout=15, data=parameters, files=files)
                obj = json.loads(resp.text)
                landMarks = []
                for name in landMarkPoints:
                    value = obj["faces"][0]["landmark"][name]
                    landMarks.append(str(value["x"]))
                    landMarks.append(str(value["y"]))
                try:
                    fp = open(r"{0}{1}.txt".format(imagePath, filename[0 : 10]), "w")
                    fp.write(" ".join(landMarks))
                    print(r"{0}{1}.txt".format(imagePath, filename[0: 10]))
                except Exception as e:
                    print(e)
                finally:
                    fp.close()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    imagePath = r"G:/python/sources/nwpu/dectImage/"
    generateLandMark(imagePath)