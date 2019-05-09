'''
Created By ILMARE
@Date 2019-3-1
'''

from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import requests
from PIL import Image
import os
import re

totalCount = 0
pre_path = r"/home/ilmare/Desktop/FaceReplace/data/image/"

class PhotoScrawler:
    def __init__(self, savePath, destUrl, maxPage):
        self._savePath = savePath
        self._destUrl = destUrl
        self._maxPage = maxPage
    def get_title_list(self, destUrl, pageNum=0):
        Url = "{0}&ie=utf-8&pn={1}".format(destUrl, pageNum * 50)
        print("Parsing page: ", Url)
        try:
            resp = requests.get(Url)
            bsObj = BeautifulSoup(resp.text, "html.parser")
            elts = bsObj.find_all("li", {"class": ["j_thread_list", "clearfix"]})
            print(len(elts))
            return_mat = []
            for elt in elts:
                repNum = int(elt.find("span", {"class": "threadlist_rep_num center_text"}).text)
                a = elt.find("a", {"class": "j_th_tit"})
                link = a.attrs.get("href")
                title = a.attrs.get("title")
                return_mat.append((title, "{0}{1}".format("http://tieba.baidu.com", link), repNum))
            return return_mat
        except Exception as e:
            print(e)
            return None
    def parse_page(self, fronted_Url, pageNum=1):
        Url = "{0}?pn={1}".format(fronted_Url, pageNum)
        global totalCount
        try:
            resp = requests.get(Url)
            bsObj = BeautifulSoup(resp.text, "html.parser")
            ul = bsObj.find("ul", {"class": "l_posts_num"})
            totalPage = int(ul.find("li", {"class": "l_reply_num"}).find_all("span", {"class": "red"})[1].text)
            print("----", "Parsing page: ", Url, ", pageNum: ", pageNum, ", totalPage: ", totalPage)
            elts = bsObj.find_all("div", {"class": ["l_post", "j_l_post", "l_post_bright", "noborder"]})
            for elt, idx in zip(elts, range(len(elts))):
                div = elt.find("div", {"class": "d_post_content j_d_post_content clearfix"})
                imgs = div.find_all("img")
                if imgs is not None:
                    for img in imgs:
                        src = img.attrs.get("src")
                        res = re.match(r"^http.*/(image_emoticon)[0-9]+.(png|jpg|jpeg|gif)$", src)
                        if res is None:
                            ret = re.search(r"(?<=\.)(png|jpg|jpeg|gif)$", src)
                            format = None
                            if ret is not None:
                                format = ret.group()
                            if format is None:
                                urlretrieve(src, "{0}{1}".format(pre_path, totalCount))
                                img = Image.open("{0}{1}".format(pre_path, totalCount))
                                format = img.format
                                img.save("{0}{1}.{2}".format(pre_path, totalCount, format.lower()))
                                os.remove("{0}{1}".format(pre_path, totalCount))
                                print("-------- ", idx, ": ", src)
                            else:
                                urlretrieve(src, "{0}{1}.{2}".format(pre_path, totalCount, format))
                                print("-------- ", idx, ": ", src, "format: ", format)
                            totalCount += 1
        except Exception as e:
            print(e)
        finally:
            if pageNum < totalPage:
                self.parse_page(fronted_Url, pageNum + 1)
            else:
                return
    def get_photo_from_tieba(self):
        for i in range(self._maxPage):
            return_mat = self.get_title_list(self._destUrl, i)
            if return_mat is None:
                continue
            for (title, link, repNum), page in zip(return_mat, range(len(return_mat))):
                if repNum <= 3000:
                    print("===>", title, ", current page: ", i + 1, ", current item: ", page)
                    self.parse_page(link)

if __name__ == "__main__":
    obj = PhotoScrawler(pre_path,
                  "http://tieba.baidu.com/f?kw=%E6%9D%A8%E5%B9%82", 1)
    obj.get_photo_from_tieba()