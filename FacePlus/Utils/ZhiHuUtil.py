'''
Created on 2017年5月4日

@author: IL MARE
'''
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from urllib.request import urlretrieve 
import re
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
import os

def getAllFriendList(broswer, userID, pageNum, friendSet, currentLevel):
    try:
        broswer.get('https://www.zhihu.com%s?page=%s' % (userID, pageNum))
        WebDriverWait(broswer, 10).until(
            expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, '.UserLink-link')))
    except:
        print('getAllFriendList异常')
    else:
        bsObj = BeautifulSoup(broswer.page_source, 'html.parser')
        elts = bsObj.findAll('a', {'class':'UserLink-link'})
        for elt in elts:
            img = elt.find('img')
            if img:
                friendSet.add(elt)
                print('......*' * currentLevel, 'https://www.zhihu.com%s' % (elt.attrs.get('href', 'no data')))

def getFriendList(broswer, userID, currentLevel=1):
    try:
        if currentLevel > totalLevel:
            return
        if userID == 'no data':
            raise Exception()
        nameTemp = userID.split('/')[2]
        if not nameTemp in alreadyParse:
            alreadyParse.add(nameTemp)
        else:
            return
        print('......*' * currentLevel ,'正在解析用户：', nameTemp, '知乎首页：https://www.zhihu.com%s' % (userID), sep=' ')
        friendSet = set()
        broswer.get('https://www.zhihu.com%s' % (userID))
        WebDriverWait(broswer, 10).until(
            expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, '.UserLink-link')))
        elt = WebDriverWait(broswer, 10).until(
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, '.Avatar.Avatar--large.UserAvatar-inner')))
        res = re.match('^(https://.*)[0-9]x$', elt.get_attribute('srcset'))
        if res:
            if not nameTemp in alreadyDownload:
                alreadyDownload.add(nameTemp)
                url = res.group(1)
                writeToFile(url, '%s.%s' % (nameTemp ,url.split('.')[-1]))
                print('......*' * currentLevel, '已经下载', nameTemp, '的用户头像', '知乎首页：https://www.zhihu.com%s' % (userID), sep=' ')
    except:
        print('......*' * currentLevel, 'getFriendList异常')
    else:
        print('......*' * currentLevel, '正在获取用户', nameTemp, '的关注列表...', sep=' ')
        bsObj = BeautifulSoup(broswer.page_source, 'html.parser')
        elts = bsObj.findAll('a', {'class':'UserLink-link'})
        for elt in elts:
            img = elt.find('img')
            if img:
                friendSet.add(elt)
                print('......*' * currentLevel, 'https://www.zhihu.com%s' % (elt.attrs.get('href', 'no data')))
        elts = bsObj.findAll('button', {'class':'Button PaginationButton Button--plain'})
        if len(elts) != 0:
            count = elts[len(elts) - 1].get_text()
            for i in range(2, int(count) + 1):
                getAllFriendList(broswer, userID, i, friendSet, currentLevel)
        print('......*' * currentLevel, '用户', nameTemp, '的关注列表获取完毕', sep=' ')
        for elt in friendSet:
            href = elt.attrs.get('href', 'no data')
            if currentLevel == totalLevel:
                img = elt.find('img')
                if img:
                    res = re.match('^(https://.*)[0-9]x$', img.attrs.get('srcset', 'no data'))
                    if res:
                        if not href.split('/')[2] in alreadyDownload:
                            alreadyDownload.add(href.split('/')[2])
                            url = res.group(1).replace('_xl', '_xll')
                            writeToFile(url, '%s.%s' % (href.split('/')[2] ,url.split('.')[-1]))
                            print('......*' * (currentLevel + 1), '已经下载用户',nameTemp, '的关注用户', href.split('/')[2], '的头像', sep=' ')
            getFriendList(broswer, '%s/%s' % (href, userID.split('/')[3]), currentLevel + 1)

totalLevel = 5#递归层数
defaultPath = 'h:\\zhihu\\'#默认目录
currentPath = '%s%s' % (defaultPath, 'pic\\')#当前目录
alreadyDownload = set()#已经下载的用户头像
alreadyParse = set()#已经解析过的用户
totalUse = 0#文件写入次数

def writeToFile(url, fileName):
    try:
        global currentPath, totalUse, defaultPath
        totalUse = totalUse + 1
        if totalUse % 500 == 0:
            tempPath = '{0}pic-{1}\\'.format(defaultPath, totalUse)
            if not os.path.exists(tempPath):
                os.mkdir(tempPath)
            currentPath = '%s' % (tempPath)
        if not os.path.exists(currentPath):
            os.mkdir(currentPath)
        urlretrieve(url, '%s%s' % (currentPath, fileName))
    except:
        print('writeToFile异常')

if __name__ == "__main__":
    try:
        start = time.clock()
        time.sleep(5)
        broswer = webdriver.PhantomJS(executable_path=
                                      r"C:\phantomjs-2.1.1-windows\phantomjs-2.1.1-windows\bin\phantomjs.exe")
        getFriendList(broswer, r'/people/tu-si-ji-63/following')
    except:
        print('顶层调用异常')
    finally:
        broswer.quit()
        print('******', '共运行 {0:.3f}秒'.format(time.clock() - start), '一共扫描%d位用户的好友列表' % (len(alreadyParse)), '一共下载%d张用户头像' % (len(alreadyDownload)), sep='  ')  