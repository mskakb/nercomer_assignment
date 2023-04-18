#画像の表示、縮小拡大、回転、二値化
import cv2

#画像読み込み
img = cv2.imread("/home/akiba/newcomer/imageprocessing/Gorilla.jpeg")

#縮小
cur = cv2.resize(img,None,None,0.5,0.5)

#拡大
mag = cv2.resize(img,None,None,1.5,1.5)

#回転
rot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

#グレースケール化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#二値化と閾値表示
ret, img_otsu = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
print("ret: {}".format(ret))

#画像表示
cv2.imshow("image",img)
cv2.imshow("kakudai",mag)
cv2.imshow("shukushou",cur)
cv2.imshow("kaiten",rot)
cv2.imshow("nitika",img_otsu)

#キー入力待ち＆ウィンドウ消去
cv2.waitKey()
cv2.destroyAllWindows()