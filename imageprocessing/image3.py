#差分画像生成
import cv2

#画像読み込み
img = cv2.imread("/home/akiba/newcomer/imageprocessing/sabun.png")

#画像サイズ取得
h, w ,c = img.shape[:3]
print(f"(height,width): ({h},{w})")

#画像分割
img1 = img[20:h, 0:int(w/2)]
img2 = img[20:h, int(w/2):w]

#平滑化
grid = 7
img1 = cv2.blur(img1,(grid,grid))
img2 = cv2.blur(img2,(grid,grid))
cv2.imshow("left",img1)
cv2.imshow("right",img2)

#差分抽出
img_diff = cv2.absdiff(img1,img2)
cv2.imshow("diff",img_diff)

#キー入力待ち＆ウィンドウ消去
cv2.waitKey()
cv2.destroyAllWindows()