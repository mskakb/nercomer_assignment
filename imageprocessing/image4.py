#特徴量抽出と図示
import cv2

#画像の読み込み（グレースケール）
img = cv2.imread("/home/akiba/newcomer/imageprocessing/lena.jpeg",0)
img_mag = cv2.resize(img,None,fx=1.3,fy=1.3)
img_rot = cv2.rotate(img,cv2.ROTATE_180)

#特徴量検出器
sift = cv2.SIFT_create() #SIFT
fast = cv2.FastFeatureDetector_create() #FAST
orb = cv2.ORB_create() #ORB

#特徴量検出
kp_sift = sift.detect(img,None)
kp_sift_mag = sift.detect(img_mag,None)
kp_sift_rot = sift.detect(img_rot,None)

kp_fast = fast.detect(img,None)
kp_fast_mag = fast.detect(img_mag,None)
kp_fast_rot = fast.detect(img_rot,None)

kp_orb = orb.detect(img,None)
kp_orb_mag = orb.detect(img_mag,None)
kp_orb_rot = orb.detect(img_rot,None)

#キーポイントを描写
skip_sift = 1
skip_fast = 10
skip_orb = 20

img_sift = cv2.drawKeypoints(img,kp_sift[::skip_sift],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_mag_sift = cv2.drawKeypoints(img_mag,kp_sift_mag[::skip_sift],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_rot_sift = cv2.drawKeypoints(img_rot,kp_sift_rot[::skip_sift],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_fast = cv2.drawKeypoints(img,kp_fast[::skip_fast],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_mag_fast = cv2.drawKeypoints(img_mag,kp_fast_mag[::skip_fast],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_rot_fast = cv2.drawKeypoints(img_rot,kp_fast_rot[::skip_fast],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_orb = cv2.drawKeypoints(img,kp_orb[::skip_orb],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_mag_orb = cv2.drawKeypoints(img_mag,kp_orb_mag[::skip_orb],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_rot_orb = cv2.drawKeypoints(img_rot,kp_orb_rot[::skip_orb],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#画像の表示
cv2.imshow("sift",img_sift)
cv2.imshow("sift_mag",img_mag_sift)
cv2.imshow("sift_rot",img_rot_sift)

cv2.imshow("fast",img_fast)
cv2.imshow("fast_mag",img_mag_fast)
cv2.imshow("fast_rot",img_rot_fast)

cv2.imshow("orb",img_orb)
cv2.imshow("orb_mag",img_mag_orb)
cv2.imshow("orb_rot",img_rot_orb)

#キー入力待ち＆ウィンドウ消去
cv2.waitKey()
cv2.destroyAllWindows()