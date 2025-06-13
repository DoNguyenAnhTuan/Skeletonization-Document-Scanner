import numpy as np
import argparse
import cv2 as cv
import imutils
import random

# Định nghĩa lớp DocumentScannerBasedFindContours
class DocumentScannerBasedFindContours:
  # Hàm khởi tạo với các tham số đầu vào và cấu hình
  def __init__(self, image, output="type2", show_process = True, t=2, minval_canny =60, maxval_canny = 120, size_adapt_output=31):
    self.img = image
    self.output = output
    self.show_process = show_process
    self.t = t
    self.minval=minval_canny
    self.maxval = maxval_canny
    self.size_adapt_output = size_adapt_output
    self.list_img = []
    self.lst_title = []

  # Hàm sắp xếp các điểm trong mảng theo thứ tự phù hợp
  def order_points(self, points):
    rect = np.zeros((4,2), dtype = "float32")
    sum_ = points.sum(axis=1)
    rect[0] = points[np.argmin(sum_)]
    rect[2] = points[np.argmax(sum_)]
    diff_ = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff_)]
    rect[3] = points[np.argmax(diff_)]
    return rect

  # Hàm biến đổi phối cảnh top-down của ảnh
  def transform_topdown(self, img, points):
    rect = self.order_points(points)
    tl, tr, br, bl = rect
    widthA = np.sqrt(((br[0]- bl[0])**2) +((br[1]- bl[1])**2))
    widthB = np.sqrt(((tl[0]- tr[0])**2) +((tl[1]- tr[1])**2))
    maxWidth = max(widthA, widthB)
    heightA = np.sqrt(((br[0]- tr[0])**2) +((br[1]- tr[1])**2))
    heightB = np.sqrt(((bl[0]- tl[0])**2) +((bl[1]- tl[1])**2))
    maxHeight = max(heightA, heightB)
    dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect, dst)
    print(M)
    warped = cv.warpPerspective(img, M, (int(maxWidth), int(maxHeight)))
    return warped

  # Hàm tiền xử lý ảnh (làm xám và xóa nhiễu)
  def preprocess(self, blur_method="gaussian"):
    image = self.img.copy()
    self.ratio = image.shape[0] / 500.0
    self.img_resize = cv.resize(image, (int(image.shape[1] / self.ratio) ,500))
    img_gray = cv.cvtColor(self.img_resize, cv.COLOR_BGR2GRAY)
    img_blur = None
    if blur_method == "gaussian":
      img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    elif blur_method == "bilateral":
      img_blur=cv.bilateralFilter(img_gray, 30, 20, 20)
    if self.show_process==True:
      self.lst_title.append("Ảnh ban đầu")
      self.list_img.append(self.img_resize)
      self.lst_title.append("Làm xám")
      self.list_img.append(img_gray)
      self.lst_title.append("Xóa nhiễu")
      self.list_img.append(img_blur)
    return img_blur

  # Hàm xử lý tìm cạnh ảnh bằng phương pháp Canny
  def handle_edge(self):
    img_blur = self.preprocess(blur_method="gaussian")
    edged = cv.Canny(img_blur, self.minval, self.maxval)
    kernel3 = np.ones((3,3), np.uint8)
    img_close = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel3)
    if self.show_process==True:
        self.lst_title.append("Tìm cạnh")
        self.list_img.append(edged)
        self.lst_title.append("Dùng phép đóng")
        self.list_img.append(img_close)
    return img_close

  # Hàm tính diện tích của vùng tìm thấy
  def calculate_area(self, points):
    area4 = 0
    fourpoints = self.order_points(points.copy())
    def dientichtamgiac(threepoints):
        A, B, C = threepoints
        area3 = 1/2 * np.abs(((B[0] - A[0])*(C[1] - A[1]) - (C[0] - A[0])*(B[1] - A[1])))
        return area3
    area4 = dientichtamgiac([points[0],points[1], points[2]]) + dientichtamgiac([points[2],points[3], points[0]])
    return area4

  # Hàm tìm các đường viền của đối tượng trong ảnh
  def find_contours(self):
    img_close = self.handle_edge()
    cnts,_ = cv.findContours(img_close.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:6]
    self.screenCnt = None
    min_area = self.img_resize.shape[0]*self.img_resize.shape[1] // 5
    img_cop_ = self.img_resize.copy()
    for c in cnts:
      perimeter = cv.arcLength(c, True)
      epsilon = perimeter*(self.t/1000)
      approx = cv.approxPolyDP(c, epsilon, True)
      try:
        cv.drawContours(img_cop_, [approx], -1, (0, 255, 0), 2)
        clr = [random.randint(0, 256) for i in range(3)]
        for point in approx:
            x, y = point[0]
            cv.circle(img_cop_,(x, y),5,clr,2)
      except:
        continue
      if len(approx) == 4 and self.calculate_area(approx.reshape(4, 2)) > min_area:
        self.screenCnt = approx
        self.screenCnt = (self.screenCnt*self.ratio).astype("int32")
        break
    self.list_img.append(img_cop_)
    self.lst_title.append("Các điểm sau khi lược bỏ theo epsilon")
    if self.show_process==True:
        img_cop = self.img.copy()
        try:
            cv.drawContours(img_cop, [self.screenCnt], -1, (0, 255, 0), int(2*self.ratio))
            for point in self.screenCnt:
                x, y = point[0]
                cv.circle(img_cop,(x, y),5,(255, 255, 127), int(10*self.ratio))
            self.lst_title.append("Tìm 4 điểm")
            self.list_img.append(img_cop)
        except:
            pass

  # Hàm lấy kết quả cuối cùng sau khi tìm được vùng cần cắt
  def get_result(self):
    self.find_contours()
    warped = self.transform_topdown(self.img, self.screenCnt.reshape(4, 2))
    self.lst_title.append("Chuyển phối cảnh")
    self.list_img.append(warped)
    if self.output == "type0":
      pass
    elif self.output == "type1":
      warped = cv.GaussianBlur(warped, (3,3), 0)
    elif self.output == "type2":
      warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
      warped = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, self.size_adapt_output, 10)
    self.result = warped
    if self.show_process == True:
      self.lst_title.append("Kết quả")
      self.list_img.append(self.result)
    return self.img, self.result
  
  # Hàm hiển thị kết quả
  def show_result(self, smaller=100):
    try:
        origin, result = self.get_result()
        origin = cv.resize(origin, (int(origin.shape[1]*smaller/ 100), int(origin.shape[0]*smaller/ 100)))
        result = cv.resize(result, (origin.shape[1],origin.shape[0]))
    except:
        pass
    return self.list_img, self.lst_title
