# Nhập các thư viện cần thiết
from os import name
import streamlit as st
import cv2 as cv
import numpy as np
from pp1_contour import DocumentScannerBasedFindContours
from pp2_houghline import DocumentScannerBasedHoughLines

# Thiết lập tiêu đề cho ứng dụng Streamlit
st.title('Document Scanner')
st.sidebar.title('Upload image')  # Tiêu đề cho phần tải lên hình ảnh trong sidebar
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])  # Chỉ định các loại file ảnh có thể tải lên

# Kiểm tra nếu có file tải lên
if uploaded_file != None:
    try:
        # Đọc ảnh gốc từ đường dẫn được chỉ định
        img_orgin = cv.imread("./image/" + uploaded_file.name)
        st.header("Origin")  # Hiển thị tiêu đề cho ảnh gốc
        st.image(img_orgin.astype("uint8"), channels="BGR", clamp=True)  # Hiển thị ảnh gốc
    except:
        st.write("Không tìm thấy ảnh!")  # Thông báo khi không tìm thấy ảnh
else:
    # Nếu không có file tải lên, sử dụng ảnh mặc định
    img_orgin = cv.imread("./image/receipt.jpg")
    pass

# Tạo danh sách chứa các ảnh mẫu
lst_test = []
for i in range(8):
    lst_test.append(img_orgin)

# Thiết lập các tham số cấu hình trên sidebar
st.sidebar.title("Config:")
st.sidebar.header("Output:")
size_adapt_output = st.sidebar.slider("Blocksize adaptive:", min_value=3, max_value=101, step=2, value=31)  # Cài đặt tham số kích thước cho adaptive

# Thiết lập các tham số cho thuật toán Contour
col1_config, col2_config = st.sidebar.columns(2)
col1_config.header("Contour:")
t_contour = col1_config.slider("t (epsilon = t/1000*perimeter):", min_value=1, max_value=200, step = 1, value=20)
minval_contour = col1_config.slider("minval Canny: ", min_value=0, max_value=256, step=1, value=60)
maxval_contour = col1_config.slider("maxval Canny: ", min_value=0, max_value=256, step=1, value=120)

# Thiết lập các tham số cho thuật toán Houghline
col2_config.header("Houghline:")
rho_houghline = col2_config.slider("Rho: ", min_value=1, max_value=3,step=1, value=1)
theta_houghline = col2_config.slider("Theta: ", min_value=180, max_value=360, step=1, value=360)
thresh_houghline = col2_config.slider("Thresh: ", min_value=50, max_value=150, step=1, value=100)
goc1_houghline = col2_config.slider("Corner 1: ", min_value=0, max_value=180, step=1, value= 80)
goc2_houghline = col2_config.slider("Corner 2: ", min_value=0, max_value=180, step=1, value=100)
goc_houghline = [goc1_houghline, goc2_houghline]
minval_hough = col2_config.slider("minval Canny: ", min_value=0, max_value=256, step=1, value=232)
maxval_hough = col2_config.slider("maxval Canny: ", min_value=0, max_value=256, step=1, value=247)

# Hiển thị kết quả khi nút được nhấn và có ảnh được tải lên
if st.sidebar.button("Click here!") and uploaded_file:
    # Thực thi các thuật toán Contour và Houghline
    scan_contour = DocumentScannerBasedFindContours(img_orgin, t=t_contour, minval_canny=minval_contour, maxval_canny=maxval_contour, size_adapt_output=size_adapt_output)
    lst_img_contour, lst_title_contour = scan_contour.show_result()
    scan_houghline = DocumentScannerBasedHoughLines(img_orgin, rho_acc=rho_houghline, theta_acc=theta_houghline, thresh=thresh_houghline, corner1=min(goc_houghline), corner2=max(goc_houghline), size_adapt_output=size_adapt_output, minval_canny=minval_hough, maxval_canny=maxval_hough)
    lst_img_hough, lst_title_houghline = scan_houghline.show_result()

    # Hiển thị kết quả cuối cùng
    st.header("Result")
    col1_contour, col2_houghline = st.columns(2)
    col1_contour.subheader("Contour")
    if len(lst_img_contour) == 9:
        col1_contour.image(lst_img_contour[-1].astype("uint8"), clamp=True)  # Hiển thị ảnh kết quả Contour
    else:
        col1_contour.image(img_orgin.astype("uint8"), channels="BGR", clamp=True)
    col2_houghline.subheader("Houghline")
    if len(lst_img_hough) == 8:
        col2_houghline.image(lst_img_hough[-1].astype("uint8"), clamp=True)  # Hiển thị ảnh kết quả Houghline
    else:
        col2_houghline.image(img_orgin.astype("uint8"), channels="BGR", clamp=True)

    # Hiển thị tiến trình xử lý
    st.header("Progress")
    col1_qt, col2_qt = st.columns(2)
    col1_qt.subheader("Contour")
    for i in range(len(lst_img_contour)):
        col1_qt.write(lst_title_contour[i])
        if len(np.array(lst_img_contour[i]).shape) == 3:
            col1_qt.image(lst_img_contour[i].astype("uint8"), channels="BGR", clamp=True)
        else:
            col1_qt.image(lst_img_contour[i].astype("uint8"), clamp=True)
    col2_qt.subheader("Houghline")
    for i in range(len(lst_img_hough)):
        col2_qt.write(lst_title_houghline[i])
        if len(np.array(lst_img_hough[i]).shape) == 3:
            col2_qt.image(lst_img_hough[i].astype("uint8"), channels="BGR", clamp=True)
        else:
            col2_qt.image(lst_img_hough[i].astype("uint8"), clamp=True)

    