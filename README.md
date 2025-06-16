```markdown
# 🧾 Skeletonization Document Scanner

This project implements a basic document scanning pipeline using OpenCV and image skeletonization techniques. It processes input images (e.g., photos of paper documents) and transforms them into clean, top-down scanned versions.

---

## 📌 Features

- 🖼️ Automatic edge detection & contour finding
- 📐 Perspective transformation (warp)
- 🧠 Skeletonization for better structure identification
- 🧾 Output of scanned-style documents

---

## 📂 Structure

```

Skeletonization-Document-Scanner/
├── data/
│   └── sample\_input.jpg
├── src/
│   └── scanner.py
│   └── skeletonize.py
├── output/
│   └── scanned\_output.jpg
└── README.md

````

---

## 🔧 Requirements

Install dependencies:

```bash
pip install opencv-python numpy
````

---

## 🚀 How to Run

```bash
python src/scanner.py --image data/sample_input.jpg
```

The script will display the scanning process step-by-step and output the result to the `output/` folder.

---

## 📊 Example Results

| Input Photo            | Scanned Output           |
| ---------------------- | ------------------------ |
| ![raw](docs/input.jpg) | ![scan](docs/output.jpg) |

---

## 🧠 Applications

* Digitizing paper documents
* Mobile scanning tools
* Preprocessing for OCR systems

---

## 👨‍💻 Author

**Do Nguyen Anh Tuan**
🎓 MSc in Information Technology @ Lac Hong University
🏢 FabLab @ EIU | Focused on AI, Computer Vision, and Robotics
🌐 [Portfolio Website](https://donguyenanhtuan.github.io/AnhTuan-Portfolio)


