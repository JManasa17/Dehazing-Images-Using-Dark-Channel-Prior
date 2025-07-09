# AI-Powered Image Dehazing Using Dark Channel Prior

A real-time image enhancement web application built using Flask and OpenCV, powered by the Dark Channel Prior (DCP) algorithm with adaptive AI logic. This tool removes haze from outdoor images — ideal for surveillance, autonomous vehicles, remote sensing, and image clarity restoration.

---

## Project Highlights

- Classifies images into Low, Medium, and High haze levels
- Uses Dark Channel Prior and Guided Filtering for accurate haze removal
- Adaptive parameter tuning based on haze intensity
- Real-time image upload, dehazing, comparison, and download
- Color enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)

---

## Key Features

- Upload hazy images (.jpg, .jpeg, .png)
- Automatic haze level detection & dehazing
- View original vs dehazed image side-by-side
- Download the enhanced, haze-free image

---

## Tech Stack

| Layer        | Tools Used                                                 |
|--------------|------------------------------------------------------------|
| Backend      | Python, Flask, OpenCV, NumPy                               |
| Frontend     | HTML, CSS, JavaScript (Canvas rendering)                   |
| Algorithm    | Dark Channel Prior (DCP), Guided Filtering, Adaptive CLAHE |
| IDE          | VS Code / PyCharm                                          |

---

## Folder Structure
Dehaze/
├── backend.py           # Main Flask backend script
├── templates/
│   └── index.html       # Frontend upload/download interface
├── Images/              # Sample hazy images for testing
└── README.md            # Project information(this file)

---

## How to Run Locally

1. Clone the Repository
```bash
git clone https://github.com/your-username/Dehaze.git
cd Dehaze
```

2. Install Dependencies
```bash
pip install flask opencv-python numpy
```

3. Run the App
```bash
python backend.py
```

4. Open in Your Browser
```
http://127.0.0.1:5000
```

---

## Test Cases Covered

- Upload and dehaze valid image files
- Upload non-image file → shows appropriate error
- Missing file upload → shows error
- Performance tested with large-resolution images
- Supports multiple formats: .png, .jpg, .jpeg
- Allows dehazed image download after processing

---

## Core Algorithm: Dark Channel Prior (DCP)

1. Estimate the dark channel of the image
2. Detect brightest pixels to estimate atmospheric light
3. Compute and refine the transmission map
4. Reconstruct the haze-free image
5. Apply CLAHE to improve visibility and colors

---

## Future Enhancements

- Batch image dehazing support
- Real-time video dehazing (frame-by-frame)
- Deploy to cloud (PythonAnywhere, Render, etc.)
- Use deep learning (CNN) for intelligent haze detection and removal

---

## License

This project is created for academic and learning purposes.  
Please contact the maintainers for reuse or citation rights.
