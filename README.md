# Crack Detection in Pavements Using Classical Image Processing Techniques

## Project Overview

This project aims to detect cracks in pavement using classical image processing techniques. The system processes unedited video footage recorded from a bicycle and identifies cracks or potholes in the pavement. The primary goal is to minimize human effort in reviewing extensive footage for infrastructure maintenance.

## Use Case

The system is designed for infrastructure maintenance teams to efficiently detect and locate cracks in pavement. By inputting video footage into the program, it processes the video and generates a list of timestamps corresponding to detected cracks. This list can be used for targeted pavement repairs.

## Implementation Steps

1. **Literature Review**: We reviewed existing methods for crack detection, focusing on both image processing and machine learning approaches.
2. **Data Collection**: Videos were collected using a phone mounted on a bicycle. The dataset was then processed using the developed methods.
3. **Crack Detection**: We implemented a thresholding technique, enhanced by various pre- and post-processing steps to minimize false positives and maximize detection accuracy.
4. **User Interface**: A minimalist user interface was developed to input video files and output the detection results.

## Theory

The project draws inspiration from several research papers and open-source projects that explore crack detection using both image processing and machine learning techniques. The literature informed our choice of methods and helped refine our approach.

## Dataset and Challenges

### Our Data

- **Collection Method**: Video footage was collected using a bike-mounted phone, capturing videos at 30 FPS with a resolution of 1080x2400 pixels.
- **Challenges**: The data collection environment presented challenges such as wet asphalt, fallen leaves, and varying lighting conditions, which introduced noise and false positives in crack detection.

### Challenges

Key challenges include:
- **False Positives**: Non-crack features like wet spots, crosswalks, and leaves often mimic cracks.
- **Bike Wheel Interference**: The bike wheel in the footage occasionally caused false detections.

## Methods

### Inputs

The program accepts two types of inputs:
- **Video Files**: Raw video footage in `.mp4` format.
- **Pickle Files**: Pre-processed image data stored in `.p` files.

### Pre-processing

1. **Grayscaling**: Converts images to grayscale to simplify crack detection.
2. **Downscaling**: Reduces image size to remove fine details and speed up processing.
3. **Gaussian Low-pass Filter**: Smooths the image to reduce noise.
4. **Median Filter**: Further removes noise by focusing on dark pixel outliers.
5. **Logarithmic Transform**: Enhances contrast between cracks and pavement.

### Thresholding

Adaptive thresholding is used to distinguish crack pixels from non-crack pixels by comparing pixel intensity with surrounding pixels.

### Post-processing

1. **Noise Reduction**: Removes small, unconnected pixel clusters identified as noise.
2. **Straight Line Removal**: Eliminates intentionally placed infrastructure, like curbs or crosswalks, which may be falsely detected as cracks.
3. **Colour Filtering**: Masks green, brown, and white areas to prevent detection of foliage, paint, and other non-crack features.
4. **Bike Wheel Removal**: Manually masks the area of the image where the bike wheel appears.

### Output

The program outputs a list of timestamps where cracks are detected in the video, along with the processed images. Each crack is scored based on the number of pixels identified.

## Results

The system successfully detects major cracks and minimizes false positives, although some challenges remain, particularly with wet asphalt and pedestrian crossings. The program is adaptable, allowing for parameter adjustments to either reduce false positives or increase the detection of smaller cracks.

## Conclusions

### Potential Improvements

- **Data Collection**: Collecting data under optimal conditions (e.g., dry weather, no leaves) could improve accuracy.
- **Result Generation**: Implementing advanced techniques like feature detection or deep learning could further enhance crack detection.

### Learning Outcomes

This project provided valuable experience in image processing techniques, including filtering, thresholding, and morphological operations. It also emphasized the importance of data quality and the challenges of real-world application in infrastructure maintenance.

## How to Run the Project

1. Clone this repository.
2. Place your video files in the `/input` directory.
3. Run the `main.py` script to process the videos.
4. Check the `/output` directory for results, including detected crack images and a text file with timestamps.

## Bibliography

1. Sun, Y., Salari, E., & Chou, E. (2009). Automated pavement distress detection using advanced image processing techniques. *2009 IEEE International Conference on Electro/Information Technology*.
2. Kinamarp Htanmohs. crack-detection-opencv. GitHub repository.
3. Huang, Z., Che, W., Al-Tabbaa, A., & Brilakis, I. (2022). NHA12D: A new pavement crack dataset and a comparison study of crack detection algorithms. *European Conference on Computing in Construction*.
