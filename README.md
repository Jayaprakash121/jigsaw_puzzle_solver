# Jigsaw Puzzle Piece Classification

This repository contains a deep learning-based project for classifying and predicting the positions of pieces in jigsaw puzzles. The project supports puzzles split into grids of various sizes (e.g., 3x3 or 5x5), using a convolutional neural network (CNN) to determine the correct positions of puzzle pieces.

## Features
- **Dynamic Grid Size**: Supports both 3x3 and 5x5 puzzle grids.
- **Image Processing**: Automatically splits images into equal grid pieces and remaps labels for classification.
- **Model Training and Evaluation**: Uses TensorFlow and Keras to build and train a CNN model for classification tasks.
- **Accuracy Evaluation**: Calculates prediction accuracy using testing data.
- **Output CSV**: Generates a CSV file with predicted piece positions for the entire dataset.

## Requirements

### Libraries
The following Python libraries are required:
- TensorFlow
- NumPy
- OpenCV
- Pandas
- scikit-learn

Install these libraries using pip:

```bash
pip install tensorflow numpy opencv-python pandas scikit-learn
```

### Input Data
1. **Dataset**: Provide images of puzzles, split into directories for training, validation, and testing.
2. **Label Files**: CSV files (e.g., `animal_3_3_jigsaw.csv` for 3x3 grids and `human_5_5_jigsaw.csv` for 5x5 grids) containing ground-truth labels for the correct positions of puzzle pieces.

## Usage

### Prepare Dataset
- Organize images in a directory, ensuring labels are properly structured in a CSV file.
- Image files should be named numerically (e.g., `0.jpg`, `1.jpg`, ...).

### Run the Script
1. Execute the main script `main.py`.
2. Provide the path to the dataset and select the desired grid size (3x3 or 5x5) when prompted.

### Training and Validation
- The script trains the CNN model on the provided dataset.
- Validation accuracy is monitored to prevent overfitting using early stopping.

### Testing and Output
- The model predicts positions for test images.
- A CSV file is generated with the predicted piece positions (e.g., `predicted_animal_3_3_jigsaw.csv` or `predicted_human_5_5_jigsaw.csv`).

## Project Workflow

### 1. Label Preprocessing
- Reads and remaps labels from the CSV files for easier processing.
- Each grid position is mapped to an integer value for classification.

### 2. Image Splitting
- Resizes input images to fit the grid size and splits them into individual pieces.

### 3. Model Architecture
- A CNN with convolutional, pooling, and dense layers.
- The output layer dynamically adapts to the grid size (9 for 3x3, 25 for 5x5).

### 4. Training
- Trains the CNN using categorical cross-entropy loss and Adam optimizer.
- Includes early stopping to terminate training if validation loss does not improve.

### 5. Testing
- Evaluates the trained model on unseen data and calculates accuracy.

### 6. Output Generation
- Maps predictions back to original grid positions and saves them in a structured CSV file.

## Results

### Accuracy
- The model provides the percentage of correctly classified pieces.

### CSV Output
- Contains predicted positions for each puzzle piece in the test set.

## Example Command
To run the script, use the following command:

```bash
python main.py --dataset_path /path/to/dataset --grid_size 3x3
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for feedback or suggestions.

---

Enjoy solving puzzles with deep learning!
