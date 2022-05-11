# project-ml

Demo homework for ml-intro course.

This demo uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

Used an src layout


![image](https://user-images.githubusercontent.com/55091681/167725817-b4d0be8e-7137-4e49-8806-fd8e3c934020.png)


The src directory is a better approach because:

  - simpler packaging code;
  - zero fuss for large libraries that have multiple packages;
  - clear separation of code being packaged and code doing the packaging.


## Usage
This package allows you to train model for detecting the presence of heart disease in the patient.
1. Clone this repository to your machine.
2. Download  dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. read  file and see head
```sh
poetry run train -d data/train.csv
```
6. read file, view head and generate profiling.html to the folder with the project:
```sh
poetry run train -eda True
```
7. or:
```sh
poetry run train -d data/train.csv -eda True
```

 

