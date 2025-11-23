Spotify Advanced Analytics Dashboard

This project presents an interactive Streamlit dashboard for exploring and analyzing the Spotify Tracks Dataset.
It was collaboratively developed by Ayşenaz Arslan, Damla Gülen, and Miray Balıkoğlu.

Project Overview

The dashboard provides advanced multi‑dimensional visualizations to explore trends in:

Musical genres

Audio features

Popularity patterns

Artist and genre‑level distributions

It is designed as a comprehensive analytical tool that supports filtering, comparison, and deeper insights into Spotify’s track metadata.

Repository Structure
project/
│── app.py               # Main Streamlit dashboard script
│── clean_dataset.py     # Preprocesses dataset.csv to create clean_dataset.csv
│── clean_dataset.csv    # Cleaned dataset used by app.py
│── dataset.csv          # Original Spotify Tracks Dataset
│── README.md            # Project documentation
│── /images              # Screenshots of visualizations

Data Preparation

clean_dataset.py: Preprocesses the original Spotify Tracks Dataset (dataset.csv) and generates clean_dataset.csv.

app.py: Uses clean_dataset.csv as the main input for all visualizations and dashboard features.

Team Responsibilities 

All team members (Ayşenaz Arslan, Damla Gülen, Miray Balıkoğlu) contributed jointly to the following tasks:

Data Preprocessing and Cleaning

Data Analysis

Streamlit Dashboard Development

Interactive Component Implementation

README and Documentation

GitHub Delivery Preparation

Testing and Final Checks

Visualization Tasks (Individual Contributions):

Correlation Heatmap – Miray Balıkoğlu

Star Glyph Plot – Miray Balıkoğlu

Violin Plot – Miray Balıkoğlu

Treemap – Damla Gülen

Scatterplot Matrix – Damla Gülen

Histogram – Damla Gülen

Parallel Coordinates Plot – Ayşenaz Arslan

Sunburst Chart – Ayşenaz Arslan

Line Chart – Ayşenaz Arslan

Visualizations

Below are previews of each visualization included in the project:

Correlation Heatmap:
Visualizes the relationships between numerical audio features.
Screenshot_1.png

Star Glyph Plot:
Shows multi‑feature profiles for selected genres.
star_glyph.png

Violin Plot:
Displays popularity distributions across genres.
Screenshot_3.png

Treemap:
Represents hierarchical relationships between genres and artists.
Screenshot_5.png

Scatterplot Matrix:
Compares multiple audio features simultaneously.
Screenshot_6.png

Histogram:
Shows the distribution of Tempo (BPM).
Screenshot_7.png

Parallel Coordinates Plot:
Enables multi‑feature comparison across tracks.
Screenshot_8.png

Sunburst Chart:
Shows hierarchical structure of genres → artists.
Screenshot_9.png

Line Chart:
Plots popularity trends across various audio metrics.
Screenshot_10.png

Features

Dynamic sidebar filters

Genre‑based exploration

Multiple resizing options (“Bigger Screen” buttons)

Clean UI with consistent styling

Highly interactive Plotly visualizations

Handles large datasets efficiently

Technologies Used

Python 3

Streamlit

Plotly

Pandas / NumPy

scikit-learn

Running the Project

⿡ Install Required Libraries
Install all necessary Python libraries in a single command:

pip install streamlit pandas plotly numpy scikit-learn


⿢ Generate Clean Dataset
Before running the dashboard, run clean_dataset.py once to create clean_dataset.csv:

python clean_dataset.py


⿣ Start the Application
After the clean dataset is ready, launch the Streamlit dashboard:

streamlit run app.py

Dataset

Spotify Tracks Dataset (Kaggle): Contains over 100k tracks with genre, tempo, valence, energy, and more.
Dataset URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

License

This project is for educational purposes.

Acknowledgments

Special thanks to our instructors and team members for their collaboration in this multi‑visualization analytics project.
