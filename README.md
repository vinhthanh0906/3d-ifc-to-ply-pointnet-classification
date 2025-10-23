# 🧱 3D IFC to PLY PointNet Classification

Turn 3D object files from **.ifc** format into **.ply** point clouds and classify them into categories using a **PointNet** neural network architecture.

---

## 📦 Requirements

Before running the project, make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## 🧰 Data Preprocessing

1. Go to the `data_preprocessing` folder:
   ```bash
   cd data_preprocessing
   ```

2. Convert your `.ifc` files to `.ply` format using:
   ```bash
   python ifctoPlyfile.py
   ```

> ⚠️ **Note:**  
> The `.ply` data should be processed with a **Poisson filter** for better surface identity and reconstruction quality.

---

## 🧠 Training the Model

1. Navigate to the `train` directory:
   ```bash
   cd train
   ```

2. Start training the PointNet model:
   ```bash
   python train.py
   ```

---

## 🖼️ Demo

Below is a sample visualization of the IFC → PLY → PointNet classification process:

**IFC file:**
![IFC file](https://github.com/user-attachments/assets/627a4366-b556-435b-a086-6154f57c6b79)

**PLY file:**
![PLY file](https://github.com/user-attachments/assets/1bb05b56-172c-4a76-a8ad-6430e8a533d5)

> Example: A 3D object converted from IFC format, processed as a point cloud, and classified by the PointNet model.



## 🚧 Project Status

This project is **still a prototype**.  
Future improvements include:
- Applying Poisson surface reconstruction for higher-fidelity mesh data  
- Expanding labeled datasets for improved classification performance  
- Adding visualization and evaluation tools  

---

## 📚 Overview

- **Input:** IFC (Industry Foundation Classes) 3D building files  
- **Output:** Classified 3D models in `.ply` point cloud format  
- **Architecture:** [PointNet](https://arxiv.org/abs/1612.00593)  
- **Goal:** Automate 3D object class recognition in BIM (Building Information Modeling) workflows

---

## 🧑‍💻 Author
Developed by **Nguyen Thanh Vinh**  
Prototype for IFC-to-PointNet classification research.
