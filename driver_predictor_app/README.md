# DriveAware — Image Upload + Model Prediction

This is a small Streamlit frontend that accepts a driver image, runs a trained ResNet-18 checkpoint, and displays the predicted class and scores for all ten classes.

## 1. Put your model checkpoint in this folder

Copy `person2_resnet18_best.pth` into the same folder as `app.py`.

The checkpoint must be the ResNet-18 model trained for 10 classes. This app expects a normal PyTorch state dict or a checkpoint containing `model_state_dict` or `state_dict`.

## 2. Install and run (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the app in your browser. Upload a JPG/PNG image and click **Predict driver activity**.

## Important: match training preprocessing

The app currently resizes images to `128 × 128` and applies ImageNet normalization. Before presenting, verify these settings against `notebooks/02_person2_transfer_learning.ipynb`. If training used different transforms, update `preprocess_image()` to match them exactly. Inference preprocessing must match training preprocessing.

If `load_state_dict` reports missing/unexpected keys or a size mismatch, the checkpoint may use a different architecture or output layer. Confirm which checkpoint you are using and how it was saved.

## Troubleshooting

- **Checkpoint not found:** copy `person2_resnet18_best.pth` into this folder.
- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again, or use Command Prompt with `.venv\\Scripts\\activate.bat`.
- **PyTorch install issue:** install a PyTorch build compatible with your computer from the official PyTorch installation selector, then install the remaining requirements.

## Limitations

This is a demonstration of the trained classifier, not a safety-critical system. A high softmax score does not guarantee a correct prediction or real-world reliability.
