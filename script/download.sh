# RAFT and sam2
cd third_party/RAFT
mkdir models
cd models
wget https://drive.google.com/file/d/1UqyHUj7796Zrjdsw1kO5yNaagrfAGCyT/view?usp=drive_link
cd ../../sam2
mkdir checkpoints
cd checkpoints
wget https://drive.google.com/file/d/1Ld1bnySdx24fQ3FRdORQYNzRDVuBdofv/view?usp=drive_link
cd ../../../

# Driv3R
mkdir checkpoints
cd checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
wget https://drive.google.com/file/d/1lFpfh1Vo47XpqVacT51xYldYcCYEZpYA/view?usp=drive_link
wget https://drive.google.com/file/d/1tFbT0oaNXYf3regjS9nso01tuQs4ztbv/view?usp=drive_link
cd ..
