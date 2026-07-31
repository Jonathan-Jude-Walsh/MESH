#MATLAB must be downloaded via https://au.mathworks.com/downloads/ and https://www.youtube.com/watch?v=ZNHJkCo5sOc

sudo apt update
sudo apt full-upgrade -y
sudo apt autoremove -y

sudo apt install -y \
build-essential \
cmake \
make \
gcc \
g++ \
clang \
gdb \
lldb \
pkg-config \
git \
git-lfs \
gh \
curl \
wget \
unzip \
zip \
p7zip-full \
tar \
tree \
jq \
htop \
btop \
tmux \
screen

sudo apt install -y \
python3 \
python3-pip \
python3-venv \
python3-dev \
ipython3 \
jupyter-notebook

python3 -m pip install --upgrade pip setuptools wheel

pip install \
numpy \
scipy \
matplotlib \
pandas \
scikit-learn \
seaborn \
jupyterlab \
notebook \
opencv-python \
soundfile \
librosa \
resampy \
h5py \
tqdm \
joblib

pip install \
tensorflow \
tflite-runtime \
keras \
torch \
torchvision \
torchaudio \
onnx \
onnxruntime \
transformers \
datasets

pip install tflite-support

sudo apt install -y \
sox \
ffmpeg \
audacity \
pulseaudio-utils

pip install \
audioread \
pydub \
sounddevice \
pyroomacoustics

sudo apt install -y \
gcc-arm-none-eabi \
gdb-multiarch \
openocd \
minicom \
picocom \
dfu-util

pip install platformio

curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh

sudo apt install -y \
minicom \
cutecom \
gtkterm

pip install pyserial

sudo snap install code --classic

code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension platformio.platformio-ide
code --install-extension ms-toolsai.jupyter
code --install-extension GitHub.copilot
code --install-extension GitHub.vscode-pull-request-github
code --install-extension eamodio.gitlens
code --install-extension twxs.cmake
code --install-extension ms-vscode.cmake-tools
code --install-extension marus25.cortex-debug

echo 'export PATH=$PATH:/usr/local/MATLAB/R2026a/bin' >> ~/.bashrc
source ~/.bashrc

sudo apt install -y kicad

sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER

pip install \
pyyaml \
rich \
click \
networkx \
pyqt5 \
pyusb \
crcmod