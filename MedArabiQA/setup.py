# Cell 1: Install requirements and start the server

# Note: The vLLM server will run continuously in the background.

!pip install "vllm==0.6.1.post1" "transformers==4.45.2" -q



import os

# This command starts the server as a background process

os.system("nohup vllm serve MBZUAI/BiMediX2-8B-hf --max-model-len 32000 --port 8000 --trust-remote-code > vllm_server.log 2>&1 &")



# Give the server a moment to start up

import time

time.sleep(60) # Wait for 1 minute for the server to initialize

print("✅ vLLM Server for BiMediX2 started in the background.")





# Cell 1: Corrected installation process to fix dependency conflicts



# First, force the upgrade of the core libraries to versions that satisfy the requirements.

# This resolves the torch, torchaudio, and numpy version conflicts.

print("Step 1: Upgrading core libraries (torch, torchaudio, numpy)...")

!pip install --upgrade torch torchaudio numpy -q



# Now, install the specific versions required for the BiMediX2 model

print("Step 2: Installing vLLM and Transformers...")

!pip install "vllm==0.6.1.post1" "transformers==4.45.2" -q



print("\n✅ All packages installed successfully!")

print("🔴 IMPORTANT: Please restart the session now.")





# Cell 1: Definitive installation of all required packages with compatible versions



print("Installing all required packages with specific, compatible versions. This may take a few minutes...")



# This single command installs vllm and its exact dependencies,

# then finds compatible versions for the other libraries.

!pip install "torch==2.4.0" "numpy==1.26.4" "vllm==0.6.1.post1" "transformers==4.45.2" openai rouge-score bert-score nltk -q



print("\nInstallation attempt complete. Please check the log above for any ERROR messages.")

print("If there are no errors, please restart the session now to load the correct versions.")





# Cell 1: Definitive installation by first removing conflicting packages



print("Step 1: Uninstalling conflicting pre-installed packages (torchaudio, thinc)...")

# The -y flag automatically confirms the uninstall.

!pip uninstall -y torchaudio thinc torchvision fastai



print("\nStep 2: Installing all required packages with specific, compatible versions...")

# This single command installs vllm and its exact dependencies,

# then finds compatible versions for the other libraries.

!pip install "torch==2.4.0" "numpy==1.26.4" "vllm==0.6.1.post1" "transformers==4.45.2" openai rouge-score bert-score nltk -q



print("\nInstallation attempt complete. Please check the log above for any ERROR messages.")

print("If there are no errors, please restart the session now to load the correct versions.")
