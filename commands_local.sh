# Use conda anaconda3/2023.3
conda create --name botbusters python==3.7.1 
conda activate botbusters
conda install --name botbusters pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch 
conda install --name botbusters pyg -c pyg -c conda-forge
conda install --name botbusters -c conda-forge matplotlib
conda install --name botbusters -c huggingface tokenizers==0.10.1 -y
conda install --name botbusters -c conda-forge transformers==4.4.2 -y
conda install --name botbusters importlib-metadata -y
conda install --name botbusters packaging==21.3 -y
