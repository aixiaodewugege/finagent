# FinAgent

# Installation
```
conda create -n finagent python=3.10
conda activate finagent

#for linux
apt-get update && apt-get install -y libmagic-dev
# for mac
pip install python-magic-bin==0.4.14

conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install -r requirements.txt

```

# Run
```
python main.py

