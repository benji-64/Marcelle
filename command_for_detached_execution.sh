nohup python -u emotion_classi_comp.py 1> out.log 2> err.log &

# pour environnement virtuel
source .venv/bin/activate 

# pour interrompre
## Afficher les process
ps aux | grep python
## kill process
pkill -9 $IDprocess

