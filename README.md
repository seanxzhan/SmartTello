Use the following command to install requirements.txt
```
$ pip install -r requirements.txt
```

TODO list:
- get tello to fly while tracking my nose
- keep same distance 
- gestures (start by experimenting with webcam.py)


Use conda environment

The controls are:
- T: Takeoff
- L: Land
- Arrow keys: Forward, backward, left and right.
- A and D: Counter clockwise and clockwise rotations
- W and S: Up and down.

python test.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel