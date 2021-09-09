# cars DCGAN

image generator which generate cars images!


## usage
```python
from cars_DCGAN import generate_car
generate_car()
```
the output image will be saved in the `new_predictions` folder


## training:
first run the `process_data()` function
then call the `train()` function. 
```python
from cars_DCGAN import train
from utils import DataPipe
dp = DataPipe
dp.import_data([r"C:\source_dir1",r"C:\source_dir2"])
train(epochs=100000, batch_size=128, save_interval=500)
```
## example:
<p align="left">
  <img width="500" src="https://github.com/matan-chan/pokemon_DCGAN/blob/main/examples/example1.png?raw=true">
</p>

## data:
[Cars Dataset][website]



[website]: http://ai.stanford.edu/~jkrause/cars/car_dataset.html