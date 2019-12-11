# CC-TV
City Comparison - Totally Virtual!

![Example of Website](/static/images/Example.png)

Using Openstreetmaps ([Link](https://nominatim.openstreetmap.org/)) I am able to get the overlay for a lot of cities. Using these, I calculate their sizes to scale them accordingly.

---
#### Requirements
- flask
- geopandas
- Pillow
- shapely
- descartes
- matplotlib
---
### Usage
1. Clone this repository
2. Enter repository
3. In command line: "flask run"
4. Go to "http://127.0.0.1:5000/" in your browser
---
### Other Files
I also tried training a UNet architecture to detect urban areas, but for now this was not very successful.

As a result there are still files in this repository that I needed for that such as `trainCNN.py`, `GetData.ipynb`, `unet_model.py` and `unet_parts.py`