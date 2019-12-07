import geopandas
import urllib
import shapely
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import json
from PIL import Image


def getDistance(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def getCitySize(cityBounds):
    lat1, lon1, lat2, lon2 = cityBounds
    return getDistance(lat1, lon1, lat2, lon2)


def getCityOutline(query: str):
    # TODO cache query locally and check if already exists
    openStreetMap_base_url = "https://nominatim.openstreetmap.org/search"
    url = openStreetMap_base_url + '?' + urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "polygon_geojson": 1
    })

    response = urllib.request.urlopen(url).read()
    responseJson = json.loads(response)

    # find first multipolygon/polygon result (most likely the correct one)
    for shape in responseJson:
        # Included this try if shape has no geojson. Now just goes to next entry
        try:
            if isinstance(shapely.geometry.shape(shape["geojson"]), shapely.geometry.MultiPolygon):
                cityShape = shapely.geometry.shape(shape["geojson"])
                lat1, long1, lat2, long2 = cityShape.bounds
                bounds = [lat1, long1, lat2, long2]
                # check if size is way too big or too small. If so --> kick out
                size = getCitySize(bounds)
                if size > 200 or size < 5:
                    continue
                cityGDF = geopandas.GeoDataFrame({
                    "geometry": cityShape
                })
                return cityGDF, bounds
            elif isinstance(shapely.geometry.shape(shape["geojson"]), shapely.geometry.Polygon):
                cityShape = shapely.geometry.shape(shape["geojson"])
                lat1, long1, lat2, long2 = cityShape.bounds
                bounds = [lat1, long1, lat2, long2]
                # check if size is way too big or too small. If so --> kick out
                size = getCitySize(bounds)
                if size > 200 or size < 5:
                    continue
                cityGDF = geopandas.GeoDataFrame({
                    "geometry": cityShape
                }, index=[0])
                return cityGDF, bounds
        except:
            continue
    # If nothing found return None
    return None, None


def overlayCities(city1, size1, city2, size2):
    if size1 > size2:
        bigCity = city1
        bigSize = size1
        smallCity = city2
        smallSize = size2
    else:
        bigCity = city2
        bigSize = size2
        smallCity = city1
        smallSize = size1
    # Save the cities as images
    bigCity.plot(color="red")
    plt.axis("off")
    plt.savefig("static/images/BigCity.png", transparent=True)
    plt.close()
    smallCity.plot(alpha=0.7)
    plt.axis("off")
    plt.savefig("static/images/SmallCity.png", transparent=True)
    plt.close()
    # Load images with Pillow
    bigCityIm = Image.open("static/images/BigCity.png")
    smallCityIm = Image.open("static/images/SmallCity.png")
    oldWidth, oldHeight = smallCityIm.size
    newWidth = int(oldWidth * smallSize / bigSize)
    newHeight = int(oldHeight * smallSize / bigSize)
    # Scale down the smaller city
    smallCityImScaled = smallCityIm.resize((newWidth, newHeight))
    # Find center of bigger city
    bigCityWidth, bigCityHeight = bigCityIm.size
    bigCenterW = int(bigCityWidth / 2)
    bigCenterH = int(bigCityHeight / 2)
    topLeftX = bigCenterW - int(newWidth / 2)
    topLeftY = bigCenterH - int(newHeight / 2)
    botRightX = bigCenterW + (newWidth - int(newWidth / 2))
    botRightY = bigCenterH + (newHeight - int(newHeight / 2))
    # Paste the small city onto the bigger one (the mask key-word reserves the transparency of the png)
    bigCityIm.paste(smallCityImScaled, (topLeftX, topLeftY, botRightX, botRightY), mask=smallCityImScaled)
    bigCityIm.save("static/images/Overlay.png", "png")
    #     bigCityIm.show()
    return bigCityIm
