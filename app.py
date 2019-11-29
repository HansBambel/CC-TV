from flask import Flask, render_template, request, url_for, redirect
import os
import overlayLib


imageFolder = os.path.join("static", "images")
cities = [
    {
        "query": "Buenos Aires, Argentina",
        "size": 24.34,
        "image": os.path.join(imageFolder, "defaultBigCity.png")
    },
    {
        "query": "Nairobi, Kenya",
        "size": 20.21,
        "image": os.path.join(imageFolder, "defaultSmallCity.png")
    }
]
overlayIm = os.path.join(imageFolder, "defaultOverlay.png")

app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/", methods=['POST'])
def compare():
    city1Query = request.form["City 1"]
    city2Query = request.form["City 2"]
    print(city1Query, " ", city2Query)
    if city1Query is "" or city2Query is "":
        # TODO display warning
        pass
    else:
        # Use the input, create new overlay, refresh page
        city1, size1 = overlayLib.getCityOutline(city1Query)
        city2, size2 = overlayLib.getCityOutline(city2Query)
        overlay = overlayLib.overlayCities(city1, size1, city2, size2)
        size1 = round(size1, 2)
        size2 = round(size2, 2)
        # TODO get Google maps background from big city
        global cities
        global overlayIm
        cities = [
            {
                "query": city1Query if size1 > size2 else city2Query,
                "size": size1 if size1 > size2 else size2,
                "image": os.path.join(imageFolder, "BigCity.png")
            },
            {
                "query": city2Query if size1 > size2 else city1Query,
                "size": size2 if size1 > size2 else size1,
                "image": os.path.join(imageFolder, "SmallCity.png")
            }
        ]
        overlayIm = os.path.join(imageFolder, "Overlay.png")
    return redirect(url_for("home"))


@app.route("/")
def home():
    return render_template("home.html", defaultCities=cities, overlayIm=overlayIm)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run(debug=True)

