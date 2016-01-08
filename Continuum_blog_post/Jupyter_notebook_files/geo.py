import geopy
import numpy as np
import pandas as pd
from bokeh.browserlib import view
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.models.glyphs import Circle
from bokeh.models import (
    GMapPlot, Range1d, ColumnDataSource, LinearAxis,
    PanTool, WheelZoomTool, BoxSelectTool, HoverTool,
    BoxSelectionOverlay, GMapOptions,
    NumeralTickFormatter, PrintfTickFormatter)
from bokeh.resources import INLINE

ADDRESS = "1959 Southern Boulevard , New York NY"

lat = []
lon = []

geocoders = [
    # geopy.geocoders.Nominatim(timeout=5),
    geopy.geocoders.ArcGIS(timeout=5),
    geopy.geocoders.GoogleV3(timeout=5),
    # geopy.geocoders.Yandex(timeout=5),
    # geopy.geocoders.OpenMapQuest(timeout=5),
    # geopy.geocoders.GeocoderDotUS(timeout=5),
    # geopy.geocoders.Baidu(),
    # geopy.geocoders.Bing(),
    # geopy.geocoders.YahooPlaceFinder(),
    # geopy.geocoders.IGNFrance(),
    # geopy.geocoders.GeoNames(),
    # geopy.geocoders.NaviData(),
    # geopy.geocoders.What3Words(),
    # geopy.geocoders.OpenCage(),
    # geopy.geocoders.smartystreets(),
    # geopy.geocoders.GeocodeFarm(),
    # geopy.geocoders.LiveAddress(),
]

for service in geocoders:
    location = service.geocode(ADDRESS)
    print (service, location.latitude, location.longitude)
    lat.append(location.latitude)
    lon.append(location.longitude)

x_range = Range1d()
y_range = Range1d()

map_options = GMapOptions(
    lat=lat[0],
    lng=lon[0],
    zoom=15,
    map_type="hybrid"
)

plot = GMapPlot(
    x_range=x_range,
    y_range=y_range,
    map_options=map_options,
    title="Geocoders"
)

source = ColumnDataSource(
    data=dict(
        lat=lat,
        lon=lon,
        fill=['orange', 'blue', 'green', 'yellow', 'red', 'black'],
        ID_list=['Nominatim', 'ArcGIS', 'Google', 'Yandex', 'OpenMapQuest', 'Geocoder.us']
    )
)

circle = Circle(
    x="lon",
    y="lat",
    size=15,
    fill_color="fill",
    line_color="black"
)
plot.add_glyph(source, circle)

pan = PanTool()
wheel_zoom = WheelZoomTool()
box_select = BoxSelectTool()
hover = HoverTool()

hover.tooltips = [
    ("Geocoder", "@ID_list"),
]

plot.add_tools(pan, wheel_zoom, box_select, hover)

xaxis = LinearAxis(
    axis_label="Latitude",
    major_tick_in=0,
    formatter=NumeralTickFormatter(format="0.000")
)
plot.add_layout(xaxis, 'below')

yaxis = LinearAxis(
    axis_label="Longitude",
    major_tick_in=0,
    formatter=PrintfTickFormatter(format="%.3f")
)
plot.add_layout(yaxis, 'left')

overlay = BoxSelectionOverlay(tool=box_select)
plot.add_layout(overlay)

doc = Document()
doc.add(plot)

if __name__ == "__main__":
    filename = "maps.html"
    with open(filename, "w") as f:
        f.write(file_html(doc, INLINE, "Google Maps Example"))
    print ("Wrote %s" % filename)
    view(filename)